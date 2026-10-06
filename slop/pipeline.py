"""Three capacity rungs for one domain, measured on one shared measly sweep.

Rung 1 is one-hot -> ridge, rung 2 a ridge head on frozen ESM-2, rung 3 ESM-2
with LoRA. All three take the same `X` of raw variant sequences and featurise
inside `fit`, so one `analyse` call scores them on identical splits. Separate
calls draw different subsets and are not comparable point by point.

    uv run python slop/pipeline.py --rungs 1,2 --plot data/curve.png
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

AA = "ACDEFGHIKLMNPQRSTVWY"
ESM = "facebook/esm2_t33_650M_UR50D"
ALPHAS = np.logspace(-1, 4, 12)

_POOLED: dict[str, np.ndarray] = {}
_ADAPTED: dict[int, object] = {}


# --- rung 1 -----------------------------------------------------------------

@lru_cache(maxsize=None)
def onehot(seq: str, wt: str) -> np.ndarray:
    """`19 x L` substitution indicator: position x the 19 non-WT residues.

    Cached because the sweep is `fractions x draws` fits, so the same sequence
    is featurised hundreds of times.
    """
    v = np.zeros(19 * len(wt))
    for i, (w, m) in enumerate(zip(wt, seq)):
        if w != m:
            v[i * 19 + AA.replace(w, "").index(m)] = 1.0
    return v


class OneHotRidge:
    """Rung 1. `19 x L` features against ~n examples, so already high-capacity."""

    def __init__(self, wt: str) -> None:
        self.wt = wt
        self.head = None

    def fit(self, X, y):
        self.head = RidgeCV(alphas=ALPHAS).fit(self._features(X), y)
        return self

    def predict(self, X):
        return self.head.predict(self._features(X))

    def _features(self, X):
        return np.stack([onehot(s, self.wt) for s in X])


# --- rung 2 -----------------------------------------------------------------

def device() -> str:
    """Where the backbones run. `mps` first: this repo's default box is a Mac."""
    import torch

    return ("mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def _esm(model_name: str, role: str = "frozen"):
    """Tokeniser and backbone. `role` exists only to key a second instance:
    rung 3 injects adapters into the module it is handed, and rung 2 must keep
    embedding through an untouched one."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_name).eval().to(device())
    return AutoTokenizer.from_pretrained(model_name), model


def embed(seqs, model_name: str = ESM, batch: int = 8) -> None:
    """Fill `_POOLED` for any sequence missing from it."""
    import torch

    tok, model = _esm(model_name)
    todo = [s for s in dict.fromkeys(seqs) if s not in _POOLED]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # The cls/eos tokens stay in the pool. They shift every sequence of one
        # length alike, so the WT subtraction in `_features` removes them.
        for s, v in zip(chunk, ((h * mask).sum(1) / mask.sum(1)).cpu().numpy()):
            _POOLED[s] = v


def pooled(seq: str, model_name: str = ESM) -> np.ndarray:
    if seq not in _POOLED:
        embed([seq], model_name)
    return _POOLED[seq]


class EsmRidge:
    """Rung 2. Rung 1's head on frozen features, so the comparison isolates
    the representation.

    The feature is the WT difference, mean-pooled. Pooling without subtracting
    would be near-useless: one substitution in a 70-residue domain barely moves
    the mean, so every variant would land on top of every other. Subtracting
    after pooling is the same vector as pooling the per-residue difference,
    since the variant and the WT have equal length and so share a mask.
    """

    def __init__(self, wt: str, model_name: str = ESM) -> None:
        self.wt, self.model_name = wt, model_name
        self.head = None

    def fit(self, X, y):
        self.head = RidgeCV(alphas=ALPHAS).fit(self._features(X), y)
        return self

    def predict(self, X):
        return self.head.predict(self._features(X))

    def _features(self, X):
        wt = pooled(self.wt, self.model_name)
        return np.stack([pooled(s, self.model_name) - wt for s in X])


# --- rung 3 -----------------------------------------------------------------

def _wrap(base, config):
    """Attach fresh adapters, stripping whatever the last fit left behind.

    Keyed on the backbone, not on the caller: `_esm` hands the same module to
    every fit and to every `EsmLora` instance, and wrapping an already-wrapped
    module stacks adapters rather than replacing them. Tracking this per
    instance leaves two rungs at different ranks quietly training each other's
    adapters.
    """
    from peft import get_peft_model

    live = _ADAPTED.pop(id(base), None)
    if live is not None:
        live.unload()
        # `unload` strips the modules but leaves the marker that peft warns on.
        base.__dict__.pop("peft_config", None)
    peft = get_peft_model(base, config)
    _ADAPTED[id(base)] = peft
    return peft


class EsmLora:
    """Rung 3. The representation moves too.

    Every trainable part is built inside `fit`. `sweep` refits one instance at
    every sample size, so a module that kept its adapter or its optimizer state
    would report the smallest size's score at every size and plot a flat curve
    — which reads as "already converged", the one conclusion measly must never
    invent.

    The budget is `steps` optimizer updates, not epochs. Epochs tie the amount
    of training to `n`, which is the axis of the learning curve: the smallest
    fraction then trains ~7x less than the largest, and the resulting ramp gets
    read as a steep descent. A fixed step count leaves data as the only thing
    varying along the curve, and makes `batch` a pure throughput knob instead
    of something that silently divides the update count.

    Cost is `steps x batch` per fit, so every fraction now costs the same, and
    the sweep no longer spends most of itself on the largest fractions.
    """

    def __init__(self, wt: str, model_name: str = ESM, *, steps: int = 200,
                 lr: float = 5e-4, batch: int = 8, rank: int = 8) -> None:
        self.wt, self.model_name = wt, model_name
        self.steps, self.lr, self.batch, self.rank = steps, lr, batch, rank
        self.peft = self.head = None

    def fit(self, X, y):
        import torch
        from peft import LoraConfig

        _, base = _esm(self.model_name, "lora")
        self.peft = _wrap(base, LoraConfig(
            r=self.rank, lora_alpha=2 * self.rank, lora_dropout=0.0,
            target_modules=["query", "value"], bias="none"))
        self.head = torch.nn.Linear(base.config.hidden_size, 1).to(base.device)
        trainable = [p for p in self.peft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable + list(self.head.parameters()), lr=self.lr)

        X = np.asarray(X)
        y = torch.as_tensor(np.asarray(y, dtype="float32"), device=base.device)
        self.peft.train()
        for _ in range(self.steps):
            idx = torch.randint(len(X), (min(self.batch, len(X)),))
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(self._forward(X[idx.numpy()]), y[idx])
            loss.backward()
            opt.step()
        self.peft.eval()
        # The sweep prints nothing until it is over, and these fits are minutes
        # each. The final train loss also shows whether `steps` reached a floor.
        print(f"  lora fit n={len(X):<5} steps={self.steps} batch={self.batch} "
              f"train_loss={loss.item():.4f}", file=sys.stderr, flush=True)
        return self

    def predict(self, X):
        import torch

        X = np.asarray(X)
        with torch.no_grad():
            chunks = [self._forward(X[i:i + 32]).cpu().numpy()
                      for i in range(0, len(X), 32)]
        return np.concatenate(chunks)

    def _forward(self, seqs):
        tok, base = _esm(self.model_name, "lora")
        # The WT rides along in every batch. The adapter moves, so its
        # embedding cannot be cached the way rung 2's can.
        enc = tok(list(seqs) + [self.wt], return_tensors="pt",
                  padding=True).to(base.device)
        h = self.peft(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        p = (h * mask).sum(1) / mask.sum(1)
        return self.head(p[:-1] - p[-1]).squeeze(-1)


# --- data and CLI -----------------------------------------------------------

def load(path: Path):
    """Records from `megascale export`, one domain only."""
    records = json.loads(path.read_text())
    refs = {r["ref_seq"] for r in records}
    if len(refs) != 1:
        raise SystemExit(f"{path}: {len(refs)} reference sequences. Split within "
                         "one domain, and only within a domain.")
    wt = refs.pop()
    X = np.array([r["seq"] for r in records])
    if {len(s) for s in X} != {len(wt)}:
        raise SystemExit(f"{path}: not every variant is {len(wt)} residues long.")
    return X, np.array([r["score"] for r in records], dtype=float), wt


def warm_cache(records: Path, seqs, model_name: str = ESM) -> Path:
    """Embed everything up front, through a cache that survives the process.

    `EsmRidge` still embeds on demand, so this is only ever a speed-up. Doing
    it here keeps the misses out of the middle of a sweep.
    """
    cache = records.parent / f"pooled-{model_name.rsplit('/', 1)[-1]}.npz"
    if cache.exists():
        z = np.load(cache)
        _POOLED.update(zip(z["seqs"].tolist(), z["vectors"]))
    embed(seqs, model_name)
    np.savez(cache, seqs=np.array(list(_POOLED)),
             vectors=np.stack(list(_POOLED.values())))
    return cache


NOTES = """Scored as MSE on ddG_ML. Replicate sd is 0.075 kcal/mol, so MSE near 0.0056 is at
the assay's noise floor, whatever a fitted L-infinity says. Do not read any rung's
floor as the achievable ddG error.
20.3% of single substitutions have no ddG_ML and were dropped at export as
dynamic-range censoring. Those skew destabilising, so every curve above
describes the uncensored subpopulation."""


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/1UFM.json"))
    parser.add_argument("--rungs", default="1,2,3", help="comma-separated, e.g. 1,2")
    parser.add_argument("--draws", type=int, help="default 100, or 10 with rung 3")
    parser.add_argument("--model", default=ESM)
    parser.add_argument("--steps", type=int, default=200,
                        help="rung 3 optimizer updates per fit, any size")
    parser.add_argument("--rank", type=int, default=8, help="rung 3 only")
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    from measly import DEFAULT_FRACTIONS, analyse

    if not args.records.exists():
        raise SystemExit(f"{args.records} missing. Run: python -m "
                         f"megascale_measly.megascale export --domain 1UFM.pdb "
                         f"--out {args.records}")
    X, y, wt = load(args.records)
    rungs = args.rungs.split(",")
    models = {}
    if "1" in rungs:
        models["one-hot -> ridge"] = OneHotRidge(wt)
    if "2" in rungs:
        models["frozen ESM-2 -> ridge"] = EsmRidge(wt, args.model)
    if "3" in rungs:
        models["ESM-2 + LoRA"] = EsmLora(wt, args.model, steps=args.steps, rank=args.rank)
    if not models:
        raise SystemExit(f"--rungs {args.rungs!r} selected nothing")
    # More draws sharpen alpha, more fractions barely do. 8 x 100 is free for
    # ridge and not for LoRA.
    draws = args.draws or (10 if "3" in rungs else 100)

    if "2" in rungs:
        warm_cache(args.records, [*X, wt], args.model)

    analysis = analyse(models, X, y, n_draws=draws, rng=0)
    print(analysis.summary(args.factor))
    print(f"\n{len(models)} rung(s) x {len(DEFAULT_FRACTIONS)} fractions x {draws} "
          f"draws = {len(models) * len(DEFAULT_FRACTIONS) * draws} fits.")
    print(NOTES)

    if args.plot:
        import matplotlib.pyplot as plt

        from measly import plot

        plot(analysis, args.factor)
        plt.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"\n{args.plot}")


if __name__ == "__main__":
    _main()
