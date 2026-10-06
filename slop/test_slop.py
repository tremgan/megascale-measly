"""One check per trap: the features, and the refit.

    uv run python slop/test_slop.py
"""

import numpy as np
from pipeline import EsmLora, OneHotRidge, onehot

WT = "ACDEFGHIKL"


def test_onehot():
    assert not onehot(WT, WT).any(), "the WT is the all-zero feature vector"

    seq = WT[:2] + "E" + WT[3:]  # D3E
    v = onehot(seq, WT)
    expected = 2 * 19 + "ACEFGHIKLMNPQRSTVWY".index("E")
    assert v.sum() == 1.0 and v[expected] == 1.0
    assert len(v) == 19 * len(WT)


def test_ridge_refits():
    """A rung that kept state would score every sample size the same."""
    X = np.array([WT[:i] + "W" + WT[i + 1:] for i in range(len(WT))])
    y = np.arange(len(X), dtype=float)

    rung = OneHotRidge(WT)
    first = rung.fit(X, y).predict(X)
    second = rung.fit(X, -y).predict(X)
    assert np.corrcoef(first, second)[0, 1] < -0.9, "fit did not re-initialise"


def test_lora_refits():
    """The expensive trap: adapters stacking, or the backbone drifting.

    Needs torch, peft and a 30 MB download, so it is skipped without them.
    """
    try:
        import torch  # noqa: F401
        from peft import LoraConfig  # noqa: F401

        from pipeline import _esm
        tiny = "facebook/esm2_t6_8M_UR50D"
        _, base = _esm(tiny, "lora")
    except Exception as exc:
        print(f"skipped test_lora_refits: {exc}")
        return

    X = np.array([WT[:i] + "W" + WT[i + 1:] for i in range(6)])
    y = np.arange(len(X), dtype=float)
    frozen = base.embeddings.word_embeddings.weight.detach().clone()

    rung = EsmLora(WT, tiny, steps=2, batch=3)
    rung.fit(X, y)
    after_first = sum("lora_A" in name for name, _ in base.named_modules())
    first = rung.predict(X)

    rung.fit(X, -y)
    after_second = sum("lora_A" in name for name, _ in base.named_modules())

    assert after_first and after_first == after_second, "adapters stacked on refit"

    # A second instance shares the one backbone, so it must strip the first
    # instance's adapters rather than train on top of them.
    other = EsmLora(WT, tiny, steps=2, batch=3, rank=2)
    other.fit(X, y)
    assert sum("lora_A" in n for n, _ in base.named_modules()) == after_first, \
        "a second rung stacked its adapters on the first rung's"
    ranks = {m["default"].out_features for n, m in other.peft.named_modules()
             if n.endswith("lora_A")}
    assert ranks == {2}, f"wrong rank attached: {ranks}"
    assert torch.equal(frozen, base.embeddings.word_embeddings.weight), \
        "LoRA moved the frozen backbone"
    assert not np.allclose(first, rung.predict(X)), "fit did not re-initialise"


if __name__ == "__main__":
    for name, check in sorted(globals().items()):
        if name.startswith("test_"):
            check()
            print(f"ok {name}")
