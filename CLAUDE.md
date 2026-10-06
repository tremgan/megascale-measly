# CLAUDE.md

A worked example of [measly](../measly) on the Tsuboyama et al. (2023) mega-scale
stability dataset. measly fits a learning curve — loss against training-set size —
so a team can ask what another round of measurement would buy.

This repo depends on `measly` as a package and MUST NOT reach into its internals.
If something here needs a private name, measly's public interface is missing
something; fix it there.

## The experiment

One domain at a time (`1UFM.pdb` is the working default, ~1000 usable single
substitutions), predict `ddG_ML` from sequence at three capacity rungs:

1. **one-hot → ridge.** `19 x L` features for an `L`-residue domain, so 1000+
   features against ~1000 examples. The cheap baseline, and already
   high-capacity relative to `n`.
2. **frozen ESM-2 → ridge head.** Embeddings computed once, never updated. Same
   head as rung 1, different features, so the comparison isolates the
   representation.
3. **ESM-2 + LoRA.** The representation moves too. The expensive rung.

The question is not which wins at full size. It is which curve is still
descending, and whether the rungs cross — a pilot measured at one size says
little about the rank at the next (see measly's README figure).

Run all three in one `analyse` call so they share the sweep's splits. Separate
calls draw different subsets and are not comparable point by point.

## Commands

```sh
uv sync --extra explore --extra models   # models pulls torch, ~2 GB
uv run python -m megascale_measly.megascale download   # 210 MB over range requests
uv run python -m megascale_measly.megascale summary
uv run marimo edit playground/explore.py --no-token --port 2718
```

Python 3.12 or later, because measly's PEP 695 `type` statements are a
SyntaxError on 3.11. Rungs 2 and 3 are the `models` extra — torch, transformers
and peft — so the loader CLI stays installable without them. ESM-2 is
`facebook/esm2_t33_650M_UR50D` via transformers; LoRA is peft on its attention
projections.

## What measly's constraints mean here

These are not style preferences. Breaking them makes the curve wrong quietly.

**Score MUST be MSE on `ddG_ML`, not Spearman.** The power-law form describes a
mean over per-example terms. A rank statistic over the test set does not
decompose, `curve_fit` still returns an α, and it means nothing. Report Spearman
alongside if you want, never as the curve's score.

**Every rung MUST fully re-initialise in `fit`.** `sweep` refits one instance at
every sample size. This is the trap for rung 3: a torch module that keeps its
weights (or its optimizer state) makes later fits no-ops, every size reports the
smallest size's score, and the curve plots flat — which reads as "already
converged", the exact conclusion measly exists to deliver. Build each rung as an
sklearn-style estimator that constructs the adapter and optimizer inside `fit`.

**`X` MUST be the raw sequences (or row indices), not features.** One `X` goes to
all three rungs, and rung 3 needs sequence. Each estimator featurises inside
`fit`/`predict`. Cache the frozen ESM-2 embeddings keyed by sequence — the sweep
is `fractions x draws` fits, so the same sequence is embedded hundreds of times
otherwise.

**Budget the draws against the LoRA rung.** More draws sharpen α; more fractions
barely do. But the default grid is 8 fractions x 100 draws = 800 fits per model,
which is free for ridge and not for LoRA. Lower `n_draws` for the shared call and
say what it cost in the write-up. Keep at least 5 fractions after `hold_back`:
never fit a 3- or 4-parameter law to 3 points.

**`L∞` is a nuisance parameter.** Do not report any rung's fitted floor as the
achievable ddG error.

## Modelling decisions that are not obvious

**The features are the WT difference, mean-pooled.** Embed the variant, embed
the wild type, subtract, then pool. Pooling first instead would be near-useless:
one substitution in a 70-residue domain barely moves the mean, so every variant
of a domain lands on top of every other and ridge sees almost no signal.

The alternative — the embedding at the mutated position only — is **parked**, not
rejected. Try it if rung 2 lands near rung 1. Do not add it as a configurable
option before then.

**Censoring is not missing data.** 20.3% of single substitutions have no `ddG_ML`,
because the measurement left the assay's dynamic range. The dropped variants skew
toward the most destabilising, so every curve here describes the uncensored
subpopulation. Say so when reporting.

**Split within a domain, and only within a domain.** 239 of 478 `WT_name` values
are scans on a one-mutation background of a parent, so domains are not
independent proteins. Pooling them is a different experiment with a leakage
problem; do not do it by accident.

## Dataset facts

Verified against the real file, and easy to get wrong. Also in the README.

- Positions in `mut_type` are 1-based into `aa_seq`, **not** `aa_seq_full`, which
  wraps the domain in `SAGG...` linkers.
- Positive `ddG_ML` means **stabilising** — the opposite of the usual convention.
- `deltaG_95CI` is a full interval width and maxes out at 0.665, so filtering on
  it removes almost nothing. It is more useful as a weight.
- Replicate sd is 0.075 kcal/mol, which is an empirical noise floor. A curve
  approaching it is done, whatever `L∞` says.
- `data/` is gitignored; the CSV is 698 MB.

## Layout

- `src/megascale_measly/zipfetch.py` — pulls one member out of a remote zip over
  HTTP range requests, so `download` moves 210 MB instead of 1 GB.
- `src/megascale_measly/megascale.py` — load, filter, and reshape to the record
  schema. CLI entry point.
- `playground/explore.py` — marimo notebook, gitignored; the substitution heatmap
  and score distribution per domain.

## Conventions

Sentences MUST be short. Use RFC 2119 keywords — MUST, MUST NOT, SHOULD, MAY —
so a reader can tell a rule from a suggestion.

Comments MUST clarify code that is not self-evident, and MUST NOT narrate what
the code already says. Docstrings SHOULD give the *why* when the reason is not
local.

Commit subjects MUST be imperative: `add`, `fix`, `drop`. A body is for a fact
the diff cannot show — a measurement, a rejected alternative, a bug it prevents.
