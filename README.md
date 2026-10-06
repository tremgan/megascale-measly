# megascale-measly

Worked example of the [measly](../measly) workflow on the Tsuboyama et al. (2023)
mega-scale protein folding stability dataset.

This repo depends on `measly` as a package. It must never reach into measly's
internals — if something here needs a private name, that is a signal measly's
public interface is missing something.

## Data

Source: [Zenodo 7992926](https://doi.org/10.5281/zenodo.7992926), CC-BY-4.0.
`zipfetch` pulls the single 210 MB table out of the 1 GB archive over HTTP range
requests rather than downloading the whole thing.

    python -m megascale_measly.megascale download
    python -m megascale_measly.megascale summary
    python -m megascale_measly.megascale export --domain 1UFM.pdb --out data/1UFM.json

`data/` is gitignored; the CSV is 698 MB.

## Dataset notes

Verified against the real file, and easy to get wrong:

- Positions in `mut_type` are 1-based into `aa_seq`, **not** `aa_seq_full`, which
  wraps the domain in `SAGG...` linkers.
- Positive `ddG_ML` means **stabilising** — the opposite of the usual convention.
- 20.3% of single substitutions have no `ddG_ML` at all. That is dynamic-range
  censoring, so the missing variants skew toward the most destabilising.
- `deltaG_95CI` is a full interval width and maxes out at 0.665 across the table,
  so filtering on it removes almost nothing. It is more useful as a weight.
- Each domain has five wild-type rows with distinct DNA sequences. About half of
  domains have one of those five on different linker padding; excluding those,
  replicate sd is 0.075 kcal/mol, which is an empirical noise floor.
- 239 of 478 `WT_name` values are scans on a one-mutation background of a parent
  (`1UBQ.pdb_L15S`), so the domains are not independent proteins.

## Exploration

    marimo edit playground/explore.py --no-token --port 2718
