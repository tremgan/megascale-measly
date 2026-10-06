"""Load the Tsuboyama et al. (2023) mega-scale stability dataset.

This is measly's proof-of-concept loader, not part of the tool's core: it turns
one protein domain into the generic record schema (variant sequence, reference
sequence, scalar score) that everything downstream consumes.

Source: https://doi.org/10.5281/zenodo.7992926 (CC-BY-4.0)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from megascale_measly.zipfetch import extract_member

ARCHIVE_URL = "https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip?download=1"
MEMBER = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
DEFAULT_PATH = Path("data/Tsuboyama2023_Dataset2_Dataset3_20230416.csv")

# Only the columns we need; the full table is 37 columns and 698 MB.
COLUMNS = ["name", "aa_seq", "mut_type", "WT_name", "ddG_ML", "deltaG_95CI"]


def download(dest: Path = DEFAULT_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    return extract_member(ARCHIVE_URL, MEMBER, dest)


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the table, coercing the two numeric columns that carry '-' placeholders."""
    df = pd.read_csv(path, usecols=COLUMNS, low_memory=False)
    for col in ("ddG_ML", "deltaG_95CI"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def single_substitutions(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are a single amino-acid substitution, e.g. mut_type 'S10A'.

    Excludes the wild type, insertions, deletions and multi-mutants.
    """
    return df[df["mut_type"].str.fullmatch(r"[A-Z]\d+[A-Z]", na=False)]


def wildtype_sequences(df: pd.DataFrame) -> dict[str, str]:
    wt = df[df["mut_type"] == "wt"]
    return dict(zip(wt["WT_name"], wt["aa_seq"]))


def quality_filter(df: pd.DataFrame, max_ci: float) -> pd.DataFrame:
    """Keep rows with a usable ddG and a tight enough confidence interval.

    ddG_ML is blank where the measurement fell outside the assay's dynamic range,
    so dropping those is a censoring decision, not just missing-data cleanup.
    """
    return df[df["ddG_ML"].notna() & (df["deltaG_95CI"] <= max_ci)]


def to_records(df: pd.DataFrame, wt_seqs: dict[str, str],
               score_name: str = "ddG_kcal_per_mol") -> list[dict]:
    return [
        {
            "seq": row.aa_seq,
            "ref_seq": wt_seqs[row.WT_name],
            "score": float(row.ddG_ML),
            "score_name": score_name,
            "group": row.WT_name,
        }
        for row in df.itertuples()
        if row.WT_name in wt_seqs
    ]


def write_records(records: list[dict], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(records, indent=1))
    return dest


def summary(df: pd.DataFrame, max_ci: float = 0.5) -> str:
    """Evidence for choosing a proof-of-concept domain and a QC threshold."""
    wt_seqs = wildtype_sequences(df)
    subs = single_substitutions(df)
    keep = quality_filter(subs, max_ci)
    per_domain = keep.groupby("WT_name").size()
    lengths = pd.Series({name: len(seq) for name, seq in wt_seqs.items()})

    lines = [
        f"rows in table:            {len(df):>8,}",
        f"single substitutions:     {len(subs):>8,}",
        f"  ddG_ML present:         {subs['ddG_ML'].notna().sum():>8,}"
        f"  ({subs['ddG_ML'].isna().mean():.1%} censored)",
        f"  surviving CI <= {max_ci}:   {len(keep):>8,}",
        "",
        f"domains with a wt sequence:   {len(wt_seqs):>4}",
        f"domains with usable singles:  {per_domain.size:>4}",
        f"wt length: {lengths.min()}-{lengths.max()} residues (median {lengths.median():.0f})",
        "",
        "usable singles per domain:",
        f"  min {per_domain.min()}  q25 {per_domain.quantile(.25):.0f}  "
        f"median {per_domain.median():.0f}  q75 {per_domain.quantile(.75):.0f}  "
        f"max {per_domain.max()}",
        "",
        "score (ddG_ML, kcal/mol; positive = stabilising):",
        f"  mean {keep['ddG_ML'].mean():.2f}  sd {keep['ddG_ML'].std():.2f}  "
        f"p05 {keep['ddG_ML'].quantile(.05):.2f}  median {keep['ddG_ML'].median():.2f}  "
        f"p95 {keep['ddG_ML'].quantile(.95):.2f}",
        "",
        "largest domains (usable singles / theoretical complete scan):",
    ]
    for name, count in per_domain.sort_values(ascending=False).head(8).items():
        scan = 19 * len(wt_seqs.get(name, ""))
        coverage = f"{count / scan:.0%}" if scan else "?"
        lines.append(f"  {name:<18} {count:>5} / {scan:>5}  ({coverage})")
    return "\n".join(lines)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "summary", "export"])
    parser.add_argument("--domain", help="WT_name to export, e.g. 1UFM.pdb")
    parser.add_argument("--max-ci", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("data/records.json"))
    args = parser.parse_args()

    if args.command == "download":
        print(download())
        return

    df = load()
    if args.command == "summary":
        print(summary(df, args.max_ci))
        return

    if not args.domain:
        parser.error("export needs --domain")
    wt_seqs = wildtype_sequences(df)
    keep = quality_filter(single_substitutions(df), args.max_ci)
    keep = keep[keep["WT_name"] == args.domain]
    if keep.empty:
        parser.error(f"no usable variants for {args.domain!r}")
    records = to_records(keep, wt_seqs)
    write_records(records, args.out)
    print(f"{len(records)} records -> {args.out}")


if __name__ == "__main__":
    _main()
