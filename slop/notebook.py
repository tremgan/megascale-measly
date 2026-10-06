# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "measly>=0.1.0",
#     "numpy>=1.26",
#     "scikit-learn>=1.5",
#     "torch>=2.4",
#     "transformers>=4.44",
#     "peft>=0.13",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import subprocess
    import sys

    import marimo as mo

    # molab puts sidebar uploads beside the notebook; locally this is slop/.
    sys.path.insert(0, str(mo.notebook_dir()))
    try:
        import measly  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "measly"], check=True)

    import pipeline

    return mo, pipeline


@app.cell
def _(mo, pipeline):
    _where = pipeline.device()
    _detail = _where
    if _where == "cuda":
        import torch

        _detail = (f"{torch.cuda.get_device_name(0)}, "
                   f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    mo.md(
        f"""
        # Three capacity rungs, one shared sweep

        Wraps `pipeline.py`. One `analyse` call, so all rungs see identical
        splits — separate calls draw different subsets and are not comparable
        point by point.

        **Running on `{_where}`** ({_detail}).
        """
    )
    return


@app.cell
def _(mo):
    records = mo.ui.text("data/1UFM.json", label="records JSON", full_width=True)
    rungs = mo.ui.multiselect(
        ["1 one-hot -> ridge", "2 frozen ESM-2 -> ridge", "3 ESM-2 + LoRA"],
        value=["1 one-hot -> ridge", "2 frozen ESM-2 -> ridge", "3 ESM-2 + LoRA"],
        label="rungs",
    )
    draws = mo.ui.slider(2, 100, value=100, label="draws", show_value=True)
    steps = mo.ui.slider(25, 500, step=25, value=200, label="LoRA steps/fit",
                         show_value=True)
    batch = mo.ui.slider(4, 64, step=4, value=64, label="LoRA batch",
                         show_value=True)
    rank = mo.ui.slider(1, 32, value=8, label="LoRA rank", show_value=True)
    mo.vstack([records, rungs, mo.hstack([draws, steps, batch, rank])])
    return batch, draws, rank, records, rungs, steps


@app.cell
def _(batch, draws, mo, rungs, steps):
    # Cost is steps x batch per fit and no longer depends on n, so the whole
    # LoRA bill is one multiplication.
    _lora = 8 * draws.value * steps.value * batch.value if "3" in str(rungs.value) else 0
    run = mo.ui.run_button(label="run sweep", kind="success")
    mo.vstack([
        mo.md(f"`{len(rungs.value)} rung(s) x 8 fractions x {draws.value} draws "
              f"= {len(rungs.value) * 8 * draws.value} fits`"
              + (f", LoRA forwards `{_lora:,}`" if _lora else "")),
        run,
    ])
    return (run,)


@app.cell
def _(batch, draws, mo, pipeline, rank, records, run, rungs, steps):
    from pathlib import Path

    mo.stop(
        not run.value,
        mo.md("Set the controls, then **run sweep**. Nothing runs on edit — "
              "the LoRA rung is minutes to hours."),
    )

    # A local run says "data/1UFM.json"; molab puts sidebar uploads somewhere
    # else entirely, so try the usual spots and then say where we looked.
    _name = Path(records.value).name
    _roots = [Path.cwd(), mo.notebook_dir()]
    _tried = [Path(records.value)]
    _tried += [root / sub / _name for root in _roots for sub in (".", "data")]
    _path = next((c for c in _tried if c.exists()), None)

    if _path is None:
        _seen = sorted({str(f) for root in _roots for f in root.glob("*")
                        if not f.name.startswith(".")})[:20]
        mo.stop(True, mo.md(
            f"`{records.value}` not found.\n\n**Looked in:**\n"
            + "\n".join(f"- `{c}`" for c in dict.fromkeys(_tried))
            + "\n\n**Files visible to the kernel:**\n"
            + ("\n".join(f"- `{f}`" for f in _seen) or "- _(none)_")
            + "\n\nPut the path to one of those in the records box above."))

    X, y, wt = pipeline.load(_path)
    _chosen = {r[0] for r in rungs.value}

    models = {}
    if "1" in _chosen:
        models["one-hot -> ridge"] = pipeline.OneHotRidge(wt)
    if "2" in _chosen:
        pipeline.warm_cache(_path, [*X, wt])
        models["frozen ESM-2 -> ridge"] = pipeline.EsmRidge(wt)
    if "3" in _chosen:
        models["ESM-2 + LoRA"] = pipeline.EsmLora(
            wt, steps=steps.value, batch=batch.value, rank=rank.value)

    from measly import analyse

    analysis = analyse(models, X, y, n_draws=draws.value, rng=0)
    return (analysis,)


@app.cell
def _(analysis, mo, pipeline):
    mo.md(f"""
    ```\n{analysis.summary()}\n\n{pipeline.NOTES}\n```
    """)
    return


@app.cell
def _(analysis):
    from measly import plot

    plot(analysis).figure
    return


if __name__ == "__main__":
    app.run()
