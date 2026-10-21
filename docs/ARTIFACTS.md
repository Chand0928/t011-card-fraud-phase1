# Artifact sources and rebuild notes

The main experiment uses only the Python requirements in the root file. Run `python scripts/run_phase1.py` to reproduce analytical tables and charts.

The notebook source builder is `scripts/build_notebook.py`. Rebuilding it clears the existing outputs by design. `scripts/execute_notebook.py` then runs each code cell sequentially with IPython and saves actual tables, text and embedded PNG outputs. This runner avoids needing a local notebook server. The resulting `.ipynb` also opens normally in Jupyter or VS Code.

To rebuild the HTML study copy:

```bash
python scripts/build_notebook.py
python scripts/execute_notebook.py
jupyter nbconvert --to html notebooks/phase1_gmm.ipynb --output phase1_notebook.html --output-dir output
```

The LaTeX report rebuild commands are in `report/README.md`. The official IEEE class comes from [CTAN](https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran.cls) and retains its original header. Final report source is `report/phase1_report.tex`.

The presentation is directly editable in PowerPoint-compatible software. Its preparation script, `scripts/build_presentation.mjs`, uses the supplied Codex `@oai/artifact-tool` runtime. It is an optional authoring tool, not a dependency for running or studying the GMM project. Its data generator uses `results/*.csv` and training distributions. You can edit the delivered PPTX directly without that runtime.

The native slide charts contain embedded workbooks with literal chart values. Their values use ten significant digits to stay within Excel's precision limit. The full analytical precision remains in the result CSV files. The deck's source notes identify the result tables and literature behind the claims.

An independent plot image is available in PNG and vector PDF for each of the 12 analytical figures. The presentation uses five separate native editable charts and four native tables rather than screenshots of those figures.
