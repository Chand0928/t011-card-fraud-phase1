# Rebuild the IEEE report

`phase1_report.tex` is the editable LaTeX source. `IEEEtran.cls` is the official class downloaded from CTAN, kept with its original license header. It is not a custom imitation of IEEE formatting.

Run these commands **from the project root** after producing the analysis results:

```bash
python scripts/build_report.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output phase1_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output phase1_report.tex
```

The second compile resolves citations and figure references. Figures use the actual result PDFs in `results/figures/`. To use Overleaf, upload this project with the report source, class and figures, and preserve the folder structure. Set the main document to `report/phase1_report.tex` and adjust the relative graphics path if the platform changes the compilation directory.

The report is five pages. Its methods explain the experiment, while the notebook and study guide provide the step-by-step learning material.
