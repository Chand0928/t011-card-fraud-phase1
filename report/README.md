# Edit or rebuild the report

`Phase1_Report.tex` is the report source. `IEEEtran.cls` is the official IEEE LaTeX class, with its original license header preserved.

If you only need to read the report, open `../output/Phase1_Report.pdf`.

To rebuild it, run these commands from this `report` folder with a LaTeX installation:

```bash
pdflatex -output-directory=../output Phase1_Report.tex
pdflatex -output-directory=../output Phase1_Report.tex
```

The second run resolves citations and figure references. Keep the adjacent `figures` folder when compiling. All numbers and charts come from the five notebooks. `../FILE_MAP.md` identifies each producing notebook. If you change the experiment, update the report and slides to match the new run.

For Overleaf, upload `../output/LaTeX_Report_Source.zip`, set `main.tex` as the main document, and compile with pdfLaTeX. This ZIP includes the class and the report figures.
