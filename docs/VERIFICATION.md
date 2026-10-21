# Verification record

Prepared on 21 October 2026 for Team T011, Section A, Phase 1.

## Analysis

- Downloaded the real recommended ULB/Worldline dataset using TensorFlow's documented mirror.
- Original file: 284,807 rows, 492 frauds. SHA256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
- One-command analysis ran successfully. The teaching notebook also executed all 22 code cells sequentially using IPython and captured actual rich outputs. It contains 12 embedded chart images and no error outputs.
- All four GMM candidates converged. Minimum training BIC selects K=8. The selected model's test AP is 0.153168, compared with 0.053970 for one Gaussian.
- Default confusion counts are TN=56,500, FP=172, FN=37 and TP=37. Derived precision, recall and FPR agree with the saved result tables and written explanations.
- All six targeted unit tests passed. They cover time boundaries, scaler leakage, excluded features, threshold ties, future FPR drift and confusion counts.

## Artifacts

- IEEE report compiled successfully with the official `IEEEtran.cls`. The final compile has no unresolved references or overfull boxes. All five pages were rendered and visually inspected, including the final reference layout.
- Presentation contains 14 slides, four native tables and five native charts. Its structural, layout, font and embedded chart-workbook checks passed. Every slide was rendered and inspected. The final chart revision uses straight segments instead of curve smoothing.
- A PDF copy of the presentation was generated through LibreOffice. All 14 pages rendered, and the overview and representative full-size charts were inspected. The PPTX remains the editable presentation. We did not test it in Microsoft PowerPoint.
- The HTML notebook contains actual embedded outputs and charts. The source notebook remains runnable in Jupyter or VS Code.

## Boundaries

No autoencoder, hybrid model, production deployment or faculty submission occurred. GitHub publishing uses a private repository. Faculty/team access must be arranged before sharing its link. Individual contribution claims remain unfilled until team members actually review and contribute. Study materials support preparation but cannot establish individual understanding without practice.
