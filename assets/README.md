# README figures

The teaser and method overview come from `figure/teaser.pdf` and
`figure/overview.pdf` in the authors' camera-ready source archive,
`NIPS_ZFM_SpectalKV_CR_已中稿_.zip`. The PDFs here are unchanged copies.
Their PNG previews were rendered with Poppler:

```bash
pdftoppm -scale-to 2400 -png -singlefile assets/teaser.pdf assets/teaser
pdftoppm -scale-to 3000 -png -singlefile assets/overview.pdf assets/overview
```

`results_summary.json` contains benchmark means computed from the archive's
`code/results.json`. Each LongBench mean weights its 16 tasks equally; each
RULER mean weights its 13 tasks equally. The overall score is the arithmetic
mean of these two benchmark means, computed before rounding. The fixed-profile
scores use the archive's `layer` results; all other scores use `main`.
These are the paper's reported results, not a new evaluation run.

To regenerate the comparison chart (Matplotlib is only needed for this figure):

```bash
pip install matplotlib
python assets/plot_results.py
```

All bar charts use the full 0–100 score axis. The README tables show two decimal
places; the JSON retains the unrounded benchmark means.
