# Computational Imaging Report

LaTeX report scaffold for the Group B sparse-view CT project.

## Build

From this directory:

```bash
make
```

The working PDF is generated as `build/report.pdf`, then copied to `../docs/report.pdf`.

## Structure

- `main.tex`: report draft and final writing skeleton.
- `references.bib`: bibliography entries used by the report.
- `figures/`: optional local figures. The report also looks for generated notebook figures under `../outputs/...`.

The current text is a realistic draft template. Metric values and final visual panels must be filled after running the reconstruction notebooks.
