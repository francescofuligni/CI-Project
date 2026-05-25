# Computational Imaging Report

LaTeX report for the Group B sparse-view CT project.

## Build

From the repository root:

```bash
make report
```

From this `report/` directory:

```bash
make
```

Both commands compile `main.tex` in a temporary directory and write the final PDF to:

```text
../docs/report.pdf
```

The temporary LaTeX build directory is:

```text
/private/tmp/ci-project-report-build
```

Clean it with:

```bash
make clean-report
```

from the repository root, or:

```bash
make clean
```

from this directory.

## Files

- `main.tex`: report source.
- `references.bib`: bibliography entries used by the report.
- `figures/`: optional report-local figures.

The report also searches for generated notebook figures under:

```text
../outputs/tpv/
../outputs/unet/
../outputs/diffpir/
../outputs/comparison/
```

Missing figures are rendered as placeholders, so the PDF remains compilable while experiments are still incomplete.
