# CI-Project

Project repository for the Computational Imaging Group B assignment.

The project studies sparse-view CT reconstruction on the Mayo dataset. The same degraded sinograms are shared by all reconstruction methods, so that the final comparison is based on identical data, geometry, and noise realizations.

## Project Setup

The current setup follows the project trace:

- task: sparse-view CT reconstruction;
- dataset: Mayo CT slices;
- image size: `256 x 256`;
- CT geometry: parallel beam;
- detector size: `256`;
- projection angles: `180`, `90`, `60`, `45`;
- measurement noise: relative Gaussian noise level `0.005`;
- metrics: PSNR and SSIM;
- visual outputs: reconstructions, absolute error maps, and comparison panels.

For a two-student group, the selected methods are:

1. Total p-Variation regularization with `0.1 < p < 0.5`;
2. a supervised end-to-end Generalized ResUNet post-processor;
3. DiffPIR adapted to sparse-view CT.

The hybrid PD-Net method is intentionally excluded, as allowed by the project specifications for groups of two students. In the current repository state, TpV and ResUNet are implemented in notebooks; DiffPIR is documented as the planned third method and still needs its final reconstruction notebook/results.

## Repository Layout

```text
CI-Project/
├── configs/      Reference configuration files.
├── IPPy/         Course library used for operators, solvers, models, and metrics.
├── notebooks/    Main executable project notebooks.
├── outputs/      Lightweight tracked output folders; generated outputs are usually outside Git.
├── report/       LaTeX source and bibliography.
├── docs/         Final exported deliverables, including docs/report.pdf.
├── homeworks/    Course homework material and pretrained homework weights.
└── Makefile      Root command for regenerating docs/report.pdf.
```

Large data, processed tensors, trained weights, and most generated figures are not tracked by Git.

## Expected External Layout

The notebooks are currently written for Google Colab with Google Drive mounted at:

```text
/content/drive/MyDrive/LM_INFORMATICA/COMPUTATIONAL_IMAGING/
```

The expected Drive layout is:

```text
COMPUTATIONAL_IMAGING/
├── Mayo2/
│   ├── train/
│   ├── val/
│   └── test/
├── IPPy/
├── processed2/
├── weights/
│   └── unet/
└── outputs/
    ├── tpv/
    ├── unet/
    └── comparison/
```

The raw Mayo archive distributed with the course may initially contain only `train/` and `test/`. The current executed setup uses a processed split with:

| Split | Patients | Slices |
| --- | ---: | ---: |
| train | 8 | 2585 |
| val | 2 | 721 |
| test | 1 | 327 |

The test patient used in the notebooks is `C081`, and the representative final evaluation uses the central test slice.

For a local non-Colab run, keep the same logical structure next to the repository or adapt the path constants in the notebooks before running them.

## Python Environment

The notebooks are designed for Colab. The first cells install or import the needed runtime components.

Minimal Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The CT projector requires ASTRA:

```bash
pip install astra-toolbox
```

On Colab, the notebooks call:

```python
!pip install astra-toolbox
from google.colab import drive
drive.mount("/content/drive")
```

## Data Generation Contract

Run the data preparation notebook first:

```text
notebooks/00_data_and_degradation.ipynb
```

It loads Mayo slices, resizes and normalizes them, applies the parallel-beam CT forward model, adds measurement noise, and saves one PyTorch file per patient under `processed2/`.

Each saved patient file has this structure:

```python
payload = torch.load(patient_path, map_location="cpu")

clean = payload["clean"]                   # [N, 1, 256, 256]
sinogram_180 = payload["sinograms"]["180"] # [N, 1, 180, 256]
sinogram_90 = payload["sinograms"]["90"]   # [N, 1, 90, 256]
sinogram_60 = payload["sinograms"]["60"]   # [N, 1, 60, 256]
sinogram_45 = payload["sinograms"]["45"]   # [N, 1, 45, 256]

source_paths = payload["source_paths"]
metadata = payload["metadata"]
```

All downstream notebooks must load these tensors instead of regenerating degraded measurements. This is the fairness constraint of the project.

## Reproducing the Experiments

Run the notebooks in this order:

1. `notebooks/00_data_and_degradation.ipynb`
   - Creates `processed2/manifest.json`.
   - Creates one `.pt` file per patient and split.
   - Saves clean images and noisy sinograms for all four view counts.

2. `notebooks/01_TpV_reconstruction.ipynb`
   - Loads the processed test data.
   - Creates one `IPPy.operators.CTProjector` per view count.
   - Runs `IPPy.solvers.ChambollePockTpVUnconstrained`.
   - Current parameters: `lambda = 0.01`, `p = 0.35`, `maxiter = 150`.
   - Saves TpV reconstruction panels and convergence plots.

3. `notebooks/02_ResUnet_reconstruction.ipynb`
   - Loads train/val/test processed patient files.
   - Computes FBP proxy inputs in memory with `IPPy.solvers.FBP`.
   - Trains one generalized `IPPy.nn.models.UNet` across all view counts.
   - Current parameters: `epochs = 20`, `learning_rate = 1e-3`, `batch_size = 8`, `base_channels = 32`, `final_activation = sigmoid`.
   - Saves the latest checkpoint at `weights/unet/resunet_generalized_latest.pt`.

4. `notebooks/04_results_comparison.ipynb`
   - Loads the same central test slice.
   - Recomputes FBP proxy inputs.
   - Runs TpV and the trained ResUNet on the same saved sinograms.
   - Produces final PSNR/SSIM tables and comparison panels.

DiffPIR should be added as the third method using the same `processed2/` data contract and the same `IPPy.operators.CTProjector` geometry.

## Report Generation

The LaTeX source is in `report/main.tex`. To regenerate only the final report PDF inside `docs/`, run from the repository root:

```bash
make report
```

This compiles LaTeX in a temporary build directory:

```text
/private/tmp/ci-project-report-build
```

and writes only the final PDF to:

```text
docs/report.pdf
```

From inside `report/`, the equivalent command is:

```bash
make
```

To remove the temporary LaTeX build directory:

```bash
make clean-report
```

## Reproducibility Notes

- The random seed used in the notebooks is `42`.
- The same noisy sinograms are reused by all methods.
- FBP is used as a ResUNet input proxy, not as a final selected method.
- PSNR and SSIM are computed through `IPPy.utilities.metrics`.
- Generated model weights, processed tensors, and full-size figures should remain outside Git unless explicitly needed for final delivery.
