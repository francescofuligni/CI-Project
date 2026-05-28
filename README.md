# CI-Project

This repository contains the materials for the Computational Imaging Group B project on sparse-view CT reconstruction with the Mayo dataset.

The implemented methods are:

1. Total p-Variation regularization (TpV);
2. a supervised ResUNet-style image-domain post-processor;
3. DiffPIR adapted to the sparse-view CT setting.

This repository is intended primarily for consultation. The notebooks are written for Google Colab and Google Drive, not for direct local execution from the cloned repository. To reproduce the experiments, upload the notebooks and the required data/support files to Google Drive, open the notebooks in Colab, adjust the absolute paths in the first cells if needed, and run them there.

## How to Use the Materials

Use Google Colab as the execution environment.

1. Upload the notebooks in `notebooks/` to Google Drive.
2. Open each notebook from Google Drive with Colab.
3. Use a GPU runtime for `02_ResUnet_reconstruction.ipynb` and `03_DiffPir_reconstruction.ipynb`.
4. Before running, check the first setup cells and adapt the path constants to your Drive layout.
5. Run the notebooks in the order described below.

The notebooks contain absolute Colab/Drive paths. If your Drive folder names differ, update those constants before executing any cell that loads data, imports `IPPy`, or saves checkpoints/outputs.

## Main Drive Layout for Notebooks 00, 01, and 02

The notebooks `00_data_and_degradation.ipynb`, `01_TpV_reconstruction.ipynb`, and `02_ResUnet_reconstruction.ipynb` assume this project root:

```text
/content/drive/MyDrive/LM_INFORMATICA/COMPUTATIONAL_IMAGING/
```

They expect the following logical structure under that root:

```text
COMPUTATIONAL_IMAGING/
├── IPPy/
├── Mayo2/
│   ├── train/<patient>/*.png
│   ├── val/<patient>/*.png
│   └── test/<patient>/*.png
├── processed2/
├── weights/
│   └── unet/
└── outputs/
    ├── tpv/
    ├── unet/
    └── comparison/
```

The `IPPy/` directory must be available at the project root because these notebooks append `PROJECT_ROOT` to `sys.path` and import from `IPPy`.

The raw Mayo images must be organized by split and patient folder. The expected input pattern is:

```text
Mayo2/<split>/<patient>/*.png
```

where `<split>` is one of `train`, `val`, or `test`.

## Final Comparison Drive Layout

`04_results_comparison.ipynb` currently uses a different Drive root:

```text
/content/drive/MyDrive/COMPUTATIONAL_IMAGING/
```

It expects the already generated processed data, TpV parameters, ResUNet checkpoints, and comparison output directory under:

```text
COMPUTATIONAL_IMAGING/
├── IPPy/
├── processed2/
├── outputs/
│   ├── tpv/
│   │   └── tpv_params.json
│   └── comparison/
└── weights/
    └── unet/
        ├── resunet_generalized_best.pt
        └── resunet_generalized_latest.pt
```

If you run notebooks `00`, `01`, and `02` under the `LM_INFORMATICA/COMPUTATIONAL_IMAGING` root, either mirror those generated files into `/content/drive/MyDrive/COMPUTATIONAL_IMAGING/` before running notebook `04`, or edit `PROJECT_ROOT` in notebook `04` to the same root used by notebooks `00`, `01`, and `02`.

## DiffPIR Drive Layout

`03_DiffPir_reconstruction.ipynb` currently uses a different Drive root:

```text
/content/drive/MyDrive/COMPUTATIONAL_IMAGING/
```

It expects the Mayo dataset archive and DiffPIR weights/checkpoints under:

```text
COMPUTATIONAL_IMAGING/
├── Mayo2.zip
└── weights/
    ├── DiffPir.ckpt
    ├── DiffPir.pth
    ├── DiffPir_raw.pth
    └── DiffPir_best.pth
```

The notebook unzips the dataset locally inside the Colab runtime:

```text
/content/Mayo/Mayo2/{train,val,test}
```

It also clones IPPy at runtime from:

```text
https://github.com/NicolasCola7/IPPy.git
```

If you want all notebooks to use a single Drive root, manually align the path variables before execution. In particular, check `PROJECT_ROOT`, `drive_dir`, `dataset_dir`, and all weight/checkpoint paths.

## Required Dependencies

For notebooks `00`, `01`, `02`, and `04`, the minimal runtime dependencies are:

```text
astra-toolbox
numpy
torch
matplotlib
tqdm
```

They also require the course/project `IPPy/` sources to be available from the configured `PROJECT_ROOT`.

`03_DiffPir_reconstruction.ipynb` additionally installs or uses:

```text
torchvision
numba
scikit-image
Pillow
cupy-cuda12x
```

Colab already provides many common packages, but the notebooks include install cells where needed. Keep those cells or update them according to the runtime you use.

## Required Files

To run everything from scratch, you need:

```text
Mayo2/train/<patient>/*.png
Mayo2/val/<patient>/*.png
Mayo2/test/<patient>/*.png
IPPy/
```

To skip the data-preparation notebook, you need the processed data contract:

```text
processed2/manifest.json
processed2/train/*.pt
processed2/val/*.pt
processed2/test/*.pt
```

To run the final TpV/ResUNet comparison in `04_results_comparison.ipynb`, you also need:

```text
outputs/tpv/tpv_params.json
weights/unet/resunet_generalized_best.pt
```

or, if the best checkpoint is not available:

```text
weights/unet/resunet_generalized_latest.pt
```

To evaluate DiffPIR without retraining, you need:

```text
weights/DiffPir_best.pth
```

The trained model weights are not stored in this repository. They can be downloaded from the shared Google Drive folder:

```text
https://drive.google.com/drive/folders/177ArWjtRR0TANeE_H0YReEmOvBxG9UDI?usp=sharing
```

After downloading them, place the files under the paths expected by the notebooks, for example `weights/unet/` for ResUNet checkpoints and `weights/` for DiffPIR checkpoints.

## Notebook Execution Order

Run the notebooks in this order.

### 1. `00_data_and_degradation.ipynb`

Creates the processed sparse-view CT data contract.

It:

- loads Mayo PNG slices from `Mayo2/{train,val,test}`;
- resizes images to `256 x 256`;
- normalizes clean images;
- creates parallel-beam CT measurements for `180`, `90`, `60`, and `45` views;
- adds relative Gaussian noise with level `0.005`;
- saves one `.pt` file per patient under `processed2/`;
- writes `processed2/manifest.json`.

The saved patient files contain:

```python
payload["clean"]                  # [N, 1, 256, 256]
payload["sinograms"]["180"]
payload["sinograms"]["90"]
payload["sinograms"]["60"]
payload["sinograms"]["45"]
payload["source_paths"]
payload["metadata"]
```

All downstream methods should use these saved degraded inputs when the goal is a fair comparison.

### 2. `01_TpV_reconstruction.ipynb`

Runs Total p-Variation reconstruction.

It:

- loads `processed2/manifest.json`;
- loads the first processed test patient for visual evaluation;
- creates one `IPPy.operators.CTProjector` and one `ChambollePockTpVUnconstrained` solver per view count;
- performs a heuristic `lambda` search on a training slice;
- assigns `lmbda = heuristic_lmbda` unless that line is manually disabled;
- reconstructs the central test slice for each view count;
- saves TpV panels and convergence plots under `outputs/tpv/`;
- exports the selected parameters to `outputs/tpv/tpv_params.json`.

Current core parameters in the notebook include:

```text
p = 0.35
maxiter = 250
tolf = 1e-4
tolx = 1e-4
```

### 3. `02_ResUnet_reconstruction.ipynb`

Trains and evaluates the supervised ResUNet-style model.

It:

- loads train/validation/test patient files from `processed2/`;
- computes FBP proxy inputs in memory from the saved sinograms;
- builds `IPPy.models.UNet` with residual down/up blocks;
- trains one generalized model across all view counts;
- saves checkpoints under `weights/unet/`;
- saves training and representative evaluation plots under `outputs/unet/`.

The main checkpoint files are:

```text
weights/unet/resunet_generalized_latest.pt
weights/unet/resunet_generalized_best.pt
```

### 4. `03_DiffPir_reconstruction.ipynb`

Trains or loads the DiffPIR diffusion prior and evaluates it inside the notebook.

It:

- clones IPPy at runtime;
- installs the additional diffusion-model dependencies;
- unzips `Mayo2.zip` into `/content/Mayo/`;
- builds train/validation/test datasets directly from the extracted Mayo PNG files;
- trains or loads the diffusion U-Net;
- saves DiffPIR weights/checkpoints under the configured `weights/` folder;
- runs DiffPIR reconstruction for `180`, `90`, `60`, and `45` views;
- prints average PSNR and SSIM over the test dataset.

The visual and quantitative DiffPIR evaluation is currently handled in this notebook itself.

### 5. `04_results_comparison.ipynb`

Computes the final full-test comparison for TpV and ResUNet.

It:

- loads `processed2/manifest.json`;
- loads `outputs/tpv/tpv_params.json`;
- loads the best available ResUNet checkpoint from `weights/unet/`;
- evaluates TpV and ResUNet on all processed test slices and all view counts;
- writes per-image metrics and summary metrics under `outputs/comparison/`;
- saves representative panels and aggregate PSNR/SSIM plots.

Important: this notebook currently compares TpV and ResUNet. It does not automatically consume DiffPIR results from `03_DiffPir_reconstruction.ipynb`.

## Generated Outputs

Common generated outputs are:

```text
processed2/
outputs/tpv/
outputs/unet/
outputs/diffpir/
outputs/comparison/
weights/unet/
weights/DiffPir*.pth
weights/DiffPir.ckpt
```

These files can be large and are not intended to be managed manually through the repository. Keep them in Drive and treat the repository as the source for notebooks, documentation, and final project materials.

Model weights are available separately at:

```text
https://drive.google.com/drive/folders/177ArWjtRR0TANeE_H0YReEmOvBxG9UDI?usp=sharing
```

## Path Variables to Check Before Running

Before executing a notebook, inspect and adjust these variables if your Drive layout differs:

```text
PROJECT_ROOT
MAYO_DIR
PROCESSED_DIR
OUTPUT_DIR
WEIGHTS_DIR
TPV_PARAMS_PATH
RESUNET_BEST_CHECKPOINT_PATH
RESUNET_LATEST_CHECKPOINT_PATH
drive_dir
dataset_dir
weights_dir
weights_path
best_weights_path
checkpoint_path
```

The exact variable names differ by notebook. They are defined near the beginning of each notebook.

## Reproducibility Notes

- The project task is sparse-view CT reconstruction on Mayo data.
- Images are resized to `256 x 256`.
- The view counts are `180`, `90`, `60`, and `45`.
- The measurement noise level is `0.005`.
- The main metrics are PSNR and SSIM.
- The processed sinograms created by notebook `00` are the shared degraded inputs for TpV and ResUNet.
