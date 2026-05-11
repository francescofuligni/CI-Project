# CI-Project

Project repository for the Computational Imaging Group B assignment.

The project studies Sparse-views CT reconstruction on the Mayo dataset. The required setup follows the official project specifications:

- resize images to 256x256;
- generate sparse-view CT observations with 180, 90, 60, and 45 angles;
- add noise level 0.005;
- compare all methods on the same degraded inputs;
- report PSNR, SSIM, visual comparisons, and discussion.

For a two-student group, this repository is planned around three methods:

1. Total p-Variation regularization with p in {0.1, 0.5};
2. an end-to-end UNet reconstructor;
3. DiffPIR adapted to Sparse-views CT.

The hybrid PD-Net method is intentionally excluded, as allowed by the
specifications for groups of two students.

## Repository Layout

```text
configs/   Experiment and path configuration.
src/       Project code modules.
scripts/   Executable scripts for each pipeline step.
notebooks/ Optional exploration and Colab notebooks.
outputs/   Generated metrics, figures, and reconstructions.
docs/      Project notes, source tracking, and presentation planning.
```

Large data, trained checkpoints, external repositories, and generated outputs are not tracked by Git.

## External Layout

The local project folder is expected to have this structure:

```text
Progetto/
├── CI-Project/      # this repository
├── Mayo/            # original Mayo dataset, outside Git
│   ├── train/
│   └── test/
├── derived/         # generated processed/degraded data, outside Git
├── checkpoints/     # trained model weights, outside Git
└── external/        # optional external code, outside Git
```

The current Mayo dataset is already split into `train/` and `test/`. The test
patient is `C081`; a validation subset will be created from the training set.

All filesystem paths are configured in `configs/paths.yaml`. The code should not
hardcode dataset paths.

## First Check

After placing the dataset in `../Mayo`, verify the structure with:

```bash
python3 scripts/inspect_dataset.py
```

On Colab, after mounting Google Drive and cloning the repository, use:

```bash
python scripts/inspect_dataset.py --paths configs/paths_colab.yaml
```
