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
