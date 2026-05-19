# pyrefly: ignore [invalid-syntax]
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from torch import nn
# pyrefly: ignore [missing-import]
from torch.nn import functional as F
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Dataset
# pyrefly: ignore [missing-import]
from torchvision import transforms
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

# pyrefly: ignore [missing-import]
from utils import load_yaml, resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Homework 2.1: End-to-End Reconstruction Before Generative Models."
    )
    parser.add_argument(
        "--paths",
        default="configs/paths.yaml",
        help="Path to the YAML file containing dataset and output paths.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-shape", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--kernel-size", type=int, default=15)
    parser.add_argument("--motion-angle", type=float, default=30.0)
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def add_ippy_to_path(ippy_dir: Path) -> None:
    if ippy_dir.exists():
        sys.path.append(str(ippy_dir))
    if ippy_dir.parent.exists():
        sys.path.append(str(ippy_dir.parent))


def setup_environment(paths: dict[str, str], seed: int):
    ippy_dir = resolve_from_root(paths["ippy_dir"])
    add_ippy_to_path(ippy_dir)

    # pyrefly: ignore [missing-import]
    from IPPy import operators, utilities

    weights_dir = resolve_from_root(paths.get("homework_weights_dir", "../weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = utilities.get_device()
    torch.manual_seed(seed)

    print("Homework 2.1 setup")
    print(f"Working device:    {device}")
    print(f"Weights directory: {weights_dir}")
    print(f"IPPy directory:    {ippy_dir}")

    return operators, utilities, device, weights_dir


# ---------------------------------------------------------------------------
# Part 1: Data Pipeline and Synthetic Measurements
# ---------------------------------------------------------------------------


class MayoDataset(Dataset):
    def __init__(self, data_path, data_shape=256):
        super().__init__()
        self.fname_list = sorted(glob.glob(f"{data_path}/*/*.png"))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((data_shape, data_shape), antialias=True),
            ]
        )

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        image_path = self.fname_list[idx]
        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        return image


def add_gaussian_noise(image: torch.Tensor, noise_level: float) -> torch.Tensor:
    noisy = image + noise_level * torch.randn_like(image)
    return torch.clamp(noisy, 0.0, 1.0)


def build_dataloaders(
    paths: dict[str, str],
    data_shape: int,
    batch_size: int,
    num_workers: int,
):
    train_dataset = MayoDataset(resolve_from_root(paths["train_dir"]), data_shape)
    test_dataset = MayoDataset(resolve_from_root(paths["test_dir"]), data_shape)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def save_clean_corrupted_pair(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_image = clean[0, 0].detach().cpu()
    corrupted_image = corrupted[0, 0].detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(clean_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Clean")
    axes[0].axis("off")
    axes[1].imshow(corrupted_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Motion blur + noise")
    axes[1].axis("off")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part 2: Reconstruction Networks
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    def __init__(self, in_ch, out_ch, n_filters, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=n_filters, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding="same")
        self.conv3 = nn.Conv2d(in_channels=n_filters, out_channels=out_ch, kernel_size=kernel_size, padding="same")

        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        out = self.conv3(h)
        return out


class ResCNN(nn.Module):
    def __init__(self, in_ch, out_ch, n_filters, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=n_filters, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding="same")
        self.conv3 = nn.Conv2d(in_channels=n_filters, out_channels=out_ch, kernel_size=kernel_size, padding="same")

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        out = self.tanh(self.conv3(h))
        return out + x


# ---------------------------------------------------------------------------
# Optional: UNet implementation
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, block_cls=DoubleConv):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = block_cls(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, block_cls=DoubleConv):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = block_cls(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, 2 * base_ch)
        self.enc3 = DownBlock(2 * base_ch, 4 * base_ch)
        self.bottleneck = DownBlock(4 * base_ch, 8 * base_ch)
        self.dec3 = UpBlock(8 * base_ch, 4 * base_ch, 4 * base_ch)
        self.dec2 = UpBlock(4 * base_ch, 2 * base_ch, 2 * base_ch)
        self.dec1 = UpBlock(2 * base_ch, base_ch, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        h = self.bottleneck(s3)
        h = self.dec3(h, s3)
        h = self.dec2(h, s2)
        h = self.dec1(h, s1)
        return self.out_conv(h)



def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_reconstruction_models() -> dict[str, nn.Module]:
    return {
        "simplecnn": SimpleCNN(in_ch=1, out_ch=1, n_filters=32),
        "rescnn": ResCNN(in_ch=1, out_ch=1, n_filters=32),
        "unet": UNet(in_ch=1, out_ch=1, base_ch=32),
    }


def print_model_summary(models: dict[str, nn.Module]) -> None:
    print()
    print("Part 2: Reconstruction Networks")
    for name, model in models.items():
        n_params = count_trainable_parameters(model)
        print(f"{name}: {n_params} trainable parameters")


# ---------------------------------------------------------------------------
# Part 3: Training Procedure
# ---------------------------------------------------------------------------


def train_model(model, train_loader, K, weights_path, device, num_epochs=20, noise_level=0.01, lr=1e-3):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for clean in progress_bar:
            clean = clean.to(device)
            corrupted = add_gaussian_noise(K(clean), noise_level)

            optimizer.zero_grad()
            reconstruction = model(corrupted)
            loss = loss_fn(reconstruction, clean)
            loss.backward()
            optimizer.step()

            batch_size = clean.shape[0]
            running_loss += loss.item() * batch_size
            num_samples += batch_size
            progress_bar.set_postfix(loss=running_loss / num_samples)

        epoch_loss = running_loss / num_samples
        history.append(epoch_loss)

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)

    return history


def save_training_curves(histories: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for model_name, history in histories.items():
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history, marker="o", label=model_name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Training curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_reconstruction_models(
    models: dict[str, nn.Module],
    train_loader: DataLoader,
    K,
    device: torch.device,
    weights_dir: Path,
    figures_dir: Path,
    num_epochs: int,
    noise_level: float,
    lr: float,
) -> dict[str, list[float]]:
    histories = {}

    print()
    print("Part 3: Training Procedure")
    for model_name, model in models.items():
        weights_path = weights_dir / f"hw1-m2_{model_name}.pt"
        print(f"Training {model_name}")
        history = train_model(
            model=model,
            train_loader=train_loader,
            K=K,
            weights_path=weights_path,
            device=device,
            num_epochs=num_epochs,
            noise_level=noise_level,
            lr=lr,
        )
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        histories[model_name] = history
        print(f"Saved and reloaded weights: {weights_path}")

    curves_path = figures_dir / "hw1-m2_training_curves.png"
    save_training_curves(histories, curves_path)
    print(f"Saved training curves: {curves_path}")

    return histories


def main() -> None:
    args = parse_args()
    paths = load_yaml(resolve_from_root(args.paths))
    operators, utilities, device, weights_dir = setup_environment(paths, args.seed)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        paths=paths,
        data_shape=args.data_shape,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    clean = next(iter(train_loader)).to(device)
    K = operators.Blurring(
        img_shape=(args.data_shape, args.data_shape),
        kernel_type="motion",
        kernel_size=args.kernel_size,
        motion_angle=args.motion_angle,
    )
    corrupted = add_gaussian_noise(K(clean), args.noise_level)

    figure_path = resolve_from_root(paths["figures_dir"]) / "hw1-m2_clean_corrupted.png"
    save_clean_corrupted_pair(clean, corrupted, figure_path)

    print()
    print("Part 1: Data Pipeline and Synthetic Measurements")
    print(f"Train images:    {len(train_dataset)}")
    print(f"Test images:     {len(test_dataset)}")
    print(f"Batch shape:     {tuple(clean.shape)}")
    print(f"Clean range:     [{clean.min().item():.4f}, {clean.max().item():.4f}]")
    print(f"Corrupted range: [{corrupted.min().item():.4f}, {corrupted.max().item():.4f}]")
    print(f"Motion blur:     kernel={args.kernel_size}, angle={args.motion_angle}")
    print(f"Noise level:     {args.noise_level}")
    print(f"Saved figure:    {figure_path}")

    models = build_reconstruction_models()
    print_model_summary(models)

    train_reconstruction_models(
        models=models,
        train_loader=train_loader,
        K=K,
        device=device,
        weights_dir=weights_dir,
        figures_dir=resolve_from_root(paths["figures_dir"]),
        num_epochs=args.num_epochs,
        noise_level=args.noise_level,
        lr=args.learning_rate,
    )


if __name__ == "__main__":
    main()
