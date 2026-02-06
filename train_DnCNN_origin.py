############################################################
########## train_dncnn.py  (DnCNN-17, Keras-aligned) ##########
############################################################
import os
import math
import glob
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def is_image_file(p):
    p = p.lower()
    return any(p.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])

def imread_gray(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32) / 255.0

def imread_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0

def to_tensor(img_np):
    # img_np: HxW or HxWxC in [0,1], float32
    if img_np.ndim == 2:
        img_np = img_np[None, ...]  # 1xHxW
    else:
        img_np = np.transpose(img_np, (2, 0, 1))  # CxHxW
    return torch.from_numpy(img_np.copy())

def psnr(x, y):
    # x, y in [0,1], torch tensors
    mse = F.mse_loss(x, y, reduction='mean').item()
    if mse == 0:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps2))

# ----------------------------
# Dataset (clean -> noisy on the fly)
# ----------------------------
class CleanImageFolder(Dataset):
    """
    Keras-like pipeline:
      - Random crop (patch_size=40)
      - Aug: random angle in [-30,30] with NEAREST fill + H/V flips
      - Return clean patch in [0,1]; noise added in collate_fn
    """
    def __init__(self, root, gray=True, patch_size=40, repeats=1, rotate_deg=30):
        self.paths = sorted([p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if is_image_file(p)])
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")
        self.gray = gray
        self.patch = patch_size
        self.repeats = max(1, int(repeats))
        self.rotate_deg = float(rotate_deg)

    def __len__(self):
        return len(self.paths) * self.repeats

    def _random_rotate_nearest(self, arr):
        # arr: HxW or HxWxC, numpy float32 in [0,1]
        if self.rotate_deg <= 0:
            return arr
        angle = random.uniform(-self.rotate_deg, self.rotate_deg)
        pil = Image.fromarray((arr * 255.0).astype(np.uint8))
        # Keras fill_mode="nearest" -> use NEAREST, keep size (expand=False)
        pil = pil.rotate(angle, resample=Image.NEAREST, expand=False)
        out = np.asarray(pil).astype(np.float32) / 255.0
        # preserve channel dim if grayscale
        if out.ndim == 2 and arr.ndim == 3:
            out = out[..., None]
        return out

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        img = imread_gray(path) if self.gray else imread_rgb(path)  # [H,W] or [H,W,3]

        H, W = img.shape[:2]
        if H < self.patch or W < self.patch:
            scale = max(self.patch / H, self.patch / W) + 1e-6
            newH, newW = int(math.ceil(H * scale)), int(math.ceil(W * scale))
            imgPIL = Image.fromarray((img * 255.0).astype(np.uint8)).resize((newW, newH), Image.BICUBIC)
            img = np.array(imgPIL, dtype=np.float32) / 255.0
            H, W = img.shape[:2]

        # random crop
        top = random.randint(0, H - self.patch)
        left = random.randint(0, W - self.patch)
        crop = img[top:top+self.patch, left:left+self.patch] if img.ndim == 2 else img[top:top+self.patch, left:left+self.patch, :]

        # augmentation: rotate (nearest), then flips (like Keras)
        crop = self._random_rotate_nearest(crop)
        if random.random() < 0.5:
            crop = np.flipud(crop).copy()
        if random.random() < 0.5:
            crop = np.fliplr(crop).copy()

        clean = to_tensor(crop)  # CxHxW in [0,1]
        return clean

# Collate: add Gaussian noise (fixed or blind)
def collate_with_noise(batch, sigma_min=0, sigma_max=55, fixed_sigma=None):
    clean = torch.stack(batch, dim=0)  # BxCxHxW
    if fixed_sigma is not None:
        sigma = float(fixed_sigma)
    else:
        sigma = random.uniform(float(sigma_min), float(sigma_max))
    sigma_norm = sigma / 255.0
    noise = torch.randn_like(clean) * sigma_norm
    noisy = clean + noise
    sigma_tensor = torch.full((clean.size(0), 1, 1, 1), sigma_norm, dtype=clean.dtype)
    return noisy, clean, sigma_tensor

# ----------------------------
# DnCNN Model (Keras-aligned BN)
# ----------------------------
class DnCNN(nn.Module):
    """
    DnCNN-17 (grayscale) with Kaiming init.
    BN set to eps=1e-3, momentum=0.01 to mimic Keras BN(mom=0.99).
    Predicts noise; return x_hat = y - noise, noise.
    """
    def __init__(self, channels=1, layers=17, features=64, use_bn=True):
        super().__init__()
        assert layers >= 3, "DnCNN needs at least 3 layers"

        mods = []
        # 1) first: Conv + ReLU (bias True)
        mods += [
            nn.Conv2d(channels, features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        # 2) middle: (Conv bias=False) + BN + ReLU, repeated
        for _ in range(layers - 2):
            mods.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            if use_bn:
                mods.append(nn.BatchNorm2d(features, eps=1e-3, momentum=0.01))
            mods.append(nn.ReLU(inplace=True))
        # 3) last: Conv to channels (bias True, linear)
        mods.append(nn.Conv2d(features, channels, 3, padding=1, bias=True))
        self.net = nn.Sequential(*mods)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y):
        noise = self.net(y)
        x_hat = y - noise
        return x_hat, noise

# ----------------------------
# Validation
# ----------------------------
@torch.no_grad()
def validate(model, val_root, device, gray=True, fixed_sigma=25):
    if val_root is None or not os.path.isdir(val_root):
        return None
    paths = sorted([p for p in glob.glob(os.path.join(val_root, "**", "*"), recursive=True) if is_image_file(p)])
    if len(paths) == 0:
        return None
    model.eval()
    psnrs = []
    for p in paths:
        clean_np = imread_gray(p) if gray else imread_rgb(p)
        clean = to_tensor(clean_np).unsqueeze(0).to(device)
        sigma = fixed_sigma / 255.0
        noisy = clean + torch.randn_like(clean) * sigma
        x_hat, _ = model(noisy)
        psnrs.append(psnr(x_hat.clamp(0,1).cpu(), clean.cpu()))
    return sum(psnrs) / len(psnrs)

# ----------------------------
# Training
# ----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_path = os.path.join(
        args.out_dir,
        "_".join(map(str, [args.layers, args.features, args.gray, args.batch_size,
                           args.patch_size, args.repeats, args.fixed_sigma if args.fixed_sigma is not None else 'blind',
                           args.loss]))
    )
    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val") if args.val else None

    train_set = CleanImageFolder(
        root=train_root,
        gray=args.gray,
        patch_size=args.patch_size,      # 40 by default
        repeats=args.repeats,
        rotate_deg=30                    # Keras rotation_range=30
    )

    def _collate(samples):
        return collate_with_noise(
            samples,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            fixed_sigma=args.fixed_sigma
        )

    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=_collate
    )

    channels = 1 if args.gray else 3
    model = DnCNN(channels=channels, layers=args.layers, features=args.features, use_bn=not args.no_bn).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Keras-like decay: dropEvery=5, factor=0.5
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    use_amp = (not args.no_amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 0.5 * MSE to mirror Keras custom loss scale
    if args.loss == "mse":
        base_loss = nn.MSELoss(reduction="mean")
        def criterion(pred, target): return 0.5 * base_loss(pred, target)
    elif args.loss == "l1":
        base_loss = nn.L1Loss(reduction="mean")
        def criterion(pred, target): return 0.5 * base_loss(pred, target)
    else:
        base_loss = CharbonnierLoss(eps=args.charb_eps)
        def criterion(pred, target): return 0.5 * base_loss(pred, target)

    best_psnr = -1.0
    os.makedirs(out_path, exist_ok=True)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for it, (noisy, clean, sigma) in enumerate(loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            target = noisy - clean  # residual target = noise

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, pred_noise = model(noisy)
                loss = criterion(pred_noise, target)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            global_step += 1
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                print(f"Epoch {epoch:03d} | Step {global_step:06d} | Loss {avg_loss:.6f} | LR {opt.param_groups[0]['lr']:.2e}")
                running_loss = 0.0

        sched.step()

        # Validate
        if args.val:
            val_psnr = validate(model, val_root, device, gray=args.gray, fixed_sigma=args.eval_sigma)
            if val_psnr is not None:
                print(f"Epoch {epoch:03d} | Val PSNR (σ={args.eval_sigma}) = {val_psnr:.2f} dB")
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    ckpt_path = os.path.join(out_path, f"dncnn_best.pth")
                    torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "best_psnr": best_psnr,
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"  ✓ Saved best checkpoint to {ckpt_path}")

        if args.save_last:
            last_path = os.path.join(out_path, f"dncnn_last.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "args": vars(args)}, last_path)

    print("Training done.")
    if best_psnr >= 0:
        print(f"Best Val PSNR: {best_psnr:.2f} dB")

# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Train DnCNN (Keras-aligned).")
    p.add_argument("--data_root", type=str, default="data/DnCNN", help="Root containing 'train' and optional 'val' clean images.")
    p.add_argument("--out_dir", type=str, default="checkpoints_dncnn_origin")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1234)

    # Model
    p.add_argument("--layers", type=int, default=17, help="DnCNN depth (17 for gray, 20 for color).")
    p.add_argument("--features", type=int, default=64)
    p.add_argument("--no_bn", action="store_true", help="Disable BatchNorm (not typical for DnCNN).")
    p.add_argument("--gray", action="store_true", help="Use grayscale (1-channel).")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=40, help="Classic DnCNN uses 40x40.")
    p.add_argument("--repeats", type=int, default=320, help="Virtually repeat dataset per epoch.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP).")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "charbonnier"])
    p.add_argument("--charb_eps", type=float, default=1e-3)

    # Noise setup
    p.add_argument("--sigma_min", type=float, default=0.0, help="Min σ (0..255) if blind.")
    p.add_argument("--sigma_max", type=float, default=55.0, help="Max σ (0..255) if blind.")
    p.add_argument("--fixed_sigma", type=float, default=25.0, help="Set σ to fixed value (Keras-like). Set to empty to enable blind.")
    # Validation
    p.add_argument("--val", action="store_true")
    p.add_argument("--eval_sigma", type=float, default=25.0)

    # Saving
    p.add_argument("--save_last", action="store_true")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    print(args)
    train(args)
