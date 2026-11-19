"""Arsitektur & utilitas model — disalin dari notebook cnn_dl_improved.ipynb."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

IMG_SIZE = 128
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
DEFAULT_CLASSES = ["Chrysanthemum", "Hibiscus", "Marigold", "Petuniya", "Rose"]


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        linear_input_size = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return F.relu(self.bn3(self.conv3(x)))  # peta fitur terakhir (untuk Grad-CAM)

    def head(self, fmap):
        x = self.pool(fmap)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

    def forward(self, x):
        return self.head(self.features(x))


transform_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_checkpoint(source, device="cpu"):
    """Terima path atau file-like. Mendukung state_dict murni maupun dict
    {'state_dict', 'class_names', 'history', ...}."""
    ckpt = torch.load(source, map_location=device, weights_only=False)
    meta = {}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        state = ckpt["state_dict"]
    else:
        state = ckpt
    class_names = meta.get("class_names", DEFAULT_CLASSES)
    model = ImprovedCNN(num_classes=len(class_names))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, class_names, meta


@torch.no_grad()
def predict(model, img: Image.Image, tta=False, device="cpu"):
    x = transform_eval(img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    if tta:  # rata-rata dengan versi cermin horizontal
        logits = (logits + model(torch.flip(x, dims=[3]))) / 2
    return F.softmax(logits, dim=1)[0].cpu().numpy()


def grad_cam(model, img: Image.Image, class_idx: int, device="cpu"):
    """Peta panas Grad-CAM dari blok konvolusi terakhir, ukuran sama dengan gambar asli."""
    model.eval()
    x = transform_eval(img.convert("RGB")).unsqueeze(0).to(device)
    fmap = model.features(x)
    fmap.retain_grad()
    out = model.head(fmap)
    model.zero_grad()
    out[0, class_idx].backward()
    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1)).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BICUBIC)
    return np.asarray(cam_img) / 255.0


def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha=0.5):
    import matplotlib
    base = np.asarray(img.convert("RGB")).astype(float) / 255
    heat = matplotlib.colormaps["magma"](cam)[..., :3]
    mixed = (1 - alpha * cam[..., None]) * base + alpha * cam[..., None] * heat
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 1) * 255))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
