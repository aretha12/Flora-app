"""
Tambah foto bunga yang lebih beragam (latar & warna) dari dataset publik.

Sumber:
  - Oxford 102 Flowers (VGG, Univ. Oxford): kelas rose, hibiscus, petunia, marigold
  - TensorFlow flower_photos: kelas roses (mawar berbagai warna, foto dari internet)
Chrysanthemum TIDAK ada di kedua dataset -> tambahkan manual (lihat pesan di akhir).

Hasil: folder resized_data_plus/ = isi resized_data/ + foto tambahan (awalan "extra_"),
semua 128x128. Lalu di notebook ganti DATA_DIR = "./resized_data_plus".

Jalankan:  python tambah_data.py
"""
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from torchvision import datasets

ROOT = Path(__file__).parent
SRC = ROOT / "resized_data"
OUT = ROOT / "resized_data_plus"
CACHE = ROOT / "_download"
SIZE = (128, 128)
MAX_EXTRA_PER_CLASS = 200   # batasi supaya kelas tetap seimbang
random.seed(42)

# indeks kelas Oxford 102 (0-based, urutan resmi) -> nama folder di dataset kamu
OXFORD = {73: "Rose", 82: "Hibiscus", 50: "Petuniya", 46: "Marigold"}
TF_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"

CACHE.mkdir(exist_ok=True)
pool = {c: [] for c in ["Chrysanthemum", "Hibiscus", "Marigold", "Petuniya", "Rose"]}

# 1) Oxford 102 Flowers (diunduh otomatis lewat torchvision, ~330 MB)
print("Mengunduh Oxford 102 Flowers ...")
for split in ["train", "val", "test"]:
    ds = datasets.Flowers102(CACHE, split=split, download=True)
    for path, label in zip(ds._image_files, ds._labels):
        if label in OXFORD:
            pool[OXFORD[label]].append(Path(path))

# 2) TensorFlow flower_photos -> hanya folder roses (~220 MB)
tgz = CACHE / "flower_photos.tgz"
if not tgz.exists():
    print("Mengunduh TensorFlow flower_photos ...")
    urllib.request.urlretrieve(TF_URL, tgz)
tf_dir = CACHE / "flower_photos"
if not tf_dir.exists():
    with tarfile.open(tgz) as t:
        t.extractall(CACHE)
pool["Rose"] += sorted((tf_dir / "roses").glob("*.jpg"))

# 3) Salin dataset asli, lalu tambahkan foto ekstra (sudah di-resize)
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
preview = []
for cls, files in pool.items():
    random.shuffle(files)
    picked = files[:MAX_EXTRA_PER_CLASS]
    for i, f in enumerate(picked):
        img = Image.open(f).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)
        img.save(OUT / cls / f"extra_{i:03d}.jpg", quality=90)
        if i < 8:
            preview.append(img)
    total = len(list((OUT / cls).iterdir()))
    print(f"{cls:14s} +{len(picked):3d} foto tambahan  -> total {total}")

# 4) Gambar contoh supaya kamu bisa cek labelnya benar
if preview:
    cols = 8
    rows = (len(preview) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 128, rows * 128), "white")
    for i, im in enumerate(preview):
        sheet.paste(im, ((i % cols) * 128, (i // cols) * 128))
    sheet.save(OUT / "_contoh_tambahan.jpg")
    print("\nCek gambar contoh: resized_data_plus/_contoh_tambahan.jpg")
    print("(baris = Hibiscus, Marigold, Petuniya, Rose - pastikan isinya sesuai)")

print("\nChrysanthemum belum ditambah: cari 50-100 foto krisan KUNING/UNGU/PINK")
print("(bukan putih) dengan latar beragam, simpan ke resized_data_plus/Chrysanthemum/.")
print('Lalu di notebook ubah: DATA_DIR = "./resized_data_plus"')
