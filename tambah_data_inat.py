"""
Tambah foto bunga dalam jumlah besar dari iNaturalist (foto warga dari seluruh dunia,
lisensi CC: cc0 / cc-by / cc-by-nc) + semua foto relevan dari Oxford 102 & TF flower_photos.

Hasil:
  resized_data_plus/<Kelas>/       data latih = data asli + tambahan (128x128)
  external_test/<Kelas>/           ±10% foto iNaturalist yang TIDAK dipakai latihan,
                                   untuk menguji model pada foto "dunia nyata"
  _download/inat_<Kelas>/          foto iNaturalist ukuran asli (medium)
  resized_data_plus/_contoh_<Kelas>.jpg  contoh untuk cek label

Jalankan:  python tambah_data_inat.py   (butuh internet, ±10-20 menit)
"""
import hashlib
import json
import random
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT / "resized_data"
OUT = ROOT / "resized_data_plus"
EXT = ROOT / "external_test"
CACHE = ROOT / "_download"
SIZE = (128, 128)
random.seed(42)

# Nama ilmiah di iNaturalist untuk tiap kelas, dan berapa foto yang diambil
INAT = {
    "Chrysanthemum": (["Chrysanthemum morifolium"], 700),  # hanya krisan budidaya (marga lama ikut memuat daisy)
    "Hibiscus":      (["Hibiscus rosa-sinensis"], 450),
    "Marigold":      (["Tagetes erecta", "Tagetes patula"], 350),
    "Petuniya":      (["Petunia"], 250),
    "Rose":          (["Rosa chinensis", "Rosa"], 300),
}
# Target jumlah foto TAMBAHAN per kelas (iNat + Oxford + TF), supaya kelas seimbang
TARGET_EXTRA = {"Chrysanthemum": 630, "Hibiscus": 500, "Marigold": 400, "Petuniya": 450, "Rose": 650}
EXTERNAL_FRAC = 0.10
OXFORD = {73: "Rose", 82: "Hibiscus", 50: "Petuniya", 46: "Marigold"}
UA = {"User-Agent": "flora-student-project/1.0 (tugas kuliah deep learning)"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def inat_photo_urls(names, want):
    urls, seen = [], set()
    for name in names:
        page = 1
        while len(urls) < want and page <= 10:
            q = urllib.parse.urlencode({
                "taxon_name": name, "photos": "true", "per_page": 200, "page": page,
                "photo_license": "cc0,cc-by,cc-by-nc", "order_by": "votes",
                # hanya observasi yang ditandai "sedang berbunga" (anotasi Flowers)
                "term_id": 12, "term_value_id": 13,
            })
            try:
                data = get_json(f"https://api.inaturalist.org/v1/observations?{q}")
            except Exception as e:  # noqa: BLE001
                print(f"   gagal ambil {name} hal {page}: {e}"); break
            res = data.get("results", [])
            if not res:
                break
            for obs in res:
                ph = (obs.get("photos") or [None])[0]
                if ph and ph.get("url") and obs["id"] not in seen:
                    seen.add(obs["id"])
                    urls.append(ph["url"].replace("square.", "medium."))
            page += 1
            time.sleep(1.1)  # sopan ke server iNaturalist
        if len(urls) >= want:
            break
    return urls[:want]


def download(args):
    url, path = args
    if path.exists():
        return path
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
        Image.open(path).verify()
        return path
    except Exception:  # noqa: BLE001
        path.unlink(missing_ok=True)
        return None


def small(path):
    return Image.open(path).convert("RGB").resize(SIZE, Image.Resampling.LANCZOS)


CACHE.mkdir(exist_ok=True)

# 1) iNaturalist
inat = {}
for cls, (names, want) in INAT.items():
    print(f"[iNaturalist] {cls}: mencari {want} foto ...")
    urls = inat_photo_urls(names, want)
    d = CACHE / (f"inatflower_{cls}" + ("_morifolium" if cls == "Chrysanthemum" else "")); d.mkdir(exist_ok=True)
    jobs = [(u, d / f"{i:04d}.jpg") for i, u in enumerate(urls)]
    with ThreadPoolExecutor(8) as ex:
        got = [p for p in ex.map(download, jobs) if p]
    inat[cls] = got
    print(f"   terunduh {len(got)} foto")

# 2) Oxford 102 & TF flower_photos (memakai unduhan sebelumnya bila sudah ada)
oxford = {c: [] for c in INAT}
try:
    from torchvision import datasets
    for split in ["train", "val", "test"]:
        ds = datasets.Flowers102(CACHE, split=split, download=True)
        for p, lab in zip(ds._image_files, ds._labels):
            if lab in OXFORD:
                oxford[OXFORD[lab]].append(Path(p))
except Exception as e:  # noqa: BLE001
    print("Oxford 102 dilewati:", e)
tf_roses = sorted((CACHE / "flower_photos" / "roses").glob("*.jpg"))

# 3) Susun dataset
for d in (OUT, EXT):
    if d.exists():
        shutil.rmtree(d)
shutil.copytree(SRC, OUT)
hashes = set()
for f in OUT.rglob("*.jpg"):
    hashes.add(hashlib.md5(f.read_bytes()).hexdigest())

for cls in INAT:
    items = list(inat[cls]); random.shuffle(items)
    n_ext = int(len(items) * EXTERNAL_FRAC)
    ext_items, train_inat = items[:n_ext], items[n_ext:]
    (EXT / cls).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(ext_items):
        small(p).save(EXT / cls / f"inat_{i:04d}.jpg", quality=90)

    others = oxford[cls] + (tf_roses if cls == "Rose" else [])
    random.shuffle(others)
    pool = [("inat", p) for p in train_inat] + [("other", p) for p in others]
    random.shuffle(pool)
    added, preview = 0, []
    for src, p in pool:
        if added >= TARGET_EXTRA[cls]:
            break
        try:
            im = small(p)
        except Exception:  # noqa: BLE001
            continue
        h = hashlib.md5(im.tobytes()).hexdigest()
        if h in hashes:
            continue
        hashes.add(h)
        im.save(OUT / cls / f"extra_{src}_{added:04d}.jpg", quality=90)
        if len(preview) < 24:
            preview.append(im)
        added += 1
    total = len(list((OUT / cls).glob("*.jpg")))
    print(f"{cls:14s} +{added:4d} tambahan -> total latih {total:4d} | external_test {n_ext}")
    if preview:
        sheet = Image.new("RGB", (8 * 128, 3 * 128), "white")
        for i, im in enumerate(preview):
            sheet.paste(im, ((i % 8) * 128, (i // 8) * 128))
        sheet.save(OUT / f"_contoh_{cls}.jpg")

print("\nSelesai. Cek resized_data_plus/_contoh_<Kelas>.jpg untuk memastikan label benar.")
