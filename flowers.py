"""Data katalog bunga + pencarian gambar."""
import base64
import io
import math
from pathlib import Path

from PIL import Image, ImageOps

APP_DIR = Path(__file__).parent

FLOWERS = {
    "Chrysanthemum": {
        "local": "Krisan",
        "latin": "Chrysanthemum morifolium",
        "family": "Asteraceae",
        "color": "#D9A21B",
        "summary": "Bunga potong favorit dengan kelopak berlapis rapat. Di Indonesia banyak dibudidayakan di dataran "
                   "tinggi yang sejuk seperti Cipanas, Bandungan, dan Berastagi.",
        "facts": {"Cahaya": "Matahari penuh, 6 jam+", "Air": "Sedang", "Suhu ideal": "18–26 °C",
                  "Media": "Gembur, pH 6–6,5"},
        "care": {
            "Penyiraman": "Siram saat permukaan tanah mulai kering, langsung ke pangkal batang. Hindari membasahi daun "
                          "di sore hari karena memicu jamur.",
            "Pemupukan": "Pupuk NPK seimbang tiap 2 minggu saat fase tumbuh. Begitu kuncup muncul, ganti ke pupuk "
                         "tinggi kalium agar bunga lebih lebat.",
            "Pemangkasan": "Pucuk dipotong (pinching) saat tanaman setinggi 15 cm supaya cabang lebih banyak. "
                           "Buang bunga yang layu agar energi pindah ke kuncup baru.",
            "Hama & penyakit": "Waspadai kutu daun, thrips, dan penggorok daun. Karat daun dan embun tepung muncul "
                               "di udara lembap, jadi jaga jarak tanam agar sirkulasi udara baik.",
            "Perbanyakan": "Paling mudah dari stek pucuk sepanjang 5–8 cm, ditanam di media lembap sampai berakar "
                           "dalam 2–3 minggu.",
        },
        "tip": "Krisan berbunga ketika malam lebih panjang dari siang. Hindari menaruhnya di dekat lampu yang menyala "
               "sepanjang malam kalau ingin cepat berbunga.",
    },
    "Hibiscus": {
        "local": "Kembang sepatu",
        "latin": "Hibiscus rosa-sinensis",
        "family": "Malvaceae",
        "color": "#C8305F",
        "summary": "Semak tropis berbunga besar dengan benang sari menjulur panjang. Tahan panas dan cocok untuk "
                   "pagar hidup maupun pot di teras.",
        "facts": {"Cahaya": "Matahari penuh", "Air": "Banyak", "Suhu ideal": "24–32 °C",
                  "Media": "Subur, porous"},
        "care": {
            "Penyiraman": "Jaga tanah tetap lembap tapi tidak becek. Di musim kemarau siram setiap hari, terutama "
                          "untuk tanaman dalam pot.",
            "Pemupukan": "Pupuk tinggi kalium (misalnya NPK 10-5-20) tiap 2–3 minggu. Terlalu banyak nitrogen "
                         "membuat daun rimbun tapi bunga sedikit.",
            "Pemangkasan": "Pangkas setelah masa berbunga untuk membentuk tajuk. Bunga muncul di ranting baru, "
                           "jadi pemangkasan rutin justru menambah bunga.",
            "Hama & penyakit": "Kutu putih, kutu kebul, dan kutu daun sering bersembunyi di balik daun. Kuncup yang "
                               "rontok sebelum mekar biasanya tanda kekurangan air atau perubahan tempat mendadak.",
            "Perbanyakan": "Stek batang semi-keras 15–20 cm, buang daun bagian bawah, tanam di media lembap. "
                           "Akar tumbuh dalam 3–4 minggu.",
        },
        "tip": "Satu bunga kembang sepatu hanya mekar sekitar satu hari. Tanaman sehat akan terus membuka kuncup baru "
               "hampir setiap hari.",
    },
    "Marigold": {
        "local": "Gemitir",
        "latin": "Tagetes erecta",
        "family": "Asteraceae",
        "color": "#E8701A",
        "summary": "Bunga oranye-kuning berbentuk bola yang akrab di Bali sebagai bagian canang sari. Mudah ditanam "
                   "dari biji dan berbunga cepat.",
        "facts": {"Cahaya": "Matahari penuh", "Air": "Sedikit–sedang", "Suhu ideal": "20–30 °C",
                  "Media": "Porous, tidak becek"},
        "care": {
            "Penyiraman": "Siram di pangkal batang saat tanah kering. Setelah tanaman kuat, gemitir cukup tahan "
                          "kering; yang lebih berbahaya justru tanah yang terus basah.",
            "Pemupukan": "Cukup pupuk kandang saat awal tanam. Pupuk nitrogen berlebih membuat tanaman tinggi "
                         "berdaun lebat tapi bunganya sedikit.",
            "Pemangkasan": "Petik bunga yang mulai layu (deadheading) agar tanaman terus membentuk kuncup baru "
                           "sepanjang musim.",
            "Hama & penyakit": "Tungau laba-laba di cuaca panas kering, siput saat masih bibit, dan busuk akar jika "
                               "drainase buruk.",
            "Perbanyakan": "Sebar biji di permukaan media, tutup tipis. Berkecambah 5–7 hari dan mulai berbunga "
                           "sekitar 8 minggu setelah semai.",
        },
        "tip": "Akar gemitir dikenal membantu menekan nematoda di tanah, jadi sering ditanam berdampingan dengan "
               "tomat dan cabai.",
    },
    "Petuniya": {
        "local": "Petunia",
        "latin": "Petunia × hybrida",
        "family": "Solanaceae",
        "color": "#7A4CB5",
        "summary": "Bunga terompet berwarna cerah yang cocok untuk pot gantung dan pinggiran taman. Lebih betah di "
                   "udara sejuk, namun tetap bisa tumbuh di dataran rendah.",
        "facts": {"Cahaya": "Matahari pagi 5–6 jam", "Air": "Sedang", "Suhu ideal": "16–28 °C",
                  "Media": "Ringan, banyak kompos"},
        "care": {
            "Penyiraman": "Siram saat 2–3 cm tanah teratas kering. Arahkan air ke tanah, bukan ke bunga, karena "
                          "kelopaknya mudah rusak dan membusuk bila sering basah.",
            "Pemupukan": "Petunia rakus hara: beri pupuk cair seimbang tiap 1–2 minggu selama berbunga.",
            "Pemangkasan": "Buang bunga layu beserta bakal bijinya. Jika batang memanjang dan jarang berbunga, "
                           "pangkas sepertiga panjangnya agar tumbuh rimbun kembali.",
            "Hama & penyakit": "Kutu daun dan ulat penggerek kuncup. Busuk batang muncul bila pot tidak punya "
                               "lubang drainase.",
            "Perbanyakan": "Biji sangat halus: tabur di permukaan tanpa ditutup tanah. Varietas tertentu lebih mudah "
                           "diperbanyak dengan stek pucuk.",
        },
        "tip": "Di dataran rendah yang panas, letakkan petunia di tempat yang teduh pada siang hingga sore hari "
               "supaya bunganya tidak cepat layu.",
    },
    "Rose": {
        "local": "Mawar",
        "latin": "Rosa spp.",
        "family": "Rosaceae",
        "color": "#B8233A",
        "summary": "Bunga klasik dengan kelopak berlapis spiral dan aroma khas. Butuh perhatian lebih, namun "
                   "berbunga sepanjang tahun di iklim tropis bila dirawat rutin.",
        "facts": {"Cahaya": "Matahari pagi 6 jam+", "Air": "Sedang–banyak", "Suhu ideal": "18–28 °C",
                  "Media": "Subur, pH 6–6,5"},
        "care": {
            "Penyiraman": "Siram dalam 2–3 kali seminggu di pagi hari, langsung ke tanah. Daun yang basah "
                          "semalaman mudah terserang bercak hitam.",
            "Pemupukan": "Pupuk khusus mawar atau NPK seimbang tiap 2 minggu, ditambah kompos atau pupuk kandang "
                         "setiap 2–3 bulan.",
            "Pemangkasan": "Potong batang mati, lemah, dan saling bersilangan. Setelah bunga layu, potong sekitar "
                           "0,5 cm di atas mata tunas yang menghadap ke luar.",
            "Hama & penyakit": "Kutu daun, tungau, bercak hitam (black spot), dan embun tepung. Buang daun "
                               "bergejala dan jangan dibiarkan berserakan di bawah tanaman.",
            "Perbanyakan": "Stek batang bekas bunga sepanjang 15–20 cm, atau okulasi untuk varietas unggul di atas "
                           "batang bawah yang kuat.",
        },
        "tip": "Sinar matahari pagi paling penting untuk mawar karena mengeringkan embun di daun dan mengurangi "
               "risiko jamur.",
    },
}

ORDER = list(FLOWERS)

# Folder yang dicek untuk mencari foto (urut prioritas)
IMAGE_DIRS = [APP_DIR / "assets", APP_DIR / "orginal data", APP_DIR / "original data", APP_DIR / "resized_data",
              APP_DIR.parent / "orginal data", APP_DIR.parent / "resized_data"]
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_photo(name: str):
    """assets/<Nama>.jpg dulu, lalu foto pertama di folder dataset <folder>/<Nama>/."""
    for d in IMAGE_DIRS:
        for ext in EXTS:
            p = d / f"{name}{ext}"
            if p.exists():
                return p
        sub = d / name
        if sub.is_dir():
            files = sorted(f for f in sub.iterdir() if f.suffix.lower() in EXTS)
            if files:
                return files[0]
    return None


def photo_data_uri(name: str, size=640):
    p = find_photo(name)
    if p is None:
        return "data:image/svg+xml;base64," + base64.b64encode(illustration(name).encode()).decode()
    img = ImageOps.fit(Image.open(p).convert("RGB"), (size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ───────────── Ilustrasi cadangan (dipakai bila belum ada foto) ─────────────
def _ring(n, r, rx, ry, fill, offset=0, cx=100, cy=100, opacity=1):
    out = []
    for i in range(n):
        a = 360 / n * i + offset
        out.append(f'<ellipse cx="{cx}" cy="{cy - r}" rx="{rx}" ry="{ry}" fill="{fill}" opacity="{opacity}" '
                   f'transform="rotate({a:.1f} {cx} {cy})"/>')
    return "".join(out)


def _dots(n, r, rad, fill, offset=0):
    return "".join(
        f'<circle cx="{100 + r*math.cos(math.radians(360/n*i+offset)):.1f}" '
        f'cy="{100 + r*math.sin(math.radians(360/n*i+offset)):.1f}" r="{rad}" fill="{fill}"/>' for i in range(n))


def illustration(name: str) -> str:
    c = FLOWERS[name]["color"]
    leaves = ('<ellipse cx="52" cy="160" rx="34" ry="13" fill="#5E8C6A" transform="rotate(-35 52 160)"/>'
              '<ellipse cx="150" cy="166" rx="30" ry="11" fill="#4C7A5A" transform="rotate(30 150 166)"/>')
    if name == "Chrysanthemum":
        body = (_ring(26, 46, 5, 30, "#E9B93A") + _ring(20, 34, 5, 22, "#D9A21B", 9) +
                _ring(14, 22, 4.5, 15, "#C48A12", 4) + '<circle cx="100" cy="100" r="11" fill="#9C6A0C"/>')
    elif name == "Hibiscus":
        body = (_ring(5, 34, 34, 40, "#D9466F") + _ring(5, 30, 26, 32, "#C8305F", 0, opacity=.6) +
                '<circle cx="100" cy="100" r="16" fill="#7E1235"/>'
                '<path d="M100 100 Q118 72 132 48" stroke="#F3C64B" stroke-width="4" fill="none" stroke-linecap="round"/>'
                + "".join(f'<circle cx="{128+dx}" cy="{50+dy}" r="3.5" fill="#F3C64B"/>'
                          for dx, dy in [(0, 0), (6, -4), (-3, -6), (8, 3)]))
    elif name == "Marigold":
        body = (_dots(16, 50, 15, "#F08A2A") + _dots(13, 38, 14, "#E8701A", 12) + _dots(10, 26, 12, "#D95F10", 6) +
                _dots(6, 13, 10, "#C9520C", 20) + '<circle cx="100" cy="100" r="8" fill="#B5470A"/>')
    elif name == "Petuniya":
        body = (_dots(5, 30, 36, "#8A5BC6", -90) + '<circle cx="100" cy="100" r="34" fill="#8A5BC6"/>' +
                "".join(f'<line x1="100" y1="100" x2="{100+56*math.cos(math.radians(72*i-90)):.1f}" '
                        f'y2="{100+56*math.sin(math.radians(72*i-90)):.1f}" stroke="#4E2A80" stroke-width="2.5" '
                        f'opacity=".55"/>' for i in range(5)) +
                '<circle cx="100" cy="100" r="13" fill="#3E1F68"/><circle cx="100" cy="100" r="4" fill="#F1E3A0"/>')
    else:  # Rose
        body = (_dots(5, 32, 34, "#C93A50", -90) + _dots(5, 20, 26, "#B8233A", -54) +
                '<circle cx="100" cy="100" r="26" fill="#A11C31"/>'
                '<path d="M100 100 m-4 0 a4 4 0 1 1 8 0 a9 9 0 1 1 -16 0 a14 14 0 1 1 26 2 a19 19 0 1 1 -34 -4" '
                'stroke="#7A1022" stroke-width="3" fill="none" stroke-linecap="round"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            f'<rect width="200" height="200" fill="{c}" fill-opacity=".13"/>{leaves}{body}</svg>')
