import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageOps

from flowers import FLOWERS, ORDER, photo_data_uri
from model import count_params, grad_cam, load_checkpoint, overlay_heatmap, predict

# ─────────────────────────── Konfigurasi ───────────────────────────
APP_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = APP_DIR / "flower_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Flora · Pengenal Bunga", page_icon="🌸", layout="wide",
                   initial_sidebar_state="collapsed")


# Warna tiap kelas diambil dari warna kelopak bunganya sendiri
PETAL = {
    "Chrysanthemum": "#D9A21B",
    "Hibiscus": "#C8305F",
    "Marigold": "#E8701A",
    "Petuniya": "#7A4CB5",
    "Rose": "#B8233A",
}
LOCAL_NAME = {
    "Chrysanthemum": "Krisan",
    "Hibiscus": "Kembang sepatu",
    "Marigold": "Gemitir",
    "Petuniya": "Petunia",
    "Rose": "Mawar",
}
INK, MUTED, LEAF, PAPER, LINE = "#1E2B24", "#66746B", "#2F5D46", "#F2F5F0", "#DCE3DB"

# Hasil dari notebook (dipakai bila checkpoint tidak membawa histori sendiri)
NB_HISTORY = {
    "train_loss": [.4674, .2608, .1832, .1762, .1167, .1221, .1167, .0920, .1124, .1168, .0760, .0869, .1037, .0800, .0842,
                   .0585, .0454, .0602, .0694, .0589, .0351, .0253, .0408, .0301, .0231, .0315, .0322, .0265, .0338, .0265],
    "val_loss": [.4560, .1938, .3539, .1633, .1173, .1955, .1114, .0814, .1103, .0658, .1535, .1312, .1006, .1066, .0597,
                 .0414, .0543, .0445, .0420, .0897, .0786, .0625, .0558, .0670, .0577, .0471, .0542, .0523, .0434, .0484],
    "train_acc": [.8276, .9112, .9357, .9408, .9653, .9663, .9622, .9684, .9582, .9602, .9776, .9704, .9673, .9724, .9755,
                  .9857, .9888, .9806, .9755, .9786, .9888, .9959, .9878, .9929, .9949, .9908, .9898, .9898, .9898, .9918],
    "val_acc": [.8429, .9429, .8810, .9429, .9619, .9286, .9524, .9762, .9619, .9714, .9429, .9524, .9714, .9667, .9857,
                .9905, .9714, .9857, .9857, .9714, .9810, .9810, .9810, .9810, .9810, .9857, .9762, .9762, .9857, .9762],
}
NB_REPORT = pd.DataFrame({
    "Kelas": ["Chrysanthemum", "Hibiscus", "Marigold", "Petuniya", "Rose"],
    "Precision": [.96, .94, 1.00, 1.00, .97],
    "Recall": [.98, 1.00, .98, 1.00, .93],
    "F1-score": [.97, .97, .99, 1.00, .95],
    "Support": [44, 29, 51, 46, 40],
})
NB_DATASET = {"Chrysanthemum": 291, "Hibiscus": 197, "Marigold": 335, "Petuniya": 309, "Rose": 268}
NB_TEST_ACC, NB_BEST_VAL = 0.9762, 0.9905

# ─────────────────────────── Gaya ───────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, label {{
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }}
#MainMenu, footer, [data-testid="stAppDeployButton"] {{ visibility: hidden; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif !important; color: {INK}; font-weight: 600; letter-spacing: -0.01em; }}

.brand {{ display:flex; align-items:baseline; gap:.9rem; margin-bottom:.2rem; }}
.brand .logo {{ font-family:'Fraunces',serif; font-weight:600; font-size:2.6rem; line-height:1; color:{INK}; letter-spacing:-.02em; }}
.brand span {{ color:{MUTED}; font-size:.98rem; }}
.petals {{ display:flex; gap:6px; margin:.9rem 0 1.6rem; }}
.petals i {{ width:22px; height:10px; border-radius:10px 0 10px 0; display:block; }}

/* Tab sebagai pil lembut */
.stTabs [role="tablist"] {{ gap:.3rem; border-bottom:none !important; box-shadow:none !important; background:#E6ECE4;
  padding:.3rem; border-radius:14px; width:fit-content; max-width:100%; overflow-x:auto; margin-bottom:1rem; }}
.stTabs [role="tab"] {{ padding:.45rem 1.1rem; border-radius:10px; color:{MUTED}; }}
.stTabs [role="tab"] p {{ font-weight:500; font-size:.95rem; }}
.stTabs [role="tab"][aria-selected="true"] {{ background:white; color:{INK}; box-shadow:0 1px 2px rgba(30,43,36,.08); }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display:none !important; }}

/* Area unggah */
[data-testid="stFileUploaderDropzone"] {{ background:white; border:1.5px dashed #B9C6B8; border-radius:18px; padding:1.6rem; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{LEAF}; }}

.stButton>button, .stDownloadButton>button {{ border-radius:999px; border:1px solid {LINE}; background:white; color:{INK}; font-weight:500; padding:.45rem 1.2rem; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{LEAF}; color:{LEAF}; }}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible {{ outline:2px solid {LEAF}; outline-offset:2px; }}

/* Kartu hasil utama */
.verdict {{ background:white; border-radius:24px; padding:1.8rem 2rem 1.6rem; position:relative; overflow:hidden; }}
.verdict::before {{ content:""; position:absolute; right:-60px; top:-60px; width:200px; height:200px;
  background:var(--c); opacity:.12; border-radius:100px 0 100px 0; transform:rotate(12deg); }}
.verdict small {{ color:{MUTED}; font-size:.9rem; }}
.verdict .name {{ font-family:'Fraunces',serif; font-size:3.3rem; line-height:1.05; color:var(--c); margin:.3rem 0 .1rem; font-weight:600; }}
.verdict .local {{ color:{MUTED}; font-size:1.05rem; margin-bottom:1.3rem; }}
.meter {{ height:8px; background:#EEF1ED; border-radius:8px; overflow:hidden; }}
.meter b {{ display:block; height:100%; background:var(--c); border-radius:8px; animation:grow .9s cubic-bezier(.2,.8,.2,1); }}
@keyframes grow {{ from {{ width:0 }} }}
@media (prefers-reduced-motion: reduce) {{ .meter b {{ animation:none; }} }}
.meter-row {{ display:flex; justify-content:space-between; margin-top:.5rem; font-size:.92rem; color:{MUTED}; }}
.meter-row strong {{ color:{INK}; font-size:1.15rem; }}

.note {{ border-left:3px solid #D9A21B; background:#FBF6E8; padding:.8rem 1rem; border-radius:0 12px 12px 0; margin-top:1rem; font-size:.93rem; }}
.empty {{ text-align:center; color:{MUTED}; padding:3rem 1rem; }}
.empty .big {{ font-family:'Fraunces',serif; font-size:1.5rem; color:{INK}; margin-bottom:.3rem; }}

.chip {{ display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600; color:white; }}
.thumb-cap {{ font-size:.85rem; color:{MUTED}; margin-top:.3rem; }}

.stat {{ background:white; border-radius:18px; padding:1.1rem 1.3rem; }}
.stat .v {{ font-family:'Fraunces',serif; font-size:2rem; line-height:1.1; }}
.stat .l {{ color:{MUTED}; font-size:.88rem; }}

[data-testid="stImage"] img {{ border-radius:18px; }}
[data-testid="stSidebar"] {{ background:white; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Model ───────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(blob: bytes | None, path: str | None, version: str = ""):
    # `version` (ukuran + waktu ubah file) membuat cache ikut diperbarui saat flower_cnn.pth diganti
    src = io.BytesIO(blob) if blob else path
    return load_checkpoint(src, DEVICE)


def color_of(name):
    return PETAL.get(name, LEAF)


def prob_chart(probs, classes, height=230):
    order = np.argsort(probs)
    fig = go.Figure(go.Bar(
        x=probs[order] * 100, y=[classes[i] for i in order], orientation="h",
        marker=dict(color=[color_of(classes[i]) for i in order], cornerradius=6),
        text=[f"{probs[i]*100:.1f}%" for i in order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(height=height, margin=dict(l=0, r=40, t=6, b=0), bargap=.38,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans", color=INK, size=13),
                      xaxis=dict(visible=False, range=[0, 112]), yaxis=dict(showgrid=False))
    return fig


def line_chart(a, b, la, lb, ytitle, pct=False):
    ep = list(range(1, len(a) + 1))
    fig = go.Figure()
    for ys, name, col, dash in [(a, la, "#9AAE9F", "solid"), (b, lb, LEAF, "solid")]:
        fig.add_trace(go.Scatter(x=ep, y=ys, name=name, mode="lines",
                                 line=dict(color=col, width=2.4 if col == LEAF else 2, dash=dash, shape="spline")))
    fig.update_layout(height=280, margin=dict(l=0, r=10, t=30, b=0), title=dict(text=ytitle, font=dict(size=15, family="Fraunces")),
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Plus Jakarta Sans", color=INK),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                      xaxis=dict(title="Epoch", showgrid=False), yaxis=dict(gridcolor="#EEF1ED", tickformat=".0%" if pct else None))
    return fig


def page_identify():

    st.markdown("## Kenali bunga dari foto")
    with st.sidebar:
        st.markdown("### Pengaturan")
        up_weights = st.file_uploader("Bobot model (.pth)", type=["pth", "pt"],
                                      help="Kosongkan untuk memakai flower_cnn.pth di folder aplikasi.")
        threshold = st.slider("Batas keyakinan", 0.30, 0.95, 0.60, 0.05,
                              help="Di bawah angka ini, hasil diberi tanda kurang yakin.")
        use_tta = st.toggle("Prediksi + cermin (TTA)", value=True,
                            help="Rata-rata prediksi gambar asli dan versi cerminnya. Sedikit lebih lambat, biasanya lebih stabil.")
        show_cam = st.toggle("Tampilkan area fokus model", value=True,
                             help="Grad-CAM: bagian gambar yang paling memengaruhi keputusan model.")
        st.caption(f"Perangkat: `{DEVICE}`")

    # Muat model
    model = classes = meta = None
    load_error = None
    try:
        if up_weights is not None:
            model, classes, meta = get_model(up_weights.getvalue(), None)
        elif DEFAULT_WEIGHTS.exists():
            st_ = DEFAULT_WEIGHTS.stat()
            model, classes, meta = get_model(None, str(DEFAULT_WEIGHTS), f"{st_.st_size}-{st_.st_mtime_ns}")
    except Exception as e:  # noqa: BLE001
        load_error = str(e)

    if "history" not in st.session_state:
        st.session_state.history = []

    if model is None:
        if load_error:
            st.error(f"Model tidak bisa dimuat: {load_error}. Pastikan file berasal dari ImprovedCNN di notebook.")
        else:
            st.markdown(f"""
<div class="verdict" style="--c:{LEAF}">
  <div class="name" style="font-size:2rem">Model belum dipasang</div>
  <div class="local">Simpan bobot dari notebook sebagai <code>flower_cnn.pth</code> di folder yang sama dengan
  <code>app.py</code>, atau unggah lewat panel Pengaturan di kiri.</div>
</div>""", unsafe_allow_html=True)
            st.code("""# Tambahkan di akhir notebook, setelah training
torch.save({
    "state_dict": best_model_wts,
    "class_names": class_names,
    "history": {"train_loss": train_losses, "val_loss": val_losses,
                "train_acc": train_accs, "val_acc": val_accs},
    "test_acc": float(test_acc),
}, "flower_cnn.pth")""", language="python")

    tab_one, tab_batch, tab_hist, tab_about = st.tabs(["Identifikasi", "Banyak foto", "Riwayat", "Tentang model"])


    def log(name, label, conf):
        st.session_state.history.insert(0, {
            "Waktu": datetime.now().strftime("%H:%M:%S"), "File": name,
            "Prediksi": label, "Keyakinan": round(float(conf) * 100, 2),
        })


    # ─────────────────────────── Tab 1: satu foto ───────────────────────────
    with tab_one:
        src = st.radio("Sumber foto", ["Unggah", "Kamera"], horizontal=True, label_visibility="collapsed")
        file = (st.file_uploader("Pilih foto bunga", type=["jpg", "jpeg", "png", "webp"], key="single")
                if src == "Unggah" else st.camera_input("Ambil foto bunga"))

        if file is None:
            st.markdown('<div class="empty"><div class="big">Mulai dengan satu foto</div>'
                        'Krisan, kembang sepatu, gemitir, petunia, atau mawar. Foto close-up memberi hasil terbaik.</div>',
                        unsafe_allow_html=True)
        elif model is not None:
            img = Image.open(file).convert("RGB")
            t0 = time.perf_counter()
            probs = predict(model, img, tta=use_tta, device=DEVICE)
            ms = (time.perf_counter() - t0) * 1000
            top = int(np.argmax(probs))
            label, conf = classes[top], float(probs[top])
            runner = np.sort(probs)[-2]

            key = (getattr(file, "name", "kamera"), file.size if hasattr(file, "size") else len(file.getvalue()))
            if st.session_state.get("last_logged") != key:
                log(getattr(file, "name", "kamera.jpg"), label, conf)
                st.session_state.last_logged = key

            left, right = st.columns([1, 1.15], gap="large")
            with left:
                if show_cam:
                    cam = grad_cam(model, img, top, DEVICE)
                    view = st.segmented_control("Tampilan", ["Foto", "Area fokus"], default="Foto",
                                                label_visibility="collapsed")
                    st.image(overlay_heatmap(img, cam) if view == "Area fokus" else img, use_container_width=True)
                    if view == "Area fokus":
                        st.caption("Area terang = bagian yang paling diperhatikan model.")
                else:
                    st.image(img, use_container_width=True)

            with right:
                c = color_of(label)
                st.markdown(f"""
<div class="verdict" style="--c:{c}">
  <small>Kemungkinan besar ini</small>
  <div class="name">{label}</div>
  <div class="local">{LOCAL_NAME.get(label, '')}</div>
  <div class="meter"><b style="width:{conf*100:.1f}%"></b></div>
  <div class="meter-row"><span>Keyakinan</span><strong>{conf*100:.1f}%</strong></div>
</div>""", unsafe_allow_html=True)

                if conf < threshold:
                    st.markdown(f'<div class="note">Keyakinan di bawah {threshold:.0%}. Foto ini mungkin bukan salah satu '
                                f'dari lima bunga yang dikenali model, atau bunganya terlalu kecil di dalam bingkai.</div>',
                                unsafe_allow_html=True)
                elif conf - runner < 0.15:
                    second = classes[int(np.argsort(probs)[-2])]
                    st.markdown(f'<div class="note">Hasilnya tipis dengan <b>{second}</b>. Coba foto dari sudut lain.</div>',
                                unsafe_allow_html=True)

                st.plotly_chart(prob_chart(probs, classes), use_container_width=True, config={"displayModeBar": False})
                st.caption(f"{img.width}×{img.height}px · diproses dalam {ms:.0f} ms" + (" dengan TTA" if use_tta else ""))
                with st.expander("Kapan hasilnya bisa meleset?"):
                    st.markdown("Model dilatih dari ±2.000 foto: foto kebun ditambah foto beragam dari dataset Oxford 102 Flowers "
                                "dan TensorFlow flower_photos. Krisan di data latihan masih didominasi warna putih, jadi krisan "
                                "berwarna lain, bunga yang bukan salah satu dari kelima jenis ini, atau foto yang buram "
                                "bisa ditebak salah, bahkan dengan angka keyakinan tinggi.")


    # ─────────────────────────── Tab 2: banyak foto ───────────────────────────
    with tab_batch:
        files = st.file_uploader("Pilih beberapa foto sekaligus", type=["jpg", "jpeg", "png", "webp"],
                                 accept_multiple_files=True, key="batch")
        if not files:
            st.markdown('<div class="empty"><div class="big">Klasifikasikan satu folder sekaligus</div>'
                        'Pilih banyak foto, lalu unduh hasilnya sebagai CSV.</div>', unsafe_allow_html=True)
        elif model is not None:
            rows, prog = [], st.progress(0.0, text="Memproses foto…")
            for i, f in enumerate(files):
                im = Image.open(f).convert("RGB")
                p = predict(model, im, tta=use_tta, device=DEVICE)
                k = int(np.argmax(p))
                rows.append({"img": ImageOps.fit(im, (320, 320)), "File": f.name, "Prediksi": classes[k], "Keyakinan": float(p[k]),
                             **{c: float(p[j]) for j, c in enumerate(classes)}})
                prog.progress((i + 1) / len(files), text=f"Memproses foto {i+1} dari {len(files)}")
            prog.empty()

            df = pd.DataFrame([{k: v for k, v in r.items() if k != "img"} for r in rows])
            low = (df["Keyakinan"] < threshold).sum()

            s1, s2, s3 = st.columns(3)
            for col, v, l in [(s1, len(df), "foto diproses"),
                              (s2, f"{df['Keyakinan'].mean()*100:.1f}%", "rata-rata keyakinan"),
                              (s3, int(low), f"perlu dicek (< {threshold:.0%})")]:
                col.markdown(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>',
                             unsafe_allow_html=True)
            st.write("")

            g1, g2 = st.columns([1.6, 1], gap="large")
            with g2:
                counts = df["Prediksi"].value_counts()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.62, sort=False,
                                       marker=dict(colors=[color_of(c) for c in counts.index], line=dict(color=PAPER, width=3)),
                                       textinfo="value", hovertemplate="%{label}: %{value} foto<extra></extra>"))
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Plus Jakarta Sans", color=INK), legend=dict(orientation="h", y=-.1))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                only_low = st.toggle("Hanya yang perlu dicek")
                out = df.copy()
                out["Keyakinan"] = (out["Keyakinan"] * 100).round(2)
                st.download_button("Unduh hasil (CSV)", out.to_csv(index=False).encode(), "hasil_klasifikasi_bunga.csv",
                                   "text/csv", use_container_width=True)
            with g1:
                shown = [r for r in rows if not only_low or r["Keyakinan"] < threshold]
                if not shown:
                    st.caption("Semua foto di atas batas keyakinan.")
                for start in range(0, len(shown), 4):
                    cols = st.columns(4)
                    for col, r in zip(cols, shown[start:start + 4]):
                        with col:
                            st.image(r["img"], use_container_width=True)
                            flag = " ⚠" if r["Keyakinan"] < threshold else ""
                            st.markdown(f'<span class="chip" style="background:{color_of(r["Prediksi"])}">{r["Prediksi"]}</span>'
                                        f'<div class="thumb-cap">{r["Keyakinan"]*100:.1f}%{flag} · {r["File"][:18]}</div>',
                                        unsafe_allow_html=True)


    # ─────────────────────────── Tab 3: riwayat ───────────────────────────
    with tab_hist:
        h = st.session_state.history
        if not h:
            st.markdown('<div class="empty"><div class="big">Belum ada riwayat</div>'
                        'Setiap foto yang kamu identifikasi di sesi ini akan tercatat di sini.</div>', unsafe_allow_html=True)
        else:
            hdf = pd.DataFrame(h)
            st.dataframe(hdf, use_container_width=True, hide_index=True,
                         column_config={"Keyakinan": st.column_config.ProgressColumn("Keyakinan", format="%.1f%%",
                                                                                    min_value=0, max_value=100)})
            a, b, _ = st.columns([1, 1, 3])
            a.download_button("Unduh riwayat", hdf.to_csv(index=False).encode(), "riwayat_flora.csv", "text/csv")
            if b.button("Hapus riwayat"):
                st.session_state.history = []
                st.session_state.pop("last_logged", None)
                st.rerun()


    # ─────────────────────────── Tab 4: tentang model ───────────────────────────
    with tab_about:
        hist = (meta or {}).get("history") or NB_HISTORY
        test_acc = (meta or {}).get("test_acc", NB_TEST_ACC)
        report_df = NB_REPORT
        if (meta or {}).get("report"):
            r = meta["report"]
            report_df = pd.DataFrame({"Kelas": list(r),
                                      "Precision": [round(v["precision"], 2) for v in r.values()],
                                      "Recall": [round(v["recall"], 2) for v in r.values()],
                                      "F1-score": [round(v["f1-score"], 2) for v in r.values()],
                                      "Support": [int(v["support"]) for v in r.values()]})
        ds = (meta or {}).get("dataset_counts") or NB_DATASET
        best_val = max(hist["val_acc"])
        n_params = count_params(model) if model is not None else 16_875_013

        s = st.columns(4)
        for col, v, l in [(s[0], f"{test_acc*100:.1f}%", "akurasi data uji"),
                          (s[1], f"{best_val*100:.1f}%", "akurasi validasi terbaik"),
                          (s[2], f"{len(hist['val_acc'])}", "epoch training"),
                          (s[3], f"{n_params/1e6:.1f} jt", "parameter")]:
            col.markdown(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns(2, gap="medium")
        c1.plotly_chart(line_chart(hist["train_acc"], hist["val_acc"], "Train", "Validasi", "Akurasi", pct=True),
                        use_container_width=True, config={"displayModeBar": False})
        c2.plotly_chart(line_chart(hist["train_loss"], hist["val_loss"], "Train", "Validasi", "Loss"),
                        use_container_width=True, config={"displayModeBar": False})

        d1, d2 = st.columns([1.3, 1], gap="large")
        with d1:
            st.markdown("### Kinerja per kelas")
            st.dataframe(report_df, hide_index=True, use_container_width=True,
                         column_config={
                             "Precision": st.column_config.NumberColumn(format="%.2f"),
                             "Recall": st.column_config.NumberColumn(format="%.2f"),
                             "F1-score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                         })
            st.caption(f"Dari {int(report_df['Support'].sum())} foto uji (split 70/15/15, stratified, seed 42).")
        with d2:
            st.markdown("### Isi dataset")
            fig = go.Figure(go.Bar(x=list(ds.values()), y=list(ds.keys()), orientation="h",
                                   marker=dict(color=[color_of(k) for k in ds], cornerradius=6),
                                   text=list(ds.values()), textposition="outside", cliponaxis=False))
            fig.update_layout(height=250, margin=dict(l=0, r=30, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans", color=INK),
                              xaxis=dict(visible=False), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"{sum(ds.values()):,} foto (asli + tambahan), diperkecil ke 128×128 px.".replace(",", "."))

        with st.expander("Arsitektur & konfigurasi training"):
            st.markdown("""
**ImprovedCNN** — 3 blok konvolusi (32 → 64 → 128 filter, kernel 3×3) masing-masing dengan BatchNorm, ReLU, dan MaxPool,
lalu Dense 512 + BatchNorm + Dropout 0,5, dan output 5 kelas.

- Input 128×128 RGB, normalisasi mean/std 0,5
- Augmentasi: RandomResizedCrop (0,8–1,0), flip horizontal, rotasi ±20°, ColorJitter 0,2
- Optimizer AdamW (lr 0,001, weight decay 1e-4), ReduceLROnPlateau (faktor 0,5, patience 3)
- Batch 32, 30 epoch, bobot terbaik dipilih dari akurasi validasi
""")


# ─────────────────────────── Katalog & detail ───────────────────────────
st.markdown(f"""
<style>
.intro {{ font-family:'Fraunces',serif; font-size:2.1rem; line-height:1.15; max-width:22ch; margin:.4rem 0 .5rem; color:{INK}; }}
.lede {{ color:{MUTED}; max-width:56ch; margin-bottom:1.6rem; }}

.strip {{ display:flex; gap:1rem; overflow-x:auto; scroll-snap-type:x mandatory; padding:.2rem .1rem 1.1rem;
  scrollbar-width:thin; scrollbar-color:#C4D0C3 transparent; }}
.strip::-webkit-scrollbar {{ height:6px; }}
.strip::-webkit-scrollbar-thumb {{ background:#C4D0C3; border-radius:6px; }}
.card {{ flex:0 0 250px; scroll-snap-align:start; background:white; border-radius:22px; overflow:hidden;
  text-decoration:none !important; color:{INK} !important; border:1px solid transparent;
  transition:transform .25s ease, border-color .25s ease; }}
.card:hover {{ transform:translateY(-4px); border-color:var(--c); }}
.card:focus-visible {{ outline:2px solid var(--c); outline-offset:3px; }}
.card img {{ width:100%; aspect-ratio:1/1; object-fit:cover; display:block; }}
.card .body {{ padding:.9rem 1.1rem 1.1rem; border-top:4px solid var(--c); }}
.card .nm {{ font-family:'Fraunces',serif; font-size:1.45rem; font-weight:600; line-height:1.1; }}
.card .lt {{ color:{MUTED}; font-style:italic; font-size:.88rem; margin-top:.15rem; }}
.card.sm {{ flex-basis:180px; border-radius:18px; }}
.card.sm .nm {{ font-size:1.15rem; }}
@media (prefers-reduced-motion: reduce) {{ .card {{ transition:none; }} .card:hover {{ transform:none; }} }}

.glance {{ width:100%; border-collapse:collapse; background:white; border-radius:18px; overflow:hidden; font-size:.93rem; }}
.glance th, .glance td {{ text-align:left; padding:.75rem 1rem; border-bottom:1px solid #EEF1ED; }}
.glance th {{ color:{MUTED}; font-weight:500; }}
.glance tr:last-child td {{ border-bottom:none; }}
.glance a {{ color:{INK}; font-weight:600; text-decoration:none; }}
.glance a:hover {{ color:{LEAF}; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:5px 0 5px 0; margin-right:.5rem; }}

.back {{ color:{MUTED} !important; text-decoration:none !important; font-size:.95rem; }}
.back:hover {{ color:{LEAF} !important; }}
.hero-img {{ width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:28px; display:block; }}
.d-name {{ font-family:'Fraunces',serif; font-size:3.6rem; font-weight:600; line-height:1; color:var(--c); margin:.2rem 0 .4rem; }}
.d-latin {{ color:{MUTED}; font-size:1rem; }}
.d-latin i {{ color:{INK}; }}
.d-sum {{ font-size:1.06rem; line-height:1.65; margin:1.1rem 0 1.4rem; max-width:52ch; }}
.facts {{ display:grid; grid-template-columns:1fr 1fr; gap:.6rem; }}
.fact {{ background:white; border-radius:16px; padding:.8rem 1rem; }}
.fact .k {{ color:{MUTED}; font-size:.82rem; }}
.fact .v {{ font-weight:600; margin-top:.1rem; }}

.care {{ margin-top:2.6rem; }}
.care-row {{ display:grid; grid-template-columns:200px 1fr; gap:1.5rem; padding:1.15rem 0; border-top:1px solid {LINE}; }}
.care-row h4 {{ font-family:'Fraunces',serif; font-size:1.2rem; margin:0; color:{INK}; font-weight:600; }}
.care-row p {{ margin:0; line-height:1.7; max-width:68ch; }}
.tipbox {{ margin-top:1.4rem; background:white; border-radius:20px; padding:1.2rem 1.4rem; border-left:5px solid var(--c); }}
.tipbox b {{ display:block; font-family:'Fraunces',serif; font-size:1.05rem; margin-bottom:.25rem; }}
.more-h {{ font-family:'Fraunces',serif; font-size:1.4rem; margin:2.6rem 0 .8rem; }}
@media (max-width: 700px) {{
  .care-row {{ grid-template-columns:1fr; gap:.4rem; }}
  .d-name {{ font-size:2.6rem; }} .intro {{ font-size:1.7rem; }}
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def img_uri(name, size=640):
    return photo_data_uri(name, size)


def card_html(name, small=False):
    f = FLOWERS[name]
    return (f'<a class="card{" sm" if small else ""}" href="?bunga={name}" target="_self" style="--c:{f["color"]}">'
            f'<img src="{img_uri(name, 360)}" alt="Foto {f["local"]}">'
            f'<div class="body"><div class="nm">{f["local"]}</div><div class="lt">{f["latin"]}</div></div></a>')


def render_home():
    st.markdown('<div class="intro">Pilih bunga untuk melihat cara merawatnya</div>'
                '<div class="lede">Lima bunga yang dikenali model Flora. Geser ke samping untuk melihat semuanya, '
                'lalu klik salah satu untuk membuka panduan perawatannya.</div>', unsafe_allow_html=True)
    st.markdown('<div class="strip">' + "".join(card_html(n) for n in ORDER) + "</div>", unsafe_allow_html=True)

    st.markdown("### Sekilas kebutuhan tiap bunga")
    rows = "".join(
        f'<tr><td><span class="swatch" style="background:{FLOWERS[n]["color"]}"></span>'
        f'<a href="?bunga={n}" target="_self">{FLOWERS[n]["local"]}</a></td>'
        f'<td>{FLOWERS[n]["facts"]["Cahaya"]}</td><td>{FLOWERS[n]["facts"]["Air"]}</td>'
        f'<td>{FLOWERS[n]["facts"]["Suhu ideal"]}</td></tr>' for n in ORDER)
    st.markdown(f'<table class="glance"><thead><tr><th>Bunga</th><th>Cahaya</th><th>Air</th><th>Suhu ideal</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)


def render_detail(name):
    f = FLOWERS[name]
    st.markdown('<a class="back" href="?" target="_self">‹ Semua bunga</a>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown(f'<img class="hero-img" src="{img_uri(name, 900)}" alt="Foto {f["local"]}">',
                    unsafe_allow_html=True)
    with right:
        facts = "".join(f'<div class="fact"><div class="k">{k}</div><div class="v">{v}</div></div>'
                        for k, v in f["facts"].items())
        st.markdown(f"""<div style="--c:{f['color']}">
<div class="d-name">{f['local']}</div>
<div class="d-latin"><i>{f['latin']}</i> &nbsp;·&nbsp; {name} &nbsp;·&nbsp; famili {f['family']}</div>
<div class="d-sum">{f['summary']}</div>
<div class="facts">{facts}</div></div>""", unsafe_allow_html=True)

    care = "".join(f'<div class="care-row"><h4>{k}</h4><p>{v}</p></div>' for k, v in f["care"].items())
    st.markdown(f'<div class="care" style="--c:{f["color"]}"><h2>Saran perawatan</h2>{care}'
                f'<div class="tipbox"><b>Perlu diketahui</b>{f["tip"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="more-h">Bunga lainnya</div><div class="strip">' +
                "".join(card_html(n, small=True) for n in ORDER if n != name) + "</div>", unsafe_allow_html=True)


def page_catalog():
    sel = st.query_params.get("bunga")
    if sel in FLOWERS:
        render_detail(sel)
    else:
        render_home()


# ─────────────────────────── Navigasi ───────────────────────────
PAGES = {
    "katalog": st.Page(page_catalog, title="Katalog bunga", url_path="katalog", default=True),
    "kenali": st.Page(page_identify, title="Kenali dari foto", url_path="kenali"),
}
nav = st.navigation(list(PAGES.values()), position="hidden")

st.markdown("""<style>
.st-key-topnav { gap:.3rem; background:#E6ECE4; padding:.3rem; border-radius:14px; width:fit-content; margin-left:auto; }
.st-key-topnav [data-testid="stPageLink"] a { border-radius:10px; padding:.25rem .9rem; }
.st-key-topnav [data-testid="stPageLink"] a[aria-current="page"],
.st-key-topnav [data-testid="stPageLink-NavLink"][aria-current="page"] { background:white; box-shadow:0 1px 2px rgba(30,43,36,.08); }
.st-key-topnav p { font-weight:500; }
</style>""", unsafe_allow_html=True)

b, n = st.columns([1.3, 1], vertical_alignment="center")
b.markdown(
    '<div class="brand"><div class="logo">Flora</div><span>Katalog bunga dan cara merawatnya</span></div>'
    '<div class="petals" style="margin-bottom:0">' + "".join(f'<i style="background:{c}"></i>' for c in PETAL.values()) + "</div>",
    unsafe_allow_html=True)
with n.container(horizontal=True, key="topnav"):
    st.page_link(PAGES["katalog"], label="Katalog bunga")
    st.page_link(PAGES["kenali"], label="Kenali dari foto")
st.write("")
nav.run()
