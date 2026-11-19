# Flora — Pengenal Bunga (Streamlit)

Antarmuka untuk model `ImprovedCNN` dari notebook `cnn_dl_improved.ipynb`
(5 kelas: Chrysanthemum, Hibiscus, Marigold, Petuniya, Rose).

## 1. Simpan model dari notebook
Tambahkan sel ini di akhir notebook, lalu jalankan:

```python
torch.save({
    "state_dict": best_model_wts,
    "class_names": class_names,
    "history": {"train_loss": train_losses, "val_loss": val_losses,
                "train_acc": train_accs, "val_acc": val_accs},
    "test_acc": float(test_acc),
}, "flower_cnn.pth")
```

Pindahkan `flower_cnn.pth` ke folder ini (sejajar dengan `app.py`).
`torch.save(best_model_wts, "flower_cnn.pth")` saja juga bisa dipakai.

## 2. Jalankan
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fitur
- **Katalog bunga (halaman utama)** — kartu lima bunga yang bisa digeser; klik untuk membuka halaman
  nama, gambar, ciri singkat, dan saran perawatan (penyiraman, pemupukan, pemangkasan, hama, perbanyakan).
  Foto diambil dari folder `assets/` (lihat `assets/README.txt`).
- **Kenali dari foto** (menu kanan atas) berisi:
- **Identifikasi** — unggah foto atau pakai kamera; nama bunga + keyakinan, grafik probabilitas semua kelas,
  peringatan saat keyakinan rendah atau hasil tipis antara dua kelas.
- **Area fokus (Grad-CAM)** — peta panas bagian gambar yang paling memengaruhi keputusan model.
- **Banyak foto** — klasifikasi massal, ringkasan, filter "perlu dicek", unduh CSV.
- **Riwayat** — catatan prediksi selama sesi, bisa diunduh.
- **Tentang model** — akurasi, kurva training, kinerja per kelas, isi dataset, arsitektur.
- **Pengaturan** (panel kiri) — ganti file bobot, batas keyakinan, TTA (prediksi + cermin), Grad-CAM on/off.
