# Fruit Object Detection System using YOLOv8

Sistem deteksi dan klasifikasi objek buah-buahan berbasis Artificial Intelligence menggunakan framework **Ultralytics YOLOv8**. Proyek ini mencakup script *training* model pada dataset Kaggle dan script *inference* yang menampilkan hasil deteksi objek melalui *pop-up window*.

---

## Spesifikasi Lingkungan Pengembangan (Environment)

Berikut adalah rincian spesifikasi sistem operasi dan perangkat yang digunakan selama pengerjaan serta proses *training* model:

* **OS**: Windows 11 Home
* **Processor (CPU)**: AMD Ryzen 7 4800H with Radeon Graphics (8 Cores, 16 Threads)
* **RAM**: 16 GB
* **Tech Stack**: Python 3.13.0
* **Framework AI**: Ultralytics YOLOv8 (`v8.4.152`)
* **Library Utama**: OpenCV (`opencv-python`), Pandas, PyYAML, Kagglehub

---

## Fitur & Ketentuan Teknis

* **Model/Weights (`weights/best.pt`)**
  * File bobot hasil pelathan model YOLOv8 yang disimpan pada direktori `weights/best.pt`.

* **Script Training (`train.py`)**
  * Otomatisasi penataan ulang struktur dataset dari Kaggle.
  * Konversi `_classes.csv` menjadi file label format standar YOLO (`.txt`).
  * Eksekusi proses *training* model menggunakan YOLOv8.

* **Script Inference (`inference.py`)**
  * **Spesifikasi Path**: Lokasi gambar ditentukan di dalam *script* (tanpa *input* terminal).
  * **Pop-up Window**: Menampilkan *preview* hasil deteksi menggunakan GUI OpenCV.
  * **Anotasi Visual**: Gambar preview sudah dilengkapi *bounding box* dan label nama buah.

---

## Panduan Instalasi & Cara Menjalankan

### 1. Instalasi Dependency

Pastikan Python sudah terinstal, lalu buat *virtual environment* dan instal seluruh pustaka yang dibutuhkan:

```bash
# Buat dan aktifkan virtual environment
python -m venv venv
.\venv\Scripts\activate   # Untuk Windows (PowerShell/CMD)

# Instal library pendukung
pip install ultralytics opencv-python pandas pyyaml kagglehub python-dotenv
```

### 2. Konfigurasi Token Kaggle

Buat file .env di root directory proyek dan masukkan token Kaggle API:

```
KAGGLE_API_TOKEN=KGAT_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Menjalankan Training (train.py)

Untuk melatih model dari awal menggunakan dataset Kaggle:

```Bash
python train.py
```

Script akan mengunduh dataset, menyiapkan file label .txt, merapikan struktur folder, dan memulai proses training YOLOv8. File best.pt akan dihasilkan di direktori runs/detect/fruits_yolo_model/weights/best.pt.

### 4. Menjalankan Inference & Pop-up Preview (inference.py)

Untuk menguji deteksi objek dan menampilkan jendela pop-up:

```Bash
python inference.py
```
Pop-up window OpenCV akan muncul menampilkan gambar dengan bounding box dan label nama buah. Tekan tombol apa saja pada keyboard untuk menutup jendela.
