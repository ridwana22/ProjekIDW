# 🗺️ Prediksi Harga Tanah dengan IDW

Aplikasi berbasis **Streamlit** untuk memprediksi harga tanah menggunakan metode **Inverse Distance Weighting (IDW)**.
Aplikasi ini juga dilengkapi dengan evaluasi parameter menggunakan **Leave-One-Out Cross-Validation (LOOCV)**, uji signifikansi statistik dengan **Uji T Berpasangan (Paired t-test)**, serta visualisasi interaktif di peta.

---

## 🚀 Fitur Utama

1. **Penentuan Power Terbaik (LOOCV)**

   * Secara otomatis mencari parameter `power` terbaik untuk IDW berdasarkan error prediksi.

2. **Uji T Berpasangan (Paired t-test)**

   * Membandingkan signifikansi statistik kinerja IDW dengan dua nilai `power`.

3. **Prediksi Harga Tanah**

   * Prediksi dapat dilakukan pada:

     * Dataset koordinat baru yang diunggah pengguna.
     * Grid otomatis (peta dibagi ke dalam resolusi tertentu).

4. **Visualisasi Peta Interaktif**

   * Mode **Circle Marker** (titik prediksi dengan warna berdasarkan harga).
   * Mode **Heatmap** (peta panas distribusi harga).
   * Legenda dinamis sesuai rentang harga.

5. **Ekspor Hasil Prediksi**

   * Unduh hasil prediksi dalam format **CSV** atau **Excel**.

---

## 📦 Instalasi

1. Clone repository ini:

   ```bash
   git clone https://github.com/ridwana22/ProjekIDW.git
   ```

2. Buat virtual environment (opsional, tapi disarankan):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Cara Menjalankan

1. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run app.py
   ```

2. Buka browser di alamat:

   ```
   http://localhost:8501
   ```

---

## 📊 Format Dataset

### Dataset Sumber

Berisi data harga tanah yang diketahui. Wajib ada kolom berikut:

* `Longitude`
* `Latitude`
* `Harga`

Contoh (`sumber.csv`):

```csv
Longitude,Latitude,Harga
106.8456,-6.2088,15000000
106.8460,-6.2090,17000000
106.8470,-6.2100,16000000
```

### Dataset Prediksi (Opsional)

Berisi koordinat lokasi yang ingin diprediksi. Wajib ada kolom:

* `Longitude`
* `Latitude`

Contoh (`prediksi.csv`):

```csv
Longitude,Latitude
106.8458,-6.2095
106.8465,-6.2102
```

---

## 🧪 Evaluasi Model

* **LOOCV (Leave-One-Out Cross Validation)**:
  Digunakan untuk mencari nilai `power` IDW dengan error terendah.

* **Paired T-Test**:
  Membandingkan dua nilai `power` untuk melihat apakah perbedaan hasilnya signifikan secara statistik (`p < 0.05`).

---

## 🎨 Visualisasi

* **Circle Marker**: Titik dengan warna sesuai harga prediksi.
* **Heatmap**: Menampilkan distribusi harga tanah dalam bentuk peta panas.

---

## 📥 Download Hasil Prediksi

* Hasil prediksi dapat diunduh dalam format:

  * CSV
  * Excel

---

## 🛠️ Dependencies

* [Streamlit](https://streamlit.io/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Folium](https://python-visualization.github.io/folium/)
* [streamlit-folium](https://github.com/randyzwitch/streamlit-folium)
* [SciPy](https://scipy.org/)
* [OpenPyXL](https://openpyxl.readthedocs.io/)

---
