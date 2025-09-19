# 🗺️ Prediksi Harga Tanah Menggunakan IDW

Aplikasi ini dibuat menggunakan **Python + Streamlit** untuk memprediksi harga tanah berdasarkan koordinat (Longitude & Latitude) menggunakan metode **IDW (Inverse Distance Weighting)**.
Aplikasi mendukung **optimasi parameter Power** melalui **cross-validation**, serta menampilkan hasil prediksi di peta interaktif dengan warna kategori harga.

---

## 🚀 Fitur Utama

* **Upload Dataset Sumber**

  * Format `.csv` atau `.xlsx` dengan kolom: `Longitude`, `Latitude`, `Harga`.

* **Optimasi Power IDW**

  * Cross-validation otomatis dengan split 80% training dan 20% validasi.
  * Menampilkan nilai **RMSE** untuk tiap power yang diuji.

* **Mode Prediksi**

  * **Koordinat Unggahan**: Upload file koordinat (Longitude, Latitude) untuk diprediksi.
  * **Grid Otomatis**: Membuat grid berdasarkan bounding area data sumber.

* **Visualisasi Peta Interaktif**

  * Menggunakan **Folium**.
  * Prediksi harga divisualisasikan dengan **3 warna**:

    * 🟢 **Murah** (< persentil 33)
    * 🟠 **Sedang**
    * 🔴 **Mahal** (> persentil 66)
  * Legenda otomatis ditampilkan di peta.

* **Download Hasil Prediksi**

  * Simpan hasil prediksi ke dalam **CSV** atau **Excel**.

---

## 🛠️ Instalasi

1. Clone repository ini:

   ```bash
   git clone https://github.com/ridwana22/ProjekIDW.git
   ```

2. Buat virtual environment (opsional tapi disarankan):

   ```bash
   python -m venv venv
   source venv/bin/activate     # Mac/Linux
   venv\Scripts\activate        # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📦 Dependencies

Beberapa library utama yang digunakan:

* [streamlit](https://streamlit.io/) – Framework web untuk Python.
* [pandas](https://pandas.pydata.org/) – Manipulasi data.
* [numpy](https://numpy.org/) – Perhitungan numerik.
* [folium](https://python-visualization.github.io/folium/) – Peta interaktif.
* [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) – Integrasi Folium dengan Streamlit.
* [openpyxl](https://openpyxl.readthedocs.io/) – Support file Excel.

---

## ▶️ Cara Menjalankan

Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

Lalu buka link yang muncul (biasanya `http://localhost:8501`) di browser.

---
