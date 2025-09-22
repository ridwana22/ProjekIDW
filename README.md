# 🗺️ Prediksi Harga Tanah dengan IDW

Aplikasi berbasis **Streamlit** untuk memprediksi harga tanah menggunakan metode **Inverse Distance Weighting (IDW)**.
Aplikasi ini memungkinkan pengguna untuk:

* Mengunggah data sumber berisi **Longitude, Latitude, dan Harga**.
* Memprediksi harga tanah pada **koordinat baru** atau pada **grid otomatis**.
* Menentukan **parameter Power IDW** terbaik menggunakan **cross-validation**.
* Menampilkan hasil prediksi dengan **visualisasi interaktif di peta** (Circle Marker & Heatmap).
* Mengunduh hasil prediksi dalam format **CSV** atau **Excel**.

---

## 🚀 Fitur Utama

1. **Cross-validation otomatis**

   * Menentukan parameter *power* terbaik untuk IDW berdasarkan nilai RMSE terkecil.

2. **Input Fleksibel**

   * Mendukung file **CSV** dan **Excel**.
   * Bisa memprediksi pada file koordinat baru atau grid otomatis yang dibangkitkan aplikasi.

3. **Visualisasi Interaktif**

   * **Circle Marker** dengan kategori harga (Murah, Sedang, Mahal).
   * **Heatmap** berbasis harga prediksi.
   * **Legend harga** ditampilkan langsung di peta.
   * **MarkerCluster** untuk data sumber.

4. **Download Hasil Prediksi**

   * Hasil dapat diunduh dalam format **CSV** atau **Excel**.

---

## 📂 Struktur Input Data

### Data Sumber (wajib)

File harus memiliki kolom:

* `Longitude`
* `Latitude`
* `Harga`

**Contoh:**

```csv
Longitude,Latitude,Harga
112.743,-7.257,1500000
112.748,-7.260,1800000
112.750,-7.265,2000000
```

### Data Prediksi (opsional)

File berisi koordinat prediksi dengan kolom:

* `Longitude`
* `Latitude`

**Contoh:**

```csv
Longitude,Latitude
112.760,-7.270
112.770,-7.280
```

Jika tidak disediakan, aplikasi akan otomatis membangkitkan **grid** berdasarkan data sumber.

---

## ⚙️ Instalasi & Menjalankan Aplikasi

### 1. Clone repositori

```bash
git clone https://github.com/ridwana22/ProjekiIDW.git
```

### 2. Buat virtual environment (opsional, tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi Streamlit

```bash
streamlit run app.py
```

---

## 📦 Dependencies

* `streamlit`
* `pandas`
* `numpy`
* `folium`
* `streamlit-folium`
* `openpyxl`
* `branca`

---
## 📥 Output

Hasil prediksi dapat diunduh dalam format:

* **CSV** (`.csv`)
* **Excel** (`.xlsx`)

---

