# 🏞️ Prediksi Harga Tanah Menggunakan Metode IDW (Inverse Distance Weighting)

Aplikasi ini merupakan **web-app berbasis Streamlit** untuk melakukan **prediksi harga tanah** menggunakan metode **IDW (Inverse Distance Weighting)**.
Dilengkapi dengan **optimasi parameter Power menggunakan LOOCV**, **uji statistik**, serta **visualisasi peta interaktif**.

---

## 🚀 Fitur Utama

### ✅ 1. Optimasi Parameter *Power* IDW (Automatic Best Power)

Aplikasi secara otomatis:

* Menguji beberapa nilai *power* (0.5 – 5.0)
* Menjalankan **Leave-One-Out Cross-Validation (LOOCV)**
* Memilih *power* terbaik berdasarkan nilai **MAE terkecil**

---

### ✅ 2. Evaluasi Statistik & Uji Signifikansi

* Menghitung **MAE**, **RMSE**, **MAPE**, dan **R² Score**
* Menyediakan uji **Paired t-test** untuk membandingkan dua nilai *power*

---

### ✅ 3. Visualisasi Peta Interaktif

* Menggunakan **Folium Map**
* Dua mode visualisasi:

  * 🟢 **Circle Marker** (titik prediksi berwarna)
  * 🔥 **Heatmap** (pemetaan intensitas harga)
* Legenda warna otomatis berdasarkan rentang prediksi

---

### ✅ 4. Prediksi pada Dua Mode

1. **Upload file koordinat prediksi sendiri**, atau
2. **Grid otomatis** yang dibuat berdasarkan bounding box dari data sumber.

---

### ✅ 5. Download Hasil

Hasil prediksi dapat diunduh ke dalam:

* **CSV**
* **Excel (XLSX)**

---

## 📂 Struktur Input Data

### **A. Data Sumber (Required)**

File berisi titik sampel dan nilai harga.

| Kolom     | Tipe      | Contoh     |
| --------- | --------- | ---------- |
| Longitude | float     | 107.12345  |
| Latitude  | float     | -6.12345   |
| Harga     | float/int | 1500000000 |

Format file:

* `.csv`
* `.xlsx`, `.xls`

---

### **B. Data Titik Prediksi (Optional)**

Jika tidak diunggah → aplikasi membuat grid otomatis.

| Kolom     | Tipe  |
| --------- | ----- |
| Longitude | float |
| Latitude  | float |

---

## 🛠️ Instalasi & Menjalankan Aplikasi

### **1. Clone repository**

```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### **2. Install library**

```bash
pip install -r requirements.txt
```

### **3. Jalankan aplikasi**

```bash
streamlit run app.py
```

---

## 🧠 Cara Kerja Metode IDW (Ringkas)

Prediksi harga tanah dilakukan berdasarkan formula:

[
\hat{Z}(x) = \frac{\sum_{i=1}^{n} \frac{Z_i}{d_i^p}}{\sum_{i=1}^{n} \frac{1}{d_i^p}}
]

Di mana:

* ( Z_i ) = nilai harga pada titik sampel ke-i
* ( d_i ) = jarak dari titik prediksi ke titik sampel
* ( p ) = parameter power

---

## 🌟 Fitur Teknis Tambahan

* Caching untuk mempercepat LOOCV menggunakan `@st.cache_data`
* MarkerCluster untuk menampilkan data sumber
* Kolor map otomatis menggunakan `branca`
* Streamlit session state untuk menyimpan parameter *power* yang dipilih


