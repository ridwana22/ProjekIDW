🗺️ Prediksi Harga Tanah Menggunakan IDW

Aplikasi ini dibuat menggunakan Python + Streamlit untuk memprediksi harga tanah berdasarkan koordinat (Longitude & Latitude) menggunakan metode IDW (Inverse Distance Weighting).
Aplikasi mendukung optimasi parameter Power melalui cross-validation, serta menampilkan hasil prediksi di peta interaktif dengan warna kategori harga.

🚀 Fitur Utama

Upload Dataset Sumber

Format .csv atau .xlsx dengan kolom: Longitude, Latitude, Harga.

Optimasi Power IDW

Cross-validation otomatis dengan split 80% training dan 20% validasi.

Menampilkan nilai RMSE untuk tiap power yang diuji.

Mode Prediksi

Koordinat Unggahan: Upload file koordinat (Longitude, Latitude) untuk diprediksi.

Grid Otomatis: Membuat grid berdasarkan bounding area data sumber.

Visualisasi Peta Interaktif

Menggunakan Folium.

Prediksi harga divisualisasikan dengan 3 warna:

🟢 Murah (< persentil 33)

🟠 Sedang

🔴 Mahal (> persentil 66)

Legenda otomatis ditampilkan di peta.

Download Hasil Prediksi

Simpan hasil prediksi ke dalam CSV atau Excel.

🛠️ Instalasi

Clone repository ini:

git clone https://github.com/username/prediksi-harga-tanah-idw.git
cd prediksi-harga-tanah-idw


Buat virtual environment (opsional tapi disarankan):

python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows


Install dependencies:

pip install -r requirements.txt

📦 Dependencies

Beberapa library utama yang digunakan:

streamlit
 – Framework web untuk Python.

pandas
 – Manipulasi data.

numpy
 – Perhitungan numerik.

folium
 – Peta interaktif.

streamlit-folium
 – Integrasi Folium dengan Streamlit.

openpyxl
 – Support file Excel.

▶️ Cara Menjalankan

Jalankan aplikasi Streamlit:

streamlit run app.py


Lalu buka link yang muncul (biasanya http://localhost:8501) di browser.
