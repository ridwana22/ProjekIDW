"""
Code ini digunakan untuk memprediksi harga tanah dengan IDW,
dengan menggunakan dataset sumber yang diunggah user,
dan memprediksi harga pada koordinat baru yang diunggah user atau pada grid otomatis.

Fitur Utama:
1. Penentuan parameter 'power' terbaik secara otomatis menggunakan Leave-One-Out Cross-Validation (LOOCV).
2. Perbandingan signifikansi statistik antara dua nilai 'power' menggunakan Uji T Berpasangan (Paired t-test).
3. Visualisasi hasil prediksi di peta interaktif (Circle Marker atau Heatmap) dengan legenda dinamis.
"""

# Import library
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
from io import BytesIO
import branca.colormap as cm
from scipy.stats import ttest_rel

# --- Fungsi untuk menghitung IDW ---
# Menghitung jarak antara titik yang ingin diprediksi dengan semua titik yang sudah diketahui.
# Memberi bobot berdasarkan jarak (semakin dekat → semakin besar bobotnya).
# Mengambil rata-rata tertimbang dari nilai-nilai yang diketahui untuk mendapatkan prediksi di titik baru.
def idw_interpolation(data_points, grid_points, power=2): 
    """Menghitung nilai prediksi di grid_points berdasarkan data_points menggunakan IDW."""
    known_coords = data_points[['Longitude', 'Latitude']].values 
    known_values = data_points['Harga'].values
    grid_coords = grid_points[['Longitude', 'Latitude']].values
    predicted_values = np.zeros(len(grid_coords))
    
    for i, grid_point in enumerate(grid_coords):
        distances = np.sqrt(np.sum((known_coords - grid_point)**2, axis=1))
        if np.any(distances == 0):
            zero_dist_index = np.where(distances == 0)[0][0]
            predicted_values[i] = known_values[zero_dist_index]
        else:
            weights = 1.0 / (distances ** power)
            predicted_values[i] = np.sum(weights * known_values) / np.sum(weights)
    return predicted_values

# --- Fungsi untuk menjalankan LOOCV dengan CACHING ---
@st.cache_data
def run_loocv_for_power(_df_source, power):
    """
    Menjalankan Leave-One-Out Cross-Validation untuk nilai power tertentu.
    Hasil dari fungsi ini akan disimpan di cache untuk performa.
    """
    errors = []
    source_values = _df_source['Harga'].values

    for i in range(len(_df_source)):
        # Siapkan data train dan test. df.drop() cukup efisien untuk ukuran data sedang.
        df_train_temp = _df_source.drop(_df_source.index[i])
        df_test_temp = _df_source.iloc[[i]]

        # Prediksi nilai untuk titik yang dihilangkan
        predicted_value = idw_interpolation(df_train_temp, df_test_temp[['Longitude', 'Latitude']], power)
        
        true_value = source_values[i]
        error = abs(true_value - predicted_value[0])
        errors.append(error)
        
    mae = np.mean(errors)
    return errors, mae

# Untuk mengevaluasi akurasi prediksi berdasarkan error
def evaluate_idw_accuracy(actual, predicted):
    """Mengembalikan MAE, RMSE, MAPE, dan R2."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot)

    return mae, rmse, mape, r2


# --- Fungsi mencari power terbaik dengan LOOCV ---
# Menguji beberapa nilai power IDW (0.5–5.0).
# Menggunakan LOOCV untuk mengukur akurasi tiap nilai.
# Mencari nilai dengan MAE terkecil → itulah power terbaik.
# Menampilkan hasil dan tabel analisis di Streamlit.
def find_best_power_loocv(df_source):
    """Mencari parameter power terbaik menggunakan LOOCV dan menampilkan hasilnya."""
    st.subheader("1. Penentuan Parameter Power IDW Terbaik")
    
    MIN_SAMPLES_REQUIRED = 10
    if len(df_source) < MIN_SAMPLES_REQUIRED:
        st.warning(f"Data sumber Anda hanya memiliki {len(df_source)} baris setelah pembersihan. Minimal {MIN_SAMPLES_REQUIRED} diperlukan untuk analisis LOOCV.")
        st.info("Parameter Power default (2.0) akan digunakan.")
        return 2.0, None

    with st.spinner("⏳ Melakukan LOOCV untuk mencari Power terbaik (panggilan pertama mungkin lambat)..."):
        powers_to_test = np.arange(0.5, 5.5, 0.5)
        mae_results = []

        progress_bar = st.progress(0, text="Menganalisis berbagai nilai Power...")
        for i, p in enumerate(powers_to_test):
            _, mae = run_loocv_for_power(df_source, power=p)
            mae_results.append({'Power': p, 'MAE': mae})
            progress_bar.progress((i + 1) / len(powers_to_test))
        
        results_df = pd.DataFrame(mae_results).sort_values(by='MAE').reset_index(drop=True)
        best_power = results_df.loc[0, 'Power']

    st.success(f"✅ Analisis selesai! Power terbaik = **{best_power:.2f}** dengan MAE terkecil.")
    st.write("Tabel Hasil Uji MAE dari LOOCV (diurutkan):")
    st.dataframe(results_df.style.highlight_min(subset=['MAE'], color='lightgreen'), width='stretch')
    
    return best_power, results_df

# --- Fungsi membaca data ---
# Menerima file yang diunggah (upload).
# Mengecek jenis file:
# Jika .csv → dibaca dengan pd.read_csv(). Jika .xlsx atau .xls → dibaca dengan pd.read_excel()
# Mengembalikan hasil sebagai DataFrame Pandas
# Menampilkan pesan error di Streamlit jika format tidak didukung atau file gagal dibaca.
def load_data(uploaded_file):
    """Membaca file CSV atau Excel dan mengembalikannya sebagai DataFrame."""
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'csv': return pd.read_csv(uploaded_file)
        elif file_ext in ['xlsx', 'xls']: return pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap gunakan CSV atau XLSX.")
            return None
    except Exception as e:
        st.error(f"Gagal memuat file: {e}"); return None

# --- Konfigurasi Halaman Streamlit ---
# untuk mengatur layout halaman menjadi lebar penuh
st.set_page_config(layout="wide", page_title="Prediksi Harga Tanah IDW")
st.title("🏞️ Prediksi Harga Tanah dengan Metode-IDW")

# Inisialisasi session state untuk menyimpan nilai power yang akan digunakan untuk prediksi
if 'power_for_prediction' not in st.session_state:
    st.session_state.power_for_prediction = 2.0

# --- Sidebar ---
# Mengunggah data sumber dan data prediksi,
# Mengatur parameter IDW (seperti power dan grid resolution),
# Menentukan tampilan visualisasi hasil (marker atau heatmap).
with st.sidebar:
    # Untuk mengunggah file sumber dan file prediksi
    st.header("⚙️ Pengaturan Aplikasi")
    uploaded_file = st.file_uploader("1. Unggah File Sumber ('Longitude', 'Latitude', 'Harga')", type=["csv", "xlsx"])
    prediction_file = st.file_uploader("2. Unggah File Koordinat Prediksi ('Longitude', 'Latitude')", type=["csv", "xlsx"])

    st.markdown("---")
    st.header("🛠️ Pengaturan Prediksi")

    # Fungsi callback untuk tombol "Gunakan Power Terbaik"
    def set_best_power(value):
        st.session_state.power_for_prediction = value

    # Slider dihubungkan dengan session state
    st.session_state.power_for_prediction = st.slider(
        "Power IDW untuk Prediksi Final", 0.5, 10.0, st.session_state.power_for_prediction, 0.5
    )

    grid_resolution = st.slider("Resolusi Grid Prediksi Otomatis", 20, 100, 50, 10, help="Jumlah titik di setiap sisi grid. Nilai lebih tinggi lebih detail tapi lebih lambat.")

    # Menu visualisasi
    st.markdown("---")
    st.header("🎨 Pengaturan Visualisasi")
    viz_mode = st.radio("Mode Visualisasi Prediksi:", ["Circle Marker", "Heatmap"])

# --- Logika Aplikasi Utama ---
# Memuat data sumber yang harus bersisi kolom 'Longitude', 'Latitude', dan 'Harga'. Jika tidak ada, tampilkan pesan error.
if uploaded_file is not None:
    df_sumber = load_data(uploaded_file)
    required_cols = ["Longitude", "Latitude", "Harga"]
    if df_sumber is not None and all(col in df_sumber.columns for col in required_cols):
        
        # --- LANGKAH PENTING: Pembersihan Data --- 
        # Menghapus baris dengan data kosong pada kolom esensial
        initial_rows = len(df_sumber)
        df_sumber.dropna(subset=required_cols, inplace=True)
        cleaned_rows = len(df_sumber)
        if initial_rows > cleaned_rows:
            st.success(f"🧹 Data telah dibersihkan. {initial_rows - cleaned_rows} baris dengan data kosong pada kolom esensial telah dihapus.")

        # --- Bagian Analisis ---
        # Menemukan power terbaik dengan LOOCV dan power terbaik bisa digunakan untuk prediksi
        st.header("Analisis Model IDW")
        best_power_cv, cv_results = find_best_power_loocv(df_sumber)
        
        # --- Menampilkan akurasi model berdasarkan LOOCV ---
        if cv_results is not None:
            st.subheader("📊 Akurasi Model IDW (berdasarkan LOOCV)")

            # gunakan power terbaik
            errors, _ = run_loocv_for_power(df_sumber, best_power_cv)

            # hitung prediksi LOOCV
            actual_values = df_sumber['Harga'].values
            predicted_values = actual_values - np.array(errors)  # karena error = |actual - pred|
            
            mae, rmse, mape, r2 = evaluate_idw_accuracy(actual_values, predicted_values)

            colA, colB, colC, colD = st.columns(4)
            colA.metric("MAE", f"{mae:,.2f}")
            colB.metric("RMSE", f"{rmse:,.2f}")
            colC.metric("MAPE", f"{mape:.2f}%")
            colD.metric("R² Score", f"{r2:.4f}")

            st.info(f"Pengukuran akurasi ini menggunakan **power terbaik = {best_power_cv}** berdasarkan LOOCV.")

        # Tombol untuk mengadopsi power terbaik
        if cv_results is not None:
            st.sidebar.button("Gunakan Power Terbaik untuk Prediksi", on_click=set_best_power, args=(best_power_cv,), type="primary")

        st.markdown("---")
        st.header("Prediksi dan Visualisasi")

        # --- Bagian Logika Prediksi ---
        points_to_predict = None
        # Secara manual membuat titik prediksi dari file yang diunggah user
        if prediction_file is not None: 
            df_predict = load_data(prediction_file) # Membaca file prediksi
            if df_predict is not None and all(col in df_predict.columns for col in ["Longitude", "Latitude"]): # Cek kolom yang diperlukan
                st.subheader("2. Prediksi pada Koordinat yang Diunggah")
                points_to_predict = df_predict.dropna(subset=["Longitude", "Latitude"]) # Hapus baris dengan data kosong
        else:
            # Jika pengguna tidak mengunggah file koordinat prediksi, aplikasi otomatis akan membuat grid titik prediksi berdasarkan data sumber
            st.subheader("2. Prediksi pada Grid Otomatis")
            padding = 0.05
            min_lon, max_lon = df_sumber['Longitude'].min()-padding, df_sumber['Longitude'].max()+padding
            min_lat, max_lat = df_sumber['Latitude'].min()-padding, df_sumber['Latitude'].max()+padding
            lon_grid, lat_grid = np.meshgrid(np.linspace(min_lon, max_lon, grid_resolution),
                                             np.linspace(min_lat, max_lat, grid_resolution))
            points_to_predict = pd.DataFrame({'Longitude': lon_grid.flatten(), 'Latitude': lat_grid.flatten()})

        # --- Menjalankan Prediksi dan Menampilkan Hasil ---
        # prediksi hanya dijalankan kalau sudah ada titik koordinat (dari upload atau grid otomatis)
        if points_to_predict is not None and not points_to_predict.empty:
            with st.spinner(f"⏳ Memprediksi harga pada {len(points_to_predict)} titik dengan Power={st.session_state.power_for_prediction:.2f}..."):
                predicted_prices = idw_interpolation(df_sumber, points_to_predict, st.session_state.power_for_prediction) 
                points_to_predict['Harga_Prediksi'] = predicted_prices
            st.success("✅ Prediksi selesai.")

            st.write("Pratinjau Hasil Prediksi:")
            st.dataframe(points_to_predict.head())

            # --- Bagian Visualisasi Peta ---
            # Menampilkan peta interaktif dengan data sumber dan hasil prediksi
            st.subheader("3. 🗺️ Visualisasi Peta")
            center_lat, center_lon = df_sumber['Latitude'].mean(), df_sumber['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            # Legenda Warna
            min_price = points_to_predict['Harga_Prediksi'].min()
            max_price = points_to_predict['Harga_Prediksi'].max()
            colormap = cm.linear.YlOrRd_09.scale(min_price, max_price) # Warna dari kuning ke merah
            colormap.caption = 'Prediksi Harga Tanah (Rp)' 
            m.add_child(colormap)

            # Layer Data Sumber
            cluster = MarkerCluster(name="Data Sumber").add_to(m) # Mengelompokkan marker
            # Menambahkan marker untuk tiap titik data sumber
            for _, row in df_sumber.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']], 
                    popup=f"Harga Asli: Rp {row['Harga']:,.0f}",
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(cluster)

            # Layer Data Prediksi
            if viz_mode == "Circle Marker": # Jika mode visualisasi adalah Circle Marker maka tambahkan marker lingkaran dan tooltip untuk tiap titik prediksi dengan menampilan harga prediksi
                pred_layer = folium.FeatureGroup(name="Prediksi (Titik)").add_to(m)
                for _, row in points_to_predict.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']], radius=4,
                        color=colormap(row['Harga_Prediksi']), fill=True, fill_color=colormap(row['Harga_Prediksi']),
                        fill_opacity=0.7, tooltip=f"Prediksi: Rp {row['Harga_Prediksi']:,.0f}"
                    ).add_to(pred_layer)
            elif viz_mode == "Heatmap": # Jika mode visualisasi adalah Heatmap maka tambahkan layer heatmap
                heat_data = [[row['Latitude'], row['Longitude'], row['Harga_Prediksi']] for _, row in points_to_predict.iterrows()]
                HeatMap(heat_data, radius=10, blur=15, name="Prediksi (Heatmap)").add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, height=700, width='stretch') 

            # --- Bagian Download ---
            st.subheader("4. ⬇️ Download Hasil Prediksi")
            col_dl1, col_dl2 = st.columns(2)
            file_name = col_dl1.text_input("Nama File untuk Unduhan:", "prediksi_harga_tanah")
            file_type = col_dl2.radio("Format File:", ('CSV', 'Excel'), horizontal=True)
            
            # Tombol unduh hasil prediksi dalam format yang dipilih
            if file_type == 'CSV': # jika fomrat CSV maka unduh sebagai CSV ubah dengan encoding utf-8
                csv_data = points_to_predict.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh sebagai CSV", data=csv_data, file_name=f"{file_name}.csv", mime='text/csv')  
            else: # Excel # jika format Excel maka unduh sebagai Excel dengan menggunakan BytesIO dan library openpyxl
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    points_to_predict.to_excel(writer, index=False, sheet_name='Prediksi Harga')
                st.download_button("Unduh sebagai Excel", data=output.getvalue(),
                                   file_name=f"{file_name}.xlsx",
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        st.error("Gagal memuat data atau file sumber tidak memiliki kolom yang diperlukan: 'Longitude', 'Latitude', 'Harga'.")
else:
    st.info("Silakan unggah file sumber pada sidebar untuk memulai analisis dan prediksi.")