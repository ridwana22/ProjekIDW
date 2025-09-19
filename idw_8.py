""" Code ini digunaknan untuk memprediksi harga tanah dengan IDW,
dengan menggunakan dataset sumber yang diunggah user,
dan memprediksi harga pada koordinat baru yang diunggah user atau pada grid otomatis,
dengan power yang diatur otomatis melalui cross-validation,
dan menampilkan hasil prediksi dengan tiga warna di peta."""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import openpyxl
from io import BytesIO
import branca.element as br_element

# --- Fungsi untuk menghitung RMSE ---
def rmse(y_true, y_pred):
    """Menghitung Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- Fungsi untuk menghitung IDW ---
def idw_interpolation(data_points, grid_points, power=2):
    known_coords = data_points[['Longitude', 'Latitude']].values
    known_values = data_points['Harga'].values
    grid_coords = grid_points[['Longitude', 'Latitude']].values
    predicted_values = np.zeros(len(grid_coords))
    
    for i, grid_point in enumerate(grid_coords):
        distances = np.sqrt(np.sum((known_coords - grid_point)**2, axis=1))
        if np.any(distances == 0):
            zero_dist_indices = np.where(distances == 0)[0]
            predicted_values[i] = known_values[zero_dist_indices[0]]
        else:
            weights = 1.0 / (distances ** power)
            predicted_values[i] = np.sum(weights * known_values) / np.sum(weights)
    return predicted_values

def find_best_power(df_source):
    st.markdown("---")
    st.subheader("1. Penentuan Parameter Power IDW Terbaik")
    
    # --- PERUBAHAN 1: Kondisi jumlah data minimum ---
    # Mengubah batas minimum agar lebih fleksibel untuk pembagian persentase
    MIN_SAMPLES_REQUIRED = 50 
    if len(df_source) < MIN_SAMPLES_REQUIRED:
        st.warning(f"Data sumber Anda hanya memiliki {len(df_source)} baris. Diperlukan minimal {MIN_SAMPLES_REQUIRED} baris untuk melakukan cross-validation.")
        st.info("Parameter Power default (2.0) akan digunakan.")
        return 2.0, None

    with st.spinner("⏳ Melakukan cross-validation untuk mencari Power terbaik..."):
        df_shuffled = df_source.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # --- PERUBAHAN 2: Pembagian data 80% training dan 20% testing ---
        # Hitung titik pembagian untuk 80% data training
        train_size = int(len(df_shuffled) * 0.8)
        
        # Bagi dataframe berdasarkan persentase
        df_train = df_shuffled.iloc[:train_size]
        df_test = df_shuffled.iloc[train_size:]
        
        # Pastikan test set tidak kosong
        if df_test.empty:
            st.error("Gagal melakukan cross-validation karena set data validasi (20%) kosong setelah pembagian.")
            st.info("Parameter Power default (2.0) akan digunakan.")
            return 2.0, None

        test_points = df_test[['Longitude', 'Latitude']]
        true_values = df_test['Harga'].values
        powers_to_test = np.arange(0.5, 5.5, 0.5)
        rmse_results = []

        for p in powers_to_test:
            predicted_prices = idw_interpolation(df_train, test_points, power=p)
            error = rmse(true_values, predicted_prices)
            rmse_results.append({'Power': p, 'RMSE': error})
        
        results_df = pd.DataFrame(rmse_results)
        best_result = results_df.loc[results_df['RMSE'].idxmin()]
        best_power = best_result['Power']

    st.success(f"✅ Cross-validation selesai! **Power terbaik adalah {best_power:.2f}** dengan RMSE terkecil.")
    # --- PERUBAHAN 3 (Opsional): Menampilkan informasi pembagian data ---
    st.info(f"Data dibagi menjadi **{len(df_train)}** baris untuk training (80%) dan **{len(df_test)}** baris untuk validasi (20%).")
    st.write("Tabel Hasil Uji RMSE:")
    st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='lightgreen'))
    
    return best_power, results_df

# --- Fungsi untuk membaca file unggahan (Disederhanakan) ---
# --- PERBAIKAN: Logika session_state dipindahkan ke proses utama agar lebih mudah dikelola ---
def load_data(uploaded_file):
    try:
        file_ext = uploaded_file.name.split('.')[-1]
        if file_ext == 'csv':
            return pd.read_csv(uploaded_file)
        elif file_ext == 'xlsx':
            return pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung.")
            return None
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return None

# --- Fungsi untuk menentukan warna berdasarkan harga ---
def get_color(price, p33, p66):
    if price < p33: return 'green'
    elif p33 <= price < p66: return 'orange'
    else: return 'red'

# --- Pengaturan Halaman Streamlit ---
st.set_page_config(layout="wide")
st.title("🗺️ Prediksi Harga Tanah Menggunakan IDW")
st.markdown("Aplikasi ini melakukan prediksi harga tanah dan memvisualisasikannya di peta, dengan parameter IDW yang dioptimalkan melalui cross-validation.")

# --- Bagian Sidebar ---
st.sidebar.header("⚙️ Pengaturan Aplikasi")
st.sidebar.subheader("1. Unggah Data")
uploaded_file = st.sidebar.file_uploader("File sumber ('Longitude', 'Latitude', 'Harga')", type=["csv", "xlsx"])
prediction_file = st.sidebar.file_uploader("File koordinat untuk diprediksi (Opsional)", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Parameter IDW")
power = st.sidebar.slider("Eksponen (Power) IDW", 0.5, 10.0, 2.0, 0.5, help="Nilai ini dapat diatur manual, atau gunakan nilai terbaik hasil cross-validation.")

# --- Proses Utama Aplikasi ---

if uploaded_file:
    # Simpan data ke session state agar tidak hilang saat ada interaksi widget
    st.session_state.df_sumber = load_data(uploaded_file)
elif 'df_sumber' in st.session_state:
    # Hapus data dari state jika file di-unselect
    del st.session_state.df_sumber

# Memeriksa apakah data sumber ada di session state untuk diproses
if 'df_sumber' in st.session_state and st.session_state.df_sumber is not None:
    df_sumber = st.session_state.df_sumber # Ambil data dari state
    
    if all(col in df_sumber.columns for col in ["Longitude", "Latitude", "Harga"]):
        best_power_cv, cv_results = find_best_power(df_sumber)
        st.info(f"💡 Anda dapat menyesuaikan slider 'Power' di sidebar. Nilai yang disarankan dari hasil uji adalah **{best_power_cv:.2f}**.")
        
        st.markdown("---")
        st.subheader("2. Pratinjau Data Sumber")
        st.dataframe(df_sumber.head())
        
        points_to_predict = None
        
        # --- PERBAIKAN: Logika untuk memuat file prediksi yang diunggah ---
        if prediction_file is not None:
            df_predict = load_data(prediction_file)
            if df_predict is not None and all(col in df_predict.columns for col in ["Longitude", "Latitude"]):
                points_to_predict = df_predict
                st.info("Mode Prediksi: Berdasarkan file koordinat yang diunggah.")
        else:
            st.info("Mode Prediksi: Berdasarkan Grid yang dibuat otomatis dari area data sumber.")
            padding = 0.05
            min_lon, max_lon = df_sumber['Longitude'].min() - padding, df_sumber['Longitude'].max() + padding
            min_lat, max_lat = df_sumber['Latitude'].min() - padding, df_sumber['Latitude'].max() + padding
            lon_grid, lat_grid = np.meshgrid(np.arange(min_lon, max_lon, 0.01), np.arange(min_lat, max_lat, 0.01))
            points_to_predict = pd.DataFrame({'Longitude': lon_grid.flatten(), 'Latitude': lat_grid.flatten()})

        if points_to_predict is not None and not points_to_predict.empty:
            with st.spinner(f"⏳ Melakukan prediksi harga menggunakan Power = {power}..."):
                predicted_prices = idw_interpolation(df_sumber, points_to_predict, power)
                points_to_predict['Harga_Prediksi'] = predicted_prices
            
            st.markdown("---")
            st.subheader("3. 📊 Hasil Prediksi Lengkap")
            st.dataframe(points_to_predict)

            st.markdown("---")
            st.subheader("4. 🗺️ Visualisasi Peta")
            
            center_lat, center_lon = points_to_predict['Latitude'].mean(), points_to_predict['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            
            p33 = points_to_predict['Harga_Prediksi'].quantile(0.33)
            p66 = points_to_predict['Harga_Prediksi'].quantile(0.66)
            
            p66_formatted, p33_formatted = f"{p66:,.0f}", f"{p33:,.0f}"
            legend_html = f'''<div style="position: fixed; bottom: 50px; left: 50px; width: 220px; border:2px solid grey; z-index:9999; font-size:14px; background-color:white;">&nbsp; <b>Legenda Harga Prediksi (Rp)</b> <br>&nbsp; <i class="fa fa-circle" style="color:red"></i>&nbsp; Mahal (&gt; {p66_formatted})<br>&nbsp; <i class="fa fa-circle" style="color:orange"></i>&nbsp; Sedang <br>&nbsp; <i class="fa fa-circle" style="color:green"></i>&nbsp; Murah (&lt; {p33_formatted})</div>'''
            m.get_root().html.add_child(br_element.Element(legend_html))

            for _, row in points_to_predict.iterrows():
                color = get_color(row['Harga_Prediksi'], p33, p66)
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']], radius=5, color=color, fill=True,
                    fill_color=color, fill_opacity=0.7, tooltip=f"Prediksi: Rp {row['Harga_Prediksi']:,.0f}"
                ).add_to(m)
            st_folium(m, width=1200, height=700)
            
            # --- PERBAIKAN: Melengkapi logika download file ---
            st.markdown("---")
            st.subheader("5. ⬇️ Download Hasil Prediksi Lengkap")
            col1, col2 = st.columns(2)
            with col1:
                file_name = st.text_input("Masukkan nama file (tanpa ekstensi):", "prediksi_harga_tanah")
            with col2:
                file_type = st.radio("Pilih format file:", ('CSV (.csv)', 'Excel (.xlsx)'))

            if file_type == 'CSV (.csv)':
                csv_data = points_to_predict.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Unduh File CSV",
                    data=csv_data,
                    file_name=f"{file_name}.csv",
                    mime="text/csv"
                )
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    points_to_predict.to_excel(writer, index=False, sheet_name='Prediksi Harga')
                excel_data = output.getvalue()
                st.download_button(
                    label="Unduh File Excel",
                    data=excel_data,
                    file_name=f"{file_name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    elif df_sumber is not None:
        st.error("File data sumber harus memiliki kolom 'Longitude', 'Latitude', dan 'Harga'.")
else:
    st.info("Silakan unggah file data sumber Anda untuk memulai.")