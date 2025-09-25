"""
Code ini digunakan untuk memprediksi harga tanah dengan IDW,
dengan menggunakan dataset sumber yang diunggah user,
dan memprediksi harga pada koordinat baru yang diunggah user atau pada grid otomatis.

Fitur Utama:
1. Penentuan parameter 'power' terbaik secara otomatis menggunakan Leave-One-Out Cross-Validation (LOOCV).
2. Perbandingan signifikansi statistik antara dua nilai 'power' menggunakan Uji T Berpasangan (Paired t-test).
3. Visualisasi hasil prediksi di peta interaktif (Circle Marker atau Heatmap) dengan legenda dinamis.
"""

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
def idw_interpolation(data_points, grid_points, power=2):
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
    """Menjalankan Leave-One-Out Cross-Validation untuk nilai power tertentu.
    Hasil dari fungsi ini akan disimpan di cache untuk performa.
    """
    errors = []
    # Konversi ke NumPy sekali saja untuk efisiensi
    source_coords = _df_source[['Longitude', 'Latitude']].values
    source_values = _df_source['Harga'].values

    for i in range(len(_df_source)):
        # Siapkan data train dan test menggunakan slicing NumPy, jauh lebih cepat dari df.drop()
        df_train_temp = _df_source.drop(i) # df.drop masih lebih mudah untuk struktur data
        df_test_temp = _df_source.iloc[[i]]

        predicted_value = idw_interpolation(df_train_temp, df_test_temp[['Longitude', 'Latitude']], power)
        
        true_value = source_values[i]
        error = abs(true_value - predicted_value[0])
        errors.append(error)
        
    mae = np.mean(errors)
    return errors, mae

# --- Fungsi mencari power terbaik dengan LOOCV ---
def find_best_power_loocv(df_source):
    st.subheader("1. Penentuan Parameter Power IDW Terbaik (via LOOCV)")
    
    MIN_SAMPLES_REQUIRED = 10
    if len(df_source) < MIN_SAMPLES_REQUIRED:
        st.warning(f"Data sumber Anda hanya memiliki {len(df_source)} baris. Minimal {MIN_SAMPLES_REQUIRED} diperlukan.")
        st.info("Parameter Power default (2.0) akan digunakan.")
        return 2.0, None

    with st.spinner("⏳ Melakukan LOOCV untuk mencari Power terbaik (panggilan pertama mungkin lambat)..."):
        powers_to_test = np.arange(0.5, 5.5, 0.5)
        mae_results = []

        progress_bar = st.progress(0)
        for i, p in enumerate(powers_to_test):
            _, mae = run_loocv_for_power(df_source, power=p) # Fungsi ini sekarang di-cache
            mae_results.append({'Power': p, 'MAE': mae})
            progress_bar.progress((i + 1) / len(powers_to_test))
        
        results_df = pd.DataFrame(mae_results).sort_values(by='MAE').reset_index(drop=True)
        best_power = results_df.loc[0, 'Power']

    st.success(f"✅ Power terbaik = **{best_power:.2f}** dengan MAE terkecil.")
    st.write("Tabel Hasil Uji MAE dari LOOCV (diurutkan):")
    st.dataframe(results_df.style.highlight_min(subset=['MAE'], color='lightgreen'))
    
    return best_power, results_df

# --- Fungsi untuk melakukan Uji T Berpasangan ---
def perform_t_test(df_source, cv_results):
    st.subheader("2. Uji Signifikansi Statistik (Uji T)")
    st.info("Bandingkan dua nilai Power untuk melihat apakah perbedaan kinerjanya signifikan.")
    
    with st.expander("Klik untuk melakukan Uji T"):
        # Nilai default yang lebih cerdas
        default_power_a = cv_results.loc[0, 'Power'] if cv_results is not None else 2.0
        default_power_b = cv_results.loc[1, 'Power'] if cv_results is not None and len(cv_results) > 1 else 3.0

        col1, col2 = st.columns(2)
        power_a = col1.number_input("Power A", 0.1, 10.0, default_power_a, 0.5, key="power_a")
        power_b = col2.number_input("Power B", 0.1, 10.0, default_power_b, 0.5, key="power_b")

        if st.button("🔬 Lakukan Uji T Berpasangan"):
            if power_a == power_b:
                st.error("Power A dan Power B tidak boleh sama.")
            else:
                with st.spinner(f"Menjalankan Uji T (akan instan jika nilai sudah di-cache)..."):
                    errors_a, mae_a = run_loocv_for_power(df_source, power_a)
                    errors_b, mae_b = run_loocv_for_power(df_source, power_b)
                    t_statistic, p_value = ttest_rel(errors_a, errors_b)

                st.markdown("---")
                st.write("#### Hasil Uji T:")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric(f"Rata-rata Error (MAE) Power {power_a}", f"{mae_a:,.2f}")
                res_col2.metric(f"Rata-rata Error (MAE) Power {power_b}", f"{mae_b:,.2f}")
                res_col3.metric("P-value", f"{p_value:.4f}")

                alpha = 0.05
                if p_value < alpha:
                    winner = f"Power {power_a}" if mae_a < mae_b else f"Power {power_b}"
                    st.success(f"**Kesimpulan:** Perbedaan kinerja **signifikan secara statistik** (p < {alpha}). Model dengan **{winner}** terbukti lebih baik.")
                else:
                    st.warning(f"**Kesimpulan:** Perbedaan kinerja **tidak signifikan** (p ≥ {alpha}). Tidak ada cukup bukti untuk menyatakan satu model lebih baik.")

# --- Fungsi membaca data ---
def load_data(uploaded_file):
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'csv': return pd.read_csv(uploaded_file)
        elif file_ext in ['xlsx', 'xls']: return pd.read_excel(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap gunakan CSV atau XLSX.")
            return None
    except Exception as e:
        st.error(f"Gagal memuat file: {e}"); return None

# --- Streamlit page ---
st.set_page_config(layout="wide")
st.title("🗺️ Prediksi Harga Tanah: IDW")

# Inisialisasi session state untuk power
if 'power_for_prediction' not in st.session_state:
    st.session_state.power_for_prediction = 2.0

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan Aplikasi")
uploaded_file = st.sidebar.file_uploader("1. Unggah File Sumber ('Longitude', 'Latitude', 'Harga')", type=["csv", "xlsx"])
prediction_file = st.sidebar.file_uploader("2. (Opsional) Unggah File Koordinat Prediksi", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Pengaturan Prediksi")

def update_power_slider(value):
    st.session_state.power_for_prediction = value

power_for_prediction = st.sidebar.slider("Power IDW untuk Prediksi Final", 0.5, 10.0, st.session_state.power_for_prediction, 0.5, key="power_slider")

grid_resolution = st.sidebar.slider("Resolusi Grid Prediksi Otomatis", 20, 100, 50, 10, help="Jumlah titik di setiap sisi grid. Nilai lebih tinggi lebih detail tapi lebih lambat.")

st.sidebar.markdown("---")
st.sidebar.header("🎨 Pengaturan Visualisasi")
viz_mode = st.sidebar.radio("Mode Visualisasi Prediksi:", ["Circle Marker", "Heatmap"])

# --- Main Application Logic ---
if uploaded_file is not None:
    df_sumber = load_data(uploaded_file)
    if df_sumber is not None and all(col in df_sumber.columns for col in ["Longitude", "Latitude", "Harga"]):
        
        st.header("Analisis Model IDW")
        best_power_cv, cv_results = find_best_power_loocv(df_sumber)
        
        st.sidebar.button("Gunakan Power Terbaik untuk Prediksi", on_click=update_power_slider, args=(best_power_cv,))

        perform_t_test(df_sumber, cv_results)
        
        st.markdown("---")
        st.header("Prediksi dan Visualisasi")

        points_to_predict = None
        if prediction_file is not None:
            df_predict = load_data(prediction_file)
            if df_predict is not None and all(col in df_predict.columns for col in ["Longitude", "Latitude"]):
                st.subheader("3. Prediksi pada Koordinat yang Diunggah")
                points_to_predict = df_predict
        else:
            st.subheader("3. Prediksi pada Grid Otomatis")
            padding = 0.05
            min_lon, max_lon = df_sumber['Longitude'].min()-padding, df_sumber['Longitude'].max()+padding
            min_lat, max_lat = df_sumber['Latitude'].min()-padding, df_sumber['Latitude'].max()+padding
            lon_grid, lat_grid = np.meshgrid(np.linspace(min_lon, max_lon, grid_resolution),
                                             np.linspace(min_lat, max_lat, grid_resolution))
            points_to_predict = pd.DataFrame({'Longitude': lon_grid.flatten(), 'Latitude': lat_grid.flatten()})

        if points_to_predict is not None and not points_to_predict.empty:
            with st.spinner(f"⏳ Memprediksi harga pada {len(points_to_predict)} titik dengan Power={power_for_prediction}..."):
                predicted_prices = idw_interpolation(df_sumber, points_to_predict, power_for_prediction)
                points_to_predict['Harga_Prediksi'] = predicted_prices
                st.success("✅ Prediksi selesai.")


            st.write("Pratinjau Hasil Prediksi:")
            st.dataframe(points_to_predict.head())

            st.subheader("4. 🗺️ Visualisasi Peta")
            center_lat, center_lon = df_sumber['Latitude'].mean(), df_sumber['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

            min_price = points_to_predict['Harga_Prediksi'].min()
            max_price = points_to_predict['Harga_Prediksi'].max()
            colormap = cm.linear.YlOrRd_09.scale(min_price, max_price)
            colormap.caption = 'Prediksi Harga Tanah (Rp)'
            m.add_child(colormap)

            cluster = MarkerCluster(name="Data Sumber").add_to(m)
            for _, row in df_sumber.iterrows():
                folium.Marker(location=[row['Latitude'], row['Longitude']], popup=f"Harga Asli: Rp {row['Harga']:,.0f}").add_to(cluster)

            if viz_mode == "Circle Marker":
                pred_layer = folium.FeatureGroup(name="Prediksi (Titik)").add_to(m)
                for _, row in points_to_predict.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']], radius=4,
                        color=colormap(row['Harga_Prediksi']), fill=True, fill_color=colormap(row['Harga_Prediksi']),
                        fill_opacity=0.7, tooltip=f"Prediksi: Rp {row['Harga_Prediksi']:,.0f}"
                    ).add_to(pred_layer)
            elif viz_mode == "Heatmap":
                heat_data = [[row['Latitude'], row['Longitude'], row['Harga_Prediksi']] for _, row in points_to_predict.iterrows()]
                HeatMap(heat_data, radius=10, blur=15, name="Prediksi (Heatmap)").add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=None, height=700, use_container_width=True)

            st.subheader("5. ⬇️ Download Hasil Prediksi")
            file_name = st.text_input("Nama File untuk Unduhan: ", "prediksi_harga_tanah")
            file_type = st.radio("Format File: ", ('CSV', 'Excel'))
            if file_type == 'CSV':
                csv_data = points_to_predict.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh sebagai CSV", data=csv_data, file_name=f"{file_name}.csv", mime='text/csv')   
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    points_to_predict.to_excel(writer, index=False, sheet_name='Prediksi Harga')
                st.download_button("Unduh sebagai Excel", data=output.getvalue(),
                                    file_name=f"{file_name}.xlsx",
                                      mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        st.error("Gagal memuat data atau file sumber tidak memiliki kolom yang diperlukan: 'Longitude', 'Latitude', 'Harga'.")
else:
    st.info("Silakan unggah file sumber untuk memulai analisis dan prediksi.")
