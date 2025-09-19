""" Code ini digunakan untuk memprediksi harga tanah dengan IDW,
dengan menggunakan dataset sumber yang diunggah user,
dan memprediksi harga pada koordinat baru yang diunggah user atau pada grid otomatis,
dengan power yang diatur otomatis melalui cross-validation,
dan menampilkan hasil prediksi dengan beberapa mode visualisasi di peta (point, heatmap)."""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import openpyxl
from io import BytesIO
import branca.element as br_element
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Fungsi untuk menghitung RMSE ---
def rmse(y_true, y_pred):
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

# --- Fungsi mencari power terbaik ---
def find_best_power(df_source):
    st.markdown("---")
    st.subheader("1. Penentuan Parameter Power IDW Terbaik")
    
    MIN_SAMPLES_REQUIRED = 50
    if len(df_source) < MIN_SAMPLES_REQUIRED:
        st.warning(f"Data sumber Anda hanya memiliki {len(df_source)} baris. Minimal {MIN_SAMPLES_REQUIRED} baris diperlukan.")
        st.info("Parameter Power default (2.0) akan digunakan.")
        return 2.0, None

    with st.spinner("⏳ Melakukan cross-validation untuk mencari Power terbaik..."):
        df_shuffled = df_source.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(len(df_shuffled) * 0.8)
        df_train = df_shuffled.iloc[:train_size]
        df_test = df_shuffled.iloc[train_size:]
        
        if df_test.empty:
            st.error("Set data validasi kosong setelah pembagian.")
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

    st.success(f"✅ Power terbaik = {best_power:.2f} dengan RMSE terkecil.")
    st.write("Tabel Hasil Uji RMSE:")
    st.dataframe(results_df.style.highlight_min(subset=['RMSE'], color='lightgreen'))
    
    return best_power, results_df

# --- Fungsi membaca data ---
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

# --- Fungsi warna prediksi ---
def get_color(price, p33, p66):
    if price < p33: return 'green'
    elif p33 <= price < p66: return 'orange'
    else: return 'red'

# --- Streamlit page ---
st.set_page_config(layout="wide")
st.title("🗺️ Prediksi Harga Tanah Menggunakan IDW")

# --- Sidebar ---
st.sidebar.header("⚙️ Pengaturan Aplikasi")
uploaded_file = st.sidebar.file_uploader("File sumber ('Longitude', 'Latitude', 'Harga')", type=["csv", "xlsx"])
prediction_file = st.sidebar.file_uploader("File koordinat prediksi (Opsional)", type=["csv", "xlsx"])
st.sidebar.markdown("---")
power = st.sidebar.slider("Eksponen (Power) IDW", 0.5, 10.0, 2.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visualisasi")
viz_mode = st.sidebar.radio("Mode Visualisasi:", ["Circle Marker", "Heatmap"])
threshold_low = st.sidebar.slider("Threshold Murah (%)", 10, 50, 33)
threshold_high = st.sidebar.slider("Threshold Mahal (%)", 50, 90, 66)

# --- Main ---
if uploaded_file:
    st.session_state.df_sumber = load_data(uploaded_file)
elif 'df_sumber' in st.session_state:
    del st.session_state.df_sumber

if 'df_sumber' in st.session_state and st.session_state.df_sumber is not None:
    df_sumber = st.session_state.df_sumber
    
    if all(col in df_sumber.columns for col in ["Longitude", "Latitude", "Harga"]):
        best_power_cv, cv_results = find_best_power(df_sumber)
        st.info(f"💡 Power hasil cross-validation: **{best_power_cv:.2f}**")

        st.subheader("2. Pratinjau Data Sumber")
        st.dataframe(df_sumber.head())

        # Data prediksi
        if prediction_file is not None:
            df_predict = load_data(prediction_file)
            if df_predict is not None and all(col in df_predict.columns for col in ["Longitude", "Latitude"]):
                points_to_predict = df_predict
        else:
            padding = 0.05
            min_lon, max_lon = df_sumber['Longitude'].min()-padding, df_sumber['Longitude'].max()+padding
            min_lat, max_lat = df_sumber['Latitude'].min()-padding, df_sumber['Latitude'].max()+padding
            lon_grid, lat_grid = np.meshgrid(np.arange(min_lon, max_lon, 0.01),
                                             np.arange(min_lat, max_lat, 0.01))
            points_to_predict = pd.DataFrame({'Longitude': lon_grid.flatten(), 'Latitude': lat_grid.flatten()})

        if points_to_predict is not None and not points_to_predict.empty:
            with st.spinner(f"⏳ Prediksi harga dengan Power={power}..."):
                predicted_prices = idw_interpolation(df_sumber, points_to_predict, power)
                points_to_predict['Harga_Prediksi'] = predicted_prices

            st.subheader("3. 📊 Hasil Prediksi")
            st.dataframe(points_to_predict.head())

            st.subheader("4. 🗺️ Visualisasi Peta")
            center_lat, center_lon = points_to_predict['Latitude'].mean(), points_to_predict['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

            # Threshold interaktif
            p33 = np.percentile(points_to_predict['Harga_Prediksi'], threshold_low)
            p66 = np.percentile(points_to_predict['Harga_Prediksi'], threshold_high)

            # Layer data sumber (cluster)
            cluster = MarkerCluster(name="Data Sumber").add_to(m)
            for _, row in df_sumber.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Harga Asli: Rp {row['Harga']:,.0f}"
                ).add_to(cluster)

            # Layer prediksi
            if viz_mode == "Circle Marker":
                pred_layer = folium.FeatureGroup(name="Prediksi (Circle Marker)").add_to(m)
                for _, row in points_to_predict.iterrows():
                    color = get_color(row['Harga_Prediksi'], p33, p66)
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']], radius=4,
                        color=color, fill=True, fill_opacity=0.6,
                        tooltip=f"Prediksi: Rp {row['Harga_Prediksi']:,.0f}"
                    ).add_to(pred_layer)

            elif viz_mode == "Heatmap":
                pred_layer = folium.FeatureGroup(name="Prediksi (Heatmap)").add_to(m)
                heat_data = [[row['Latitude'], row['Longitude'], row['Harga_Prediksi']]
                             for _, row in points_to_predict.iterrows()]
                HeatMap(heat_data, radius=10, blur=15).add_to(pred_layer)

            # Legend manual
            legend_html = f'''
            <div style="position: fixed; bottom: 50px; left: 50px; width: 220px;
            border:2px solid grey; z-index:9999; font-size:14px; background:white;">
            &nbsp;<b>Legenda Harga Prediksi (Rp)</b><br>
            &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp; Mahal (&gt; {p66:,.0f})<br>
            &nbsp;<i class="fa fa-circle" style="color:orange"></i>&nbsp; Sedang<br>
            &nbsp;<i class="fa fa-circle" style="color:green"></i>&nbsp; Murah (&lt; {p33:,.0f})
            </div>'''
            m.get_root().html.add_child(br_element.Element(legend_html))

            folium.LayerControl().add_to(m)
            if viz_mode in ["Circle Marker", "Heatmap"]:
                st_folium(m, width=1200, height=700)

            # Download
            st.subheader("5. ⬇️ Download Hasil Prediksi")
            file_name = st.text_input("Nama file:", "prediksi_harga_tanah")
            file_type = st.radio("Format file:", ('CSV', 'Excel'))
            if file_type == 'CSV':
                csv_data = points_to_predict.to_csv(index=False).encode('utf-8')
                st.download_button("Unduh CSV", data=csv_data, file_name=f"{file_name}.csv", mime="text/csv")
            else:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    points_to_predict.to_excel(writer, index=False, sheet_name='Prediksi Harga')
                st.download_button("Unduh Excel", data=output.getvalue(),
                                   file_name=f"{file_name}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    else:
        st.error("File sumber harus punya kolom: Longitude, Latitude, Harga.")
else:
    st.info("Silakan unggah file sumber untuk memulai.")
