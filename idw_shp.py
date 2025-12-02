# Import Library
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import tempfile
import zipfile
import os
import branca.colormap as cm
from io import BytesIO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="IDW Complete: LOOCV + SHP")
st.title("Prediksi Harga Tanah dengan Inverse Distance Weighting (IDW)")

# --- 1. FUNGSI INTI IDW & LOOCV ---

def idw_interpolation(data_points, grid_points, power=2):
    """Fungsi dasar perhitungan IDW."""
    known_coords = data_points[['Longitude', 'Latitude']].values # Koordinat sumber 
    known_values = data_points['Harga'].values # Nilai sumber
    target_coords = grid_points[['Longitude', 'Latitude']].values # Koordinat target
    
    predicted_values = np.zeros(len(target_coords)) # Tempat hasil prediksi
    
    # Fungsi untuk IDW
    for i, target_point in enumerate(target_coords):
        distances = np.sqrt(np.sum((known_coords - target_point)**2, axis=1))
        if np.any(distances == 0):
            zero_dist_index = np.where(distances == 0)[0][0]
            predicted_values[i] = known_values[zero_dist_index]
        else:
            weights = 1.0 / (distances ** power)
            predicted_values[i] = np.sum(weights * known_values) / np.sum(weights)
    return predicted_values

@st.cache_data
def run_loocv_optimization(df_source):
    """
    Menjalankan LOOCV untuk mencari Power terbaik antara 0.5 s/d 5.0.
    Disimpan di cache agar tidak berat saat reload.
    """
    powers_to_test = np.arange(0.5, 5.5, 0.5)
    best_mae = float('inf')
    best_power = 2.0
    results = []

    # Loop setiap kemungkinan power
    for p in powers_to_test:
        errors = []
        # Loop LOOCV (Leave-One-Out)
        for i in range(len(df_source)):
            train = df_source.drop(df_source.index[i])
            test = df_source.iloc[[i]]
            
            # Prediksi 1 titik yang dibuang
            pred = idw_interpolation(train, test[['Longitude', 'Latitude']], power=p)
            actual = test['Harga'].values[0]
            errors.append(abs(actual - pred[0]))
        
        avg_mae = np.mean(errors)
        results.append({'Power': p, 'MAE': avg_mae})
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_power = p
            
    return best_power, pd.DataFrame(results)

# --- 2. FUNGSI LOAD FILE ---

def load_shapefile(uploaded_zip):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
        if not shp_files: return None
        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
        if gdf.crs != "EPSG:4326": gdf = gdf.to_crs("EPSG:4326")
        return gdf

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): return pd.read_csv(uploaded_file)
        else: return pd.read_excel(uploaded_file)
    except: return None

# --- UI & LOGIKA UTAMA ---

# Inisialisasi Session State
if 'current_power' not in st.session_state:
    st.session_state.current_power = 2.0
if 'optimization_done' not in st.session_state:
    st.session_state.optimization_done = False
if 'best_power_found' not in st.session_state:
    st.session_state.best_power_found = None

with st.sidebar:
    st.header("📂 1. Upload Data")
    f_source = st.file_uploader("Data Sumber (Latih)", type=["csv", "xlsx"])
    f_target = st.file_uploader("Data Koordinat Target", type=["csv", "xlsx"])
    f_shp = st.file_uploader("File Peta (.zip)", type=["zip"])
    
    st.divider()
    st.header("⚙️ 2. Kontrol Power")
    
    # Mode Manual override
    # Kunci: Slider akan mengikuti session state current_power jika di-reset
    manual_power = st.slider(
        "Power Parameter", 
        0.5, 5.0, 
        float(st.session_state.current_power), 
        0.1
    )
    
    # Update session state jika slider digeser user
    if manual_power != st.session_state.current_power:
        st.session_state.current_power = manual_power

# Cek Kelengkapan File
if f_source and f_target and f_shp:
    df_src = load_data(f_source)
    df_tgt = load_data(f_target)
    gdf_shp = load_shapefile(f_shp)
    
    # Validasi Kolom
    if (df_src is not None and {'Longitude','Latitude','Harga'}.issubset(df_src.columns) and
        df_tgt is not None and {'Longitude','Latitude'}.issubset(df_tgt.columns)):
        
        # Bersihkan Data
        df_src = df_src.dropna(subset=['Longitude','Latitude','Harga'])
        
        # --- TAB MENU ---
        tab1, tab2 = st.tabs(["📊 Optimasi Model (LOOCV)", "🌍 Peta Prediksi"])
        
        with tab1:
            st.subheader("Cari Parameter Power Terbaik")
            st.write(f"Power yang sedang digunakan: **{st.session_state.current_power}**")
            
            # Tombol untuk menjalankan LOOCV
            if st.button("🚀 Jalankan Analisis LOOCV", type="primary"):
                with st.spinner("Sedang menguji berbagai nilai Power..."):
                    best_p, res_df = run_loocv_optimization(df_src)
                    
                    # Simpan hasil ke session state
                    st.session_state.best_power_found = best_p
                    st.session_state.current_power = best_p  # Otomatis pakai yang terbaik saat pertama kali run
                    st.session_state.optimization_done = True
                    
                    st.success(f"Analisis Selesai! Power Terbaik: {best_p}")
                    st.rerun() # Rerun agar slider di sidebar terupdate otomatis

            # Tampilkan Hasil jika sudah pernah dijalankan
            if st.session_state.optimization_done:
                st.divider()
                col_res1, col_res2 = st.columns([1, 1])
                
                with col_res1:
                    # Tampilkan Tombol Gunakan Power Terbaik
                    best_val = st.session_state.best_power_found
                    st.info(f"Rekomendasi Power Terbaik: **{best_val}**")
                    
                    # Button khusus untuk revert ke best power
                    if st.button(f"✅ Gunakan Power Terbaik ({best_val})"):
                        st.session_state.current_power = best_val
                        st.rerun() # Refresh halaman untuk update slider dan perhitungan
                
                with col_res2:
                    st.write("**Tabel Error (MAE):**")
                    # Kita panggil lagi fungsinya (akan cepat karena cache) untuk ambil data tabel
                    _, res_df = run_loocv_optimization(df_src)
                    st.dataframe(res_df.style.highlight_min(subset=['MAE'], color='lightgreen'), height=200)

        with tab2:
            st.subheader("Visualisasi Hasil Prediksi")
            
            # --- PROSES PREDIKSI ---
            # Menggunakan Power yang ada di session state (baik dari slider atau hasil tombol best power)
            with st.spinner(f"Menghitung prediksi dengan Power {st.session_state.current_power}..."):
                preds = idw_interpolation(df_src, df_tgt, power=st.session_state.current_power)
                df_tgt['Harga_Prediksi'] = preds
            
            # --- VISUALISASI PETA ---
            m = folium.Map(location=[df_tgt['Latitude'].mean(), df_tgt['Longitude'].mean()], zoom_start=12)
            
            # 1. SHP Layer
            folium.GeoJson(gdf_shp, name="Batas Wilayah",
                           style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'dashArray': '5, 5'}).add_to(m)
            
            # 2. Points Layer
            min_v, max_v = df_tgt['Harga_Prediksi'].min(), df_tgt['Harga_Prediksi'].max()
            cmap = cm.linear.YlOrRd_09.scale(min_v, max_v)
            cmap.caption = f"Prediksi Harga (Power={st.session_state.current_power})"
            m.add_child(cmap)
            
            fg = folium.FeatureGroup(name="Hasil Prediksi")
            for _, row in df_tgt.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=6, color='gray', weight=1, fill=True,
                    fill_color=cmap(row['Harga_Prediksi']), fill_opacity=0.9,
                    tooltip=f"Rp {row['Harga_Prediksi']:,.0f}"
                ).add_to(fg)
            fg.add_to(m)
            
            folium.LayerControl().add_to(m)
            st_folium(m, width="100%", height=600)
            
            # --- DOWNLOAD ---
            st.divider()
            csv = df_tgt.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Prediksi (CSV)", csv, "prediksi_loocv_final.csv", "text/csv")
            
    else:
        st.error("Kolom wajib tidak ditemukan. Pastikan CSV Sumber punya (Longitude, Latitude, Harga) dan Target punya (Longitude, Latitude).")
else:
    st.info("Silakan upload Data Sumber, Data Target, dan File SHP.")