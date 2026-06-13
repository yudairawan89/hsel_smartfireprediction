import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from streamlit_folium import folium_static
import folium
from folium.plugins import MousePosition, Fullscreen
from PIL import Image
import re
import altair as alt
import json
import shap
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import requests
import time
from io import StringIO, BytesIO
import base64
import streamlit.components.v1 as components
import os
import copy
import cv2
from ultralytics import YOLO

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Fire Prediction HSEL & YOLO", page_icon="🔥", layout="wide")

# === INISIALISASI SESSION STATE UNTUK NAVIGASI & MULTIMODAL ===
if "page" not in st.session_state:
    st.session_state.page = "Dashboard Utama"
if "yolo_fire_detected" not in st.session_state:
    st.session_state.yolo_fire_detected = None

# === STYLE KUSTOM ===
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    table {width: 100%; border-collapse: collapse;}
    th, td {border: 1px solid #ddd; padding: 8px;}
    th {background-color: #e0e0e0; text-align: center;}
    td {text-align: center;}
    .section-title {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 6px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .scrollable-table { overflow-x: auto; }
    
    /* Frame khusus untuk iframe Folium */
    iframe[title="folium_static"] {
        border: 4px solid #444 !important;
        border-radius: 4px !important;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# === FUNGSI BANTUAN ===
def convert_day_to_indonesian(day_name):
    return {
        'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
        'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {
        'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
        'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
        'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
        'November': 'November', 'December': 'Desember'
    }.get(month_name, month_name)

def convert_to_label(pred):
    return {
        0: "Low / Rendah",
        1: "Moderate / Sedang",
        2: "High / Tinggi",
        3: "Very High / Sangat Tinggi"
    }.get(pred, "Unknown")

risk_styles = {
    "Low / Rendah": ("white", "blue"),
    "Moderate / Sedang": ("white", "green"),
    "High / Tinggi": ("black", "yellow"),
    "Very High / Sangat Tinggi": ("white", "red")
}

# === LOAD MODEL (IoT & YOLO) ===
@st.cache_resource
def load_model(): return joblib.load("HSEL_IoT_Model.joblib")

@st.cache_resource
def load_scaler(): return joblib.load("scaler.joblib")

@st.cache_resource
def load_sastrawi():
    stop_factory = StopWordRemoverFactory()
    stopword_remover = stop_factory.create_stop_word_remover()
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    return stopword_remover, stemmer

@st.cache_resource
def load_text_models():
    vec = joblib.load("tfidf_vectorizer.joblib")
    mdl = joblib.load("stacking_text_model.joblib")
    return vec, mdl

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        return None

# Eksekusi instansiasi model
model = load_model()
scaler = load_scaler()
stopword_remover, stemmer = load_sastrawi()
yolo_model = load_yolo_model()

try:
    vectorizer, model_text = load_text_models()
except Exception:
    vectorizer, model_text = None, None

@st.cache_data
def load_riau_geojson():
    try:
        with open("Provinsi Riau-KAB_KOTA.geojson", "r") as f:
            return json.load(f)
    except Exception:
        return None

# === KONFIG GOOGLE SHEET ===
SHEET_ID = "1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_EDIT_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
SHEET_CSV_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def load_data():
    try:
        url_no_cache = f"{SHEET_CSV_LINK}&t={int(time.time())}"
        response = requests.get(url_no_cache, timeout=5)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        else: return None
    except Exception: return None

def preprocess_sensor_data(df):
    if df is None or df.empty: return None, None, None, None
    df.columns = [c.strip() for c in df.columns]
    rename_map_candidates = {
        'Waktu': ['Waktu', 'Timestamp', 'Time'],
        'Tavg: Temperatur rata-rata (°C)': ['Suhu Udara', 'Suhu', 'Temperatur', 'Suhu (°C)'],
        'RH_avg: Kelembapan rata-rata (%)': ['Kelembapan Udara', 'Kelembapan', 'RH (%)'],
        'RR: Curah hujan (mm)': ['Curah Hujan/Jam', 'Curah Hujan', 'RR', 'Curah Hujan (mm)'],
        'ff_avg: Kecepatan angin rata-rata (m/s)': ['Kecepatan Angin (ms)', 'Kecepatan Angin', 'Angin (m/s)', 'ff_avg'],
        'Kelembaban Permukaan Tanah': ['Kelembapan Tanah', 'Kelembaban Tanah', 'Soil Moisture']
    }

    actual_rename = {}
    for target_name, candidates in rename_map_candidates.items():
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand; break
        if found is not None: actual_rename[found] = target_name

    df = df.rename(columns=actual_rename)
    fitur = [
        'Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)', 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
    ]

    missing = [c for c in fitur + ['Waktu'] if c not in df.columns]
    if missing: return "error", missing, None, None

    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = (clean_df[col].astype(str).str.replace(',', '.', regex=False).astype(float).fillna(0))
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaled_all = scaler.transform(clean_df)
    df["Prediksi Kebakaran"] = [convert_to_label(p) for p in model.predict(scaled_all)]
    return df, clean_df, scaled_all, fitur

# =========================================================================
# === SIDEBAR NAVIGATION ===
# =========================================================================
with st.sidebar:
    try:
        st.image("logo upi yptk.png", width=100)
    except: pass
    
    st.markdown("### Navigasi Sistem")
    
    if st.button("🌍 Dashboard IoT Utama", use_container_width=True, type="primary" if st.session_state.page == "Dashboard Utama" else "secondary"):
        st.session_state.page = "Dashboard Utama"
        st.rerun()
        
    if st.button("📸 Multimodal (YOLO + IoT)", use_container_width=True, type="primary" if st.session_state.page == "Multimodal" else "secondary"):
        st.session_state.page = "Multimodal"
        st.rerun()
        
    st.markdown("---")
    st.markdown("<div style='font-size:12px; color:gray;'><b>Dikembangkan oleh:</b><br>Mahasiswa Doctoral TI<br>Universitas Putra Indonesia YPTK Padang</div>", unsafe_allow_html=True)

# =========================================================================
# === HEADER GLOBAL ===
# =========================================================================
col1, col2 = st.columns([1, 9])
with col1:
    try: st.image("logo.png", width=170)
    except: pass
with col2:
    st.markdown("""
        <div style='margin-left: 20px;'>
            <h2 style='margin-bottom: 0px;'>Smart Fire Prediction Command Center</h2>
            <p style='font-size: 16px; line-height: 1.5; margin-top: 8px;'>
                Sistem Peringatan Dini Kebakaran Lahan Terpadu. Menggabungkan <b>Hybrid Stacking Ensemble Learning (HSEL)</b> untuk analisis cuaca/IoT, 
                dan <b>Computer Vision (YOLO)</b> untuk deteksi visual secara real-time.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# =========================================================================
# === HALAMAN 1: DASHBOARD UTAMA (IOT) ===
# =========================================================================
@st.fragment(run_every=7)
def indikator_kiri_realtime():
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    
    if res[0] is None:
        st.warning("Data belum tersedia atau gagal dimuat.")
        return
    if isinstance(res[0], str) and res[0] == "error":
        st.error("Kolom wajib tidak ditemukan: " + ", ".join(res[1]))
        return
        
    df, clean_df, scaled_all, fitur = res
    last_row = df.iloc[-1]
    last_num = clean_df.iloc[-1]
    waktu = pd.to_datetime(last_row['Waktu'], errors='coerce')
    if pd.isna(waktu):
        try: waktu = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
        except: waktu = None

    if isinstance(waktu, pd.Timestamp):
        hari = convert_day_to_indonesian(waktu.strftime('%A'))
        tanggal = waktu.strftime(f'%d %B %Y')
    else:
        hari, tanggal = "-", str(last_row['Waktu'])

    risk_label = last_row["Prediksi Kebakaran"]
    font, bg = risk_styles.get(risk_label, ("black", "white"))

    st.markdown(
        f"<p style='background-color:{bg}; color:{font}; padding:15px; border-radius:8px; font-weight:bold; text-align:center; font-size:18px;'>"
        f"Prediksi Risiko Saat Ini:<br>"
        f"<span style='text-decoration: underline; font-size: 26px;'>{risk_label}</span></p>",
        unsafe_allow_html=True
    )

    sensor_df = pd.DataFrame({
        "Variabel": ["Suhu (°C)", "Kelembapan (%)", "Curah Hujan (mm)", "Angin (m/s)", "Kel. Tanah (%)"],
        "Value": [f"{float(last_num[col]):.1f}" for col in fitur]
    })

    st.markdown("<h5 style='text-align: center; margin-top:20px;'>Data Sensor Realtime</h5>", unsafe_allow_html=True)
    sensor_html = "<table style='width: 100%; border-collapse: collapse; font-size:14px;'>"
    sensor_html += "<thead><tr><th style='padding:8px;'>Parameter</th><th style='padding:8px;'>Nilai</th></tr></thead><tbody>"
    for i in range(len(sensor_df)):
        sensor_html += f"<tr><td style='padding:8px;'>{sensor_df.iloc[i, 0]}</td><td style='padding:8px;'>{sensor_df.iloc[i, 1]}</td></tr>"
    sensor_html += "</tbody></table>"
    st.markdown(sensor_html, unsafe_allow_html=True)
    
    st.markdown(f"<div style='text-align:center; font-size:12px; margin-top:10px; color:gray;'>Pembaruan Terakhir: {hari}, {tanggal}</div>", unsafe_allow_html=True)

@st.fragment(run_every=7)
def peta_realtime_fragment():
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    if res[0] is not None and not isinstance(res[0], str):
        df, clean_df, scaled_all, fitur = res
        risk_label = df.iloc[-1]["Prediksi Kebakaran"]
        marker_color = {"Low / Rendah": "blue", "Moderate / Sedang": "green", "High / Tinggi": "orange", "Very High / Sangat Tinggi": "red"}.get(risk_label, "gray")

        m = folium.Map(location=[0.5333, 101.4500], zoom_start=9.5, control_scale=True, tiles='OpenStreetMap')
        Fullscreen(position='topright').add_to(m)

        try:
            riau_geojson_data = load_riau_geojson()
            pku_geo = {"type": "FeatureCollection", "features": []}
            if riau_geojson_data:
                for feature in riau_geojson_data['features']:
                    if 'pekanbaru' in feature['properties'].get('nama', '').lower():
                        pku_geo["features"].append(feature)
                        break
            if pku_geo["features"]:
                folium.GeoJson(pku_geo, style_function=lambda f, c=marker_color: {'fillColor': c, 'color': c, 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
        except: pass

        folium.Marker(location=[0.5333, 101.4500], icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m)
        folium_static(m, width=450, height=350)

def page_dashboard_utama():
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; background-color:#e8f4f8; padding:15px; border-radius:10px; border-left: 5px solid #1f77b4; margin-bottom:20px;">
            <div>
                <h4 style="margin:0; color:#1f77b4;">Pemantauan Lingkungan IoT & Prediksi HSEL</h4>
                <p style="margin:0; font-size:14px; color:#555;">Dashboard ini menampilkan data real-time parameter cuaca dan prediksi risiko kebakaran lahan (HSEL).</p>
            </div>
            <button onclick="window.parent.document.querySelectorAll('.stButton button')[1].click()" style="background-color:#e67e22; color:white; border:none; padding:10px 15px; border-radius:5px; cursor:pointer; font-weight:bold;">
                📸 Buka Multimodal Vision
            </button>
        </div>
        """, unsafe_allow_html=True
    )

    col_kiri, col_tengah, col_kanan = st.columns([1.2, 1.5, 1.2])
    with col_kiri:
        st.markdown("<div class='section-title'>Indikator Prediksi</div>", unsafe_allow_html=True)
        indikator_kiri_realtime()
    with col_tengah:
        st.markdown("<div class='section-title'>Peta Sebaran Risiko (Pekanbaru)</div>", unsafe_allow_html=True)
        peta_realtime_fragment()
    with col_kanan:
        st.markdown("<div class='section-title'>Pemantauan Lahan</div>", unsafe_allow_html=True)
        try: st.image(Image.open("forestiot4.jpg").resize((480, 360)), use_container_width=True)
        except: st.info("Gambar referensi lahan tidak ditemukan.")

    # Tampilkan Chart Historis
    st.markdown("<div class='section-title' style='margin-top: 30px;'>Visualisasi Historis Sensor IoT</div>", unsafe_allow_html=True)
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    if res[0] is not None and not isinstance(res[0], str):
        df, clean_df, scaled_all, fitur = res
        df_chart = clean_df.copy()
        df_chart['Waktu_DT'] = pd.to_datetime(df['Waktu'].astype(str).str.replace(' - ', ' '), errors='coerce')
        df_chart = df_chart.dropna(subset=['Waktu_DT']).set_index('Waktu_DT')
        if not df_chart.empty:
            df_daily = df_chart[fitur].resample('D').mean().dropna().tail(15).reset_index()
            df_melted = df_daily.melt(id_vars=['Waktu_DT'], var_name='Parameter Sensor', value_name='Nilai')
            chart = alt.Chart(df_melted).mark_line(strokeWidth=3, point=True).encode(
                x=alt.X('Waktu_DT:T', title='Tanggal'),
                y=alt.Y('Nilai:Q', title='Nilai'),
                color='Parameter Sensor:N',
                tooltip=['Waktu_DT:T', 'Parameter Sensor:N', alt.Tooltip('Nilai:Q', format='.1f')]
            ).properties(height=350).interactive()
            st.altair_chart(chart, use_container_width=True)

# =========================================================================
# === HALAMAN 2: DASHBOARD MULTIMODAL (YOLO + IoT) ===
# =========================================================================
@st.fragment(run_every=7)
def sensor_mini_fragment_multimodal():
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    
    st.markdown("<div style='background-color:#fff; padding:15px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
    st.markdown("<h4 style='border-bottom:2px solid #eee; padding-bottom:10px;'>📊 Konteks Lingkungan (IoT)</h4>", unsafe_allow_html=True)
    
    if res[0] is not None and not isinstance(res[0], str):
        df, clean_df, scaled_all, fitur = res
        last_row = df.iloc[-1]
        hsel_risk = last_row["Prediksi Kebakaran"]
        
        font, bg = risk_styles.get(hsel_risk, ("black", "white"))
        
        # Grid Info Cuaca
        c1, c2, c3 = st.columns(3)
        c1.metric("Suhu", f"{float(clean_df.iloc[-1][fitur[0]]):.1f} °C")
        c2.metric("Kelembapan", f"{float(clean_df.iloc[-1][fitur[1]]):.1f} %")
        c3.metric("Angin", f"{float(clean_df.iloc[-1][fitur[3]]):.1f} m/s")
        
        st.markdown(f"<div style='text-align:center; padding:10px; background-color:{bg}; color:{font}; border-radius:5px; font-weight:bold; margin-top:10px;'>Status HSEL: {hsel_risk}</div>", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
        st.markdown("#### 🚨 Keputusan Akhir Multimodal")
        
        # KEPUTUSAN GABUNGAN (Baca dari session_state YOLO)
        fire_detected = st.session_state.yolo_fire_detected
        
        if fire_detected is None:
            st.info("Silakan ambil gambar atau unggah citra di panel sebelah kiri untuk melihat hasil keputusan Multimodal.")
        else:
            if fire_detected and ("High" in hsel_risk or "Tinggi" in hsel_risk):
                st.markdown("""
                <div style='background-color: #8b0000; color: white; padding: 15px; border-radius: 8px; border-left: 8px solid red; animation: blinker 2s linear infinite;'>
                    <h3 style='color: white; margin-top:0;'>CRITICAL ALARM! 🔥</h3>
                    <b>Api Terdeteksi (YOLO) + Cuaca Sangat Mendukung Api Menyebar (HSEL)</b><br>
                    Segera kirimkan armada pemadam kebakaran dan lakukan evakuasi darurat! Kondisi sangat kritis.
                </div>
                """, unsafe_allow_html=True)
            elif fire_detected and ("Low" in hsel_risk or "Moderate" in hsel_risk or "Rendah" in hsel_risk or "Sedang" in hsel_risk):
                st.markdown("""
                <div style='background-color: #e67e22; color: white; padding: 15px; border-radius: 8px; border-left: 8px solid #d35400;'>
                    <h3 style='color: white; margin-top:0;'>WASPADA! ⚠️</h3>
                    <b>Api Terdeteksi (YOLO) + Cuaca Stabil (HSEL)</b><br>
                    Terdapat titik api aktif. Namun, kondisi alam cukup lembab/stabil sehingga api mungkin tidak cepat membesar. Padamkan segera sebelum merembet!
                </div>
                """, unsafe_allow_html=True)
            elif not fire_detected and ("High" in hsel_risk or "Tinggi" in hsel_risk):
                st.markdown("""
                <div style='background-color: #f1c40f; color: black; padding: 15px; border-radius: 8px; border-left: 8px solid #f39c12;'>
                    <h3 style='color: black; margin-top:0;'>SIAGA PATROLI 🔍</h3>
                    <b>Aman Visual (YOLO) + Cuaca Kritis (HSEL)</b><br>
                    Belum ada titik api secara visual, namun kondisi lahan sangat kering, panas, dan rawan terbakar tiba-tiba. Tingkatkan frekuensi patroli!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #27ae60; color: white; padding: 15px; border-radius: 8px; border-left: 8px solid #2ecc71;'>
                    <h3 style='color: white; margin-top:0;'>AMAN TERKENDALI ✅</h3>
                    <b>Aman Visual (YOLO) + Cuaca Stabil (HSEL)</b><br>
                    Tidak terdeteksi api dan lingkungan terpantau stabil. Lanjutkan pemantauan rutin.
                </div>
                """, unsafe_allow_html=True)

        # Mini Map
        st.markdown("<h5 style='margin-top:20px;'>Peta Pantauan Lokal</h5>", unsafe_allow_html=True)
        m_mini = folium.Map(location=[0.5333, 101.4500], zoom_start=10, tiles='CartoDB positron')
        folium.Marker(location=[0.5333, 101.4500], popup="Lokasi Sensor").add_to(m_mini)
        folium_static(m_mini, width=400, height=200)
    else:
        st.warning("Data IoT Terputus.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_multimodal():
    st.markdown("<div class='section-title' style='background-color:#2c3e50;'>📸 Command Center Multimodal (YOLO11 + HSEL IoT)</div>", unsafe_allow_html=True)
    st.write("Modul ini mengintegrasikan **Kecerdasan Visual (Deteksi Api)** dengan **Kecerdasan Lingkungan (IoT Cuaca)** untuk menghasilkan keputusan yang komprehensif.")

    if yolo_model is None:
        st.error("🚨 Model YOLO (`best.pt`) tidak ditemukan. Pastikan Anda telah mengunggah file `best.pt` ke direktori aplikasi!")
        return

    col_vis, col_sensor = st.columns([1.5, 1.1])
    
    with col_vis:
        st.markdown("<div style='background-color:#fff; padding:15px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1); min-height:650px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='border-bottom:2px solid #eee; padding-bottom:10px;'>👁️ Mata AI (YOLO Vision)</h4>", unsafe_allow_html=True)
        
        # Tabs untuk Kamera vs Upload
        tab_cam, tab_up = st.tabs(["🎥 Tangkapan Langsung (Kamera/USB)", "📁 Unggah File Citra"])
        
        img_to_process = None
        
        with tab_cam:
            st.info("Gunakan kamera laptop, webcam, atau kamera USB yang terhubung ke perangkat.")
            camera_image = st.camera_input("Ambil Citra Pantauan Lahan")
            if camera_image:
                img_to_process = Image.open(camera_image)
                
        with tab_up:
            uploaded_image = st.file_uploader("Atau unggah citra dari Drone / Satelit / CCTV lokal", type=['jpg','png','jpeg'])
            if uploaded_image:
                img_to_process = Image.open(uploaded_image)

        if img_to_process is not None:
            st.markdown("##### Proses Inferensi Berlangsung...")
            with st.spinner("Mendeteksi anomali api..."):
                # YOLO Inference
                results = yolo_model(img_to_process)
                res_plotted = results[0].plot() # Bounding box image (BGR)
                
                # Cek jumlah deteksi
                detections = results[0].boxes
                st.session_state.yolo_fire_detected = len(detections) > 0
                
                # Konversi BGR ke RGB
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, caption="Hasil Deteksi YOLO11", use_container_width=True)
                
                if st.session_state.yolo_fire_detected:
                    st.error(f"🔥 Visual Mendeteksi {len(detections)} Titik Api!")
                else:
                    st.success("✅ Tidak terdeteksi api secara visual.")
        else:
            st.session_state.yolo_fire_detected = None
            st.write("\n\n*Menunggu input visual dari kamera atau file...*")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sensor:
        # Panggil fragment yang auto-refresh
        sensor_mini_fragment_multimodal()


# =========================================================================
# === ROUTING RENDERER ===
# =========================================================================
if st.session_state.page == "Dashboard Utama":
    page_dashboard_utama()
elif st.session_state.page == "Multimodal":
    page_multimodal()

# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='margin-top: 20px; background-color: #2c3e50; padding: 10px 20px; border-radius: 10px; text-align: center; color: white;'>
    <p style='margin: 0; font-size: 24px; font-weight: bold; line-height: 1.2;'>Smart Fire Prediction HSEL & Multimodal Vision</p>
    <p style='margin: 0; font-size: 13px; line-height: 1.2; color:#bdc3c7;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026</p>
</div>
""", unsafe_allow_html=True)
