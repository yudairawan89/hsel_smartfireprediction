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
st.set_page_config(page_title="Smart Fire Command Center", page_icon="🔥", layout="wide")

# === ROUTING / MANAJEMEN HALAMAN NEW TAB ===
query_params = st.query_params
current_page = query_params.get("page", "main")

# Inisialisasi state untuk Multimodal
if "yolo_fire_detected" not in st.session_state:
    st.session_state.yolo_fire_detected = None

# === STYLE KUSTOM ===
st.markdown("""
    <style>
    .main {background-color: #f4f7f6;}
    table {width: 100%; border-collapse: collapse;}
    th, td {border: 1px solid #ddd; padding: 8px;}
    th {background-color: #e0e0e0; text-align: center;}
    td {text-align: center;}
    .section-title {
        background-color: #1f77b4;
        color: white;
        padding: 10px 15px;
        border-radius: 6px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .scrollable-table { overflow-x: auto; }
    
    /* Custom Styling Card */
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Frame khusus untuk iframe Folium */
    iframe[title="folium_static"] {
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
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
    "Low / Rendah": ("white", "#228B22"),
    "Moderate / Sedang": ("black", "#FFD700"),
    "High / Tinggi": ("white", "#FF6347"),
    "Very High / Sangat Tinggi": ("white", "#8B0000")
}

def get_image_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

def get_multimodal_decision(visual_label, iot_label):
    if visual_label == 1 and "Very High" in iot_label:
        return "Kebakaran Telah Terjadi", "Indikasi sangat kuat bahwa kebakaran telah terjadi berdasarkan konfirmasi visual dan sensor.", "#B22222", "🔥"
    elif visual_label == 1 and "High" in iot_label:
        return "Kebakaran Sangat Mungkin", "Gambaran visual api dan kondisi lingkungan berisiko tinggi memperkuat dugaan kebakaran aktif.", "#DC143C", "🚨"
    elif visual_label == 1 and "Moderate" in iot_label:
        return "Kemungkinan Kebakaran", "Visual mendeteksi api, namun kondisi sensor menunjukkan tingkat risiko sedang.", "#FF8C00", "⚠️"
    elif visual_label == 1 and "Low" in iot_label:
        return "Terdeteksi Api Isolated", "Visual menunjukkan api meskipun kondisi lingkungan kurang mendukung penyebaran. Kemungkinan aktivitas manusia (terkendali).", "#FFA500", "🟠"
    elif visual_label == 0 and "Very High" in iot_label:
        return "Risiko Kebakaran Sangat Tinggi", "Belum ada deteksi visual api, namun lingkungan sangat rentan kebakaran. Waspada dini diperlukan.", "#8B0000", "🌡️"
    elif visual_label == 0 and "High" in iot_label:
        return "Potensi Kebakaran Tinggi", "Belum ada api terdeteksi, tetapi kondisi sekitar menunjukkan risiko tinggi kebakaran.", "#FF6347", "🔥"
    elif visual_label == 0 and "Moderate" in iot_label:
        return "Kondisi Rentan Kebakaran", "Lingkungan menunjukkan risiko sedang terhadap kebakaran. Pemantauan disarankan.", "#FFD700", "⚠️"
    elif visual_label == 0 and "Low" in iot_label:
        return "Tidak Terindikasi Kebakaran", "Tidak ada api terdeteksi dan kondisi lingkungan tergolong aman.", "#228B22", "✅"
    else:
        return "Status Tidak Diketahui", "Data tidak mencukupi untuk menarik kesimpulan.", "#808080", "❓"

# === LOAD MODEL, SCALER, SASTRAWI, YOLO ===
@st.cache_resource
def load_model(): return joblib.load("HSEL_IoT_Model.joblib")

@st.cache_resource
def load_scaler(): return joblib.load("scaler.joblib")

@st.cache_resource
def load_sastrawi():
    stop_factory = StopWordRemoverFactory()
    return stop_factory.create_stop_word_remover(), StemmerFactory().create_stemmer()

@st.cache_resource
def load_text_models():
    return joblib.load("tfidf_vectorizer.joblib"), joblib.load("stacking_text_model.joblib")

@st.cache_resource
def load_yolo_model():
    try: return YOLO("best.pt")
    except Exception: return None

model = load_model()
scaler = load_scaler()
stopword_remover, stemmer = load_sastrawi()
yolo_model = load_yolo_model()
try: vectorizer, model_text = load_text_models()
except: vectorizer, model_text = None, None

@st.cache_data
def load_riau_geojson():
    try:
        with open("Provinsi Riau-KAB_KOTA.geojson", "r") as f: return json.load(f)
    except: return None

SHEET_ID = "1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_CSV_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
SHEET_EDIT_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"

def load_data():
    try:
        response = requests.get(f"{SHEET_CSV_LINK}&t={int(time.time())}", timeout=5)
        if response.status_code == 200: return pd.read_csv(StringIO(response.text))
    except: return None

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
        for cand in candidates:
            if cand in df.columns:
                actual_rename[cand] = target_name; break

    df = df.rename(columns=actual_rename)
    fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)', 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Permukaan Tanah']
    missing = [c for c in fitur + ['Waktu'] if c not in df.columns]
    if missing: return "error", missing, None, None

    clean_df = df[fitur].copy()
    for col in fitur: clean_df[col] = (clean_df[col].astype(str).str.replace(',', '.', regex=False).astype(float).fillna(0))
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    scaled_all = scaler.transform(clean_df)
    df["Prediksi Kebakaran"] = [convert_to_label(p) for p in model.predict(scaled_all)]
    return df, clean_df, scaled_all, fitur


# =========================================================================
# === RENDERING HALAMAN ===
# =========================================================================

# -------------------------------------------------------------------------
# HALAMAN 2 (NEW TAB): DASHBOARD MULTIMODAL
# -------------------------------------------------------------------------
if current_page == "multimodal":
    
    # Elegan Header
    logo_b64 = get_image_base64("logo upi yptk.png")
    img_tag = f'<img src="data:image/png;base64,{logo_b64}" width="65" style="border-radius:8px; background:white; padding:5px;">' if logo_b64 else '🔥'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 25px 30px; border-radius: 12px; color: white; display: flex; align-items: center; margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
        <div style="margin-right: 20px;">
            {img_tag}
        </div>
        <div>
            <h1 style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px; color: #ffffff;">COMMAND CENTER MULTIMODAL</h1>
            <p style="margin: 5px 0 0 0; font-size: 15px; color: #b2bec3;">Sistem Peringatan Dini Cerdas: Integrasi Kecerdasan Visual YOLO11 dan Sensor Lingkungan HSEL Terpadu</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if yolo_model is None:
        st.error("🚨 Model YOLO (`best.pt`) tidak ditemukan. Pastikan Anda telah mengunggah file `best.pt` ke direktori aplikasi!")
    else:
        col_vis, col_sensor = st.columns([1.5, 1.1], gap="large")
        
        # === KOLOM VISUAL (KIRI) ===
        with col_vis:
            st.markdown("<div class='stCard'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#2c3e50; border-bottom:2px solid #e0e0e0; padding-bottom:10px; margin-top:0;'>👁️ AI Visual (YOLO11)</h4>", unsafe_allow_html=True)
            
            # Pengganti Tabs: Menggunakan Radio Button agar kamera mati saat pindah menu
            input_method = st.radio("Pilih Sumber Pengamatan:", ["📁 Unggah File Citra", "🎥 Kamera Langsung / USB"], horizontal=True)
            img_to_process = None
            
            st.markdown("<div style='min-height: 400px; display: flex; justify-content: center; align-items: center; border: 1px dashed #ccc; border-radius: 8px; background-color: #fafafa; padding: 10px;'>", unsafe_allow_html=True)
            
            if "Kamera" in input_method:
                st.info("💡 Pastikan memberikan izin akses kamera pada browser Anda.")
                camera_image = st.camera_input("Ambil Citra Lahan")
                if camera_image: img_to_process = Image.open(camera_image)
            else:
                uploaded_image = st.file_uploader("Unggah citra dari Drone / CCTV / Satelit (JPG/PNG)", type=['jpg','png','jpeg'])
                if uploaded_image: img_to_process = Image.open(uploaded_image)

            if img_to_process is not None:
                with st.spinner("🔍 Sedang mengidentifikasi titik api..."):
                    results = yolo_model(img_to_process)
                    res_plotted = results[0].plot()
                    detections = results[0].boxes
                    st.session_state.yolo_fire_detected = len(detections) > 0
                    
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, caption="Hasil Analisis Visi Komputer YOLO11", use_container_width=True)
                    
                    if st.session_state.yolo_fire_detected:
                        st.error(f"🔥 Sistem mendeteksi keberadaan {len(detections)} titik api aktif!")
                    else:
                        st.success("✅ Tidak terdeteksi adanya anomali api pada citra ini.")
            else:
                st.session_state.yolo_fire_detected = None
                
            st.markdown("</div></div>", unsafe_allow_html=True)

        # === KOLOM SENSOR & KEPUTUSAN (KANAN) ===
        with col_sensor:
            @st.fragment(run_every=7)
            def sensor_and_decision_fragment():
                df_raw = load_data()
                res = preprocess_sensor_data(df_raw)
                
                st.markdown("<div class='stCard' style='margin-bottom:0;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#2c3e50; border-bottom:2px solid #e0e0e0; padding-bottom:10px; margin-top:0;'>📡 Konteks Lingkungan (IoT)</h4>", unsafe_allow_html=True)
                
                if res[0] is not None and not isinstance(res[0], str):
                    df, clean_df, scaled_all, fitur = res
                    last_row = df.iloc[-1]
                    last_num = clean_df.iloc[-1]
                    hsel_risk = last_row["Prediksi Kebakaran"]
                    
                    # Layout Grid untuk 5 Parameter
                    st.markdown("<p style='font-size:13px; color:gray; margin-bottom:5px;'>Pembacaan Node Sensor Real-Time:</p>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🌡️ Suhu", f"{float(last_num[fitur[0]]):.1f} °C")
                    m2.metric("💧 Kelembapan", f"{float(last_num[fitur[1]]):.1f} %")
                    m3.metric("🌧️ Curah Hujan", f"{float(last_num[fitur[2]]):.1f} mm")
                    
                    m4, m5, _ = st.columns([1,1,1])
                    m4.metric("💨 Kec. Angin", f"{float(last_num[fitur[3]]):.1f} m/s")
                    m5.metric("🌱 Kel. Tanah", f"{float(last_num[fitur[4]]):.1f} %")
                    
                    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)
                    
                    # LOGIKA MULTIMODAL KEPUTUSAN
                    st.markdown("<h4 style='color:#2c3e50; margin-bottom:0;'>🧠 Output Keputusan Terpadu</h4>", unsafe_allow_html=True)
                    fire_detected = st.session_state.get("yolo_fire_detected", None)
                    
                    if fire_detected is None:
                        st.info("ℹ️ Menunggu konfirmasi visual (Silakan unggah atau ambil gambar di panel kiri).")
                    else:
                        visual_val = 1 if fire_detected else 0
                        title, desc, color, icon = get_multimodal_decision(visual_val, hsel_risk)
                        
                        font_col = "white" if color not in ["#FFD700", "#FF8C00", "#FFA500"] else "black"
                        
                        st.markdown(f"""
                        <div style='background-color: {color}; color: {font_col}; padding: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.15); margin-top: 10px;'>
                            <h3 style='color: {font_col}; margin-top: 0; font-size:20px;'>{icon} {title}</h3>
                            <p style='font-size: 14px; margin-bottom: 0; line-height:1.4;'>{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # MINI MAP (Dengan Data GeoJSON Pekanbaru)
                    st.markdown("<h5 style='margin-top:25px; color:#2c3e50;'>🗺️ Peta Konteks Lokal</h5>", unsafe_allow_html=True)
                    marker_color = {"Low / Rendah": "blue", "Moderate / Sedang": "green", "High / Tinggi": "orange", "Very High / Sangat Tinggi": "red"}.get(hsel_risk, "gray")
                    
                    m_mini = folium.Map(location=[0.5333, 101.4500], zoom_start=9.5, tiles='CartoDB positron', control_scale=True)
                    
                    try:
                        riau_geojson_data = load_riau_geojson()
                        pku_geo = {"type": "FeatureCollection", "features": []}
                        if riau_geojson_data:
                            for feature in riau_geojson_data['features']:
                                if 'pekanbaru' in feature['properties'].get('nama', '').lower() or 'pekanbaru' in feature['properties'].get('kab_kota', '').lower():
                                    pku_geo["features"].append(feature)
                                    break
                        if pku_geo["features"]:
                            folium.GeoJson(pku_geo, style_function=lambda f, c=marker_color: {'fillColor': c, 'color': c, 'weight': 2, 'fillOpacity': 0.3}).add_to(m_mini)
                    except Exception: pass
                    
                    popup_html = f"<b>Pekanbaru</b><br>HSEL: {hsel_risk}"
                    folium.Marker(location=[0.5333, 101.4500], popup=popup_html, icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m_mini)
                    folium_static(m_mini, width=420, height=220)
                    
                else:
                    st.warning("⚠️ Data IoT Terputus atau Tidak Tersedia.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            sensor_and_decision_fragment()
            
            # BAGIAN GAMBAR ALAT IOT DAN INFORMASI RISIKO
            st.markdown("<div class='stCard'>", unsafe_allow_html=True)
            
            col_alat, col_info = st.columns([1, 1])
            with col_alat:
                st.markdown("<b style='color:#333; font-size:14px;'>⚙️ Node Sensor IoT</b>", unsafe_allow_html=True)
                try:
                    # Ganti "alat_iot.png" dengan nama file gambar IoT Anda
                    st.image(Image.open("alat_iot.png"), use_container_width=True)
                except Exception:
                    try: 
                        st.image(Image.open("forestiot4.jpg"), use_container_width=True)
                    except:
                        st.info("Gambar alat tidak ditemukan.")
                        
            with col_info:
                st.markdown("<b style='color:#333; font-size:14px;'>🗂️ Referensi Cepat</b>", unsafe_allow_html=True)
                with st.expander("ℹ️ Panduan Risiko & Tindakan", expanded=False):
                    st.markdown("""
                    <div style="font-size:11px; line-height:1.4;">
                    <b style="color:#228B22;">Low (Rendah):</b> Resiko rendah. Api mudah dikendalikan.<br><br>
                    <b style="color:#FFD700;">Moderate (Sedang):</b> Resiko sedang. Pantauan ditingkatkan.<br><br>
                    <b style="color:#FF6347;">High (Tinggi):</b> Resiko tinggi. Api sulit dikendalikan. Siapkan tim.<br><br>
                    <b style="color:#8B0000;">Very High (Sangat Tinggi):</b> Darurat! Api sangat sulit dikendalikan.
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# HALAMAN 1 (DEFAULT): DASHBOARD UTAMA (TIDAK BERUBAH DARI PERMINTAAN SEBELUMNYA)
# -------------------------------------------------------------------------
else:
    # === HEADER ===
    col1, col2 = st.columns([1, 9])
    with col1:
        try:
            st.image("logo.png", width=170)
        except:
            pass
    with col2:
        st.markdown("""
            <div style='margin-left: 20px;'>
                <h2 style='margin-bottom: 0px;'>Smart Fire Prediction HSEL Model</h2>
                <p style='font-size: 16px; line-height: 1.5; margin-top: 8px;'>
                    Sistem ini menggunakan Hybrid Stacking Ensemble Learning (HSEL) untuk memprediksi risiko kebakaran hutan secara real-time dengan tingkat akurasi tinggi.
                    Model prediksi dikembangkan dari kombinasi berbagai algoritma pembelajaran mesin yang dioptimalkan menggunakan optimasi hyperparameter untuk meningkatkan performa klasifikasi.
                    Data pengujian secara real-time berasal dari perangkat IoT yang mengukur parameter lingkungan seperti suhu, kelembapan, curah hujan, kecepatan angin, dan kelembapan tanah.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_btn = st.columns([10, 1])[1]
        with col_btn:
            st.markdown(
                f"""
                <a href='{SHEET_EDIT_LINK}' target='_blank'>
                <button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>Data Cloud</button>
                </a>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)


    # === BAGIAN REALTIME FRAGMENT (KOLOM KIRI YANG REFRESH 7 DETIK) ==========
    @st.fragment(run_every=7)
    def indikator_kiri_realtime():
        df_raw = load_data()
        res = preprocess_sensor_data(df_raw)
        
        if res[0] is None:
            st.warning("Data belum tersedia atau gagal dimuat dari Google Sheets.")
            return
        if isinstance(res[0], str) and res[0] == "error":
            st.error("Kolom wajib tidak ditemukan di Sheets: " + ", ".join(res[1]))
            return
            
        df, clean_df, scaled_all, fitur = res
        
        last_row = df.iloc[-1]
        last_num = clean_df.iloc[-1]
        waktu = pd.to_datetime(last_row['Waktu'], errors='coerce')
        if pd.isna(waktu):
            try:
                waktu = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
            except Exception:
                waktu = None

        if isinstance(waktu, pd.Timestamp):
            hari = convert_day_to_indonesian(waktu.strftime('%A'))
            bulan = convert_month_to_indonesian(waktu.strftime('%B'))
            tanggal = waktu.strftime(f'%d {bulan} %Y')
        else:
            hari, tanggal = "-", str(last_row['Waktu'])

        risk_label = last_row["Prediksi Kebakaran"]
        font, bg = risk_styles.get(risk_label, ("black", "white"))

        sensor_df = pd.DataFrame({
            "Variabel": fitur,
            "Value": [f"{float(last_num[col]):.1f}" for col in fitur]
        })

        st.markdown("<div class='stCard' style='padding: 15px;'>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align: center; margin-top:0;'>Data Sensor Realtime</h5>", unsafe_allow_html=True)
        sensor_html = "<table style='width: 100%; border-collapse: collapse; font-size:14px;'>"
        sensor_html += "<thead><tr><th>Variabel</th><th>Value</th></tr></thead><tbody>"
        for i in range(len(sensor_df)):
            var = sensor_df.iloc[i, 0]
            val = sensor_df.iloc[i, 1]
            sensor_html += f"<tr><td style='padding:6px; text-align:left;'>{var}</td><td style='padding:6px;'>{val}</td></tr>"
        sensor_html += "</tbody></table>"
        st.markdown(sensor_html, unsafe_allow_html=True)

        st.markdown(
            f"<div style='background-color:{bg}; color:{font}; padding:15px; border-radius:8px; font-weight:bold; text-align:center; margin-top:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
            f"<span style='font-size:13px; font-weight:normal;'>{hari}, {tanggal}</span><br>"
            f"Tingkat Resiko Kebakaran:<br>"
            f"<span style='text-decoration: underline; font-size: 20px;'>{risk_label}</span></div>",
            unsafe_allow_html=True
        )

        with st.expander("📊 Analisis Keputusan Model (XAI)"):
            st.markdown("<span style='font-size:13px; color:gray;'>Grafik di bawah menunjukkan seberapa besar setiap parameter sensor berkontribusi terhadap prediksi.</span>", unsafe_allow_html=True)

            try:
                data_realtime_scaled = pd.DataFrame(scaled_all[-1:], columns=fitur)
                background_data = pd.DataFrame(shap.sample(scaled_all, 50), columns=fitur)

                explainer = shap.Explainer(model.predict, background_data)
                shap_values = explainer(data_realtime_scaled)

                data_realtime_raw = clean_df.iloc[-1:].values
                shap_values.data = data_realtime_raw

                plt.rcParams.update({'font.size': 14})
                fig, ax = plt.subplots(figsize=(10, 6))

                shap.plots.waterfall(shap_values[0], show=False)

                total_abs_shap = sum(abs(v) for v in shap_values[0].values)

                for text in ax.texts:
                    text_str = text.get_text().strip()
                    clean_str = text_str.replace('−', '-')
                    try:
                        val = float(clean_str)
                        if total_abs_shap > 0:
                            pct = (abs(val) / total_abs_shap) * 100
                            text.set_text(f"{text_str} ({pct:.1f}%)")
                    except ValueError: pass

                st.pyplot(fig, bbox_inches='tight', dpi=300)
                plt.close(fig) 
                plt.clf()
                plt.rcParams.update({'font.size': 10})

                shap_vals_arr = shap_values[0].values
                kontribusi = []
                for nama_f, shap_v in zip(fitur, shap_vals_arr):
                    pct_f = (abs(float(shap_v)) / total_abs_shap) * 100 if total_abs_shap > 0 else 0.0
                    kontribusi.append({
                        "fitur": nama_f, "shap_val": float(shap_v), "pct": pct_f
                    })
                kontribusi = sorted(kontribusi, key=lambda x: x["pct"], reverse=True)

                st.markdown("<h5 style='margin-top: 20px;'>Detail Keputusan</h5>", unsafe_allow_html=True)
                icons = ["🔴", "🟠", "🟡", "🟢", "⚪"]

                for i, factor in enumerate(kontribusi):
                    icon = icons[i] if i < len(icons) else "⚪"
                    nama_fitur = str(factor['fitur']).lower()
                    persen = factor['pct']
                    arah = factor['shap_val']

                    st.markdown(f"<span style='font-size:14px;'>**{icon} {factor['fitur'].title()} ({persen:.1f}%)**</span>", unsafe_allow_html=True)
                    if persen < 5.0:
                        desc_xai = "Dorongan minor terhadap potensi risiko saat ini."
                    else:
                        if "tanah" in nama_fitur: desc_xai = "Meningkatkan Risiko (Kering)" if arah > 0 else "Meredam Risiko (Lembab)"
                        elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur: desc_xai = "Udara Kering" if arah > 0 else "Menjaga Kebasahan"
                        elif "angin" in nama_fitur or "ff" in nama_fitur: desc_xai = "Mempercepat Eskalasi" if arah > 0 else "Kondisi Stabil"
                        elif "suhu" in nama_fitur or "temperatur" in nama_fitur or "tavg" in nama_fitur: desc_xai = "Memicu Penguapan Panas" if arah > 0 else "Stabilitas Termal Terjaga"
                        elif "hujan" in nama_fitur or "rr" in nama_fitur: desc_xai = "Pendingin Alami Hilang" if arah > 0 else "Faktor Pendingin Aktif"
                        else: desc_xai = "Meningkatkan Potensi" if arah > 0 else "Menstabilkan Potensi"
                    st.markdown(f"<p style='font-size:12px; margin-top:-5px; padding-left:25px;'>{desc_xai}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Visualisasi XAI belum siap.")
        st.markdown("</div>", unsafe_allow_html=True)


    # === BAGIAN PETA REALTIME PEKANBARU FRAGMENT ===========
    @st.fragment(run_every=7)
    def peta_realtime_fragment():
        df_raw = load_data()
        res = preprocess_sensor_data(df_raw)
        
        if res[0] is not None and not isinstance(res[0], str):
            df, clean_df, scaled_all, fitur = res
            last_row = df.iloc[-1]
            last_num = clean_df.iloc[-1]
            risk_label = last_row["Prediksi Kebakaran"]
            
            waktu_valid = pd.to_datetime(last_row['Waktu'], errors='coerce')
            if pd.isna(waktu_valid):
                try: waktu_valid = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
                except Exception: pass
                
            if pd.notna(waktu_valid): tanggal_valid = waktu_valid.strftime('%d %B %Y, %H:%M WIB')
            else: tanggal_valid = str(last_row['Waktu'])
            
            pekanbaru_coords = [0.5333, 101.4500] 
            marker_color = {"Low / Rendah": "blue", "Moderate / Sedang": "green", "High / Tinggi": "orange", "Very High / Sangat Tinggi": "red"}.get(risk_label, "gray")

            try:
                riau_geojson_data = load_riau_geojson()
                pekanbaru_geojson = {"type": "FeatureCollection", "features": []}
                if riau_geojson_data:
                    for feature in riau_geojson_data['features']:
                        nama_wilayah = feature['properties'].get('nama', '').lower()
                        kab_kota = feature['properties'].get('kab_kota', '').lower()
                        if 'pekanbaru' in nama_wilayah or 'pekanbaru' in kab_kota:
                            pekanbaru_geojson["features"].append(feature)
                            break
            except Exception:
                pekanbaru_geojson = None

            popup_text = folium.Popup(f"""
                <div style='width: 230px; font-size: 13px; line-height: 1.5;'>
                <b>Wilayah:</b> Kota Pekanbaru<br>
                <b>Prediksi:</b> {risk_label}<br>
                <b>Suhu:</b> {float(last_num[fitur[0]]):.1f} °C<br>
                <b>Kelembapan:</b> {float(last_num[fitur[1]]):.1f} %<br>
                <b>Curah Hujan:</b> {float(last_num[fitur[2]]):.1f} mm<br>
                <b>Kecepatan Angin:</b> {float(last_num[fitur[3]]):.1f} m/s<br>
                <b>Kelembaban Tanah:</b> {float(last_num[fitur[4]]):.1f} %
                </div>
            """, max_width=250)

            m = folium.Map(location=pekanbaru_coords, zoom_start=9.5, control_scale=True, tiles='OpenStreetMap')

            formatter = "function(num) {return L.Util.formatNum(num, 5) + ' &deg;';};"
            MousePosition(
                position="bottomleft", separator=" | ", empty_string="Koordinat tidak tersedia",
                lng_first=True, num_digits=20, prefix="Posisi:", lat_formatter=formatter, lng_formatter=formatter,
            ).add_to(m)
            Fullscreen(position='topright').add_to(m)

            if pekanbaru_geojson and pekanbaru_geojson["features"]:
                folium.GeoJson(
                    pekanbaru_geojson,
                    style_function=lambda feature, color=marker_color: {'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.4},
                    tooltip=folium.GeoJsonTooltip(fields=['nama'], aliases=['Wilayah:'], style="font-weight: bold; font-size: 14px;")
                ).add_to(m)

            folium.Marker(location=pekanbaru_coords, popup=popup_text, icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m)

            logo_b64 = get_image_base64("logo.png")
            logo_upi_b64 = get_image_base64("logo upi yptk.png")
                
            logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="height: 45px; background: white; padding: 4px; border-radius: 4px;" alt="Logo">' if logo_b64 else ''
            logo_upi_tag = f'<img src="data:image/png;base64,{logo_upi_b64}" style="width: 50px; height: auto;" alt="Logo UPI YPTK">' if logo_upi_b64 else ''
            
            raw_map_html = m.get_root().render()

            custom_css_and_layout_start = f"""
            <body style="background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
                <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 100%; max-width: 1450px; height: 95vh; display: flex; flex-direction: column;">
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #1f77b4 0%, #175a8a 100%); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(31,119,180,0.2);">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            {logo_img_tag}
                            <div>
                                <h2 style="margin: 0; font-size: 20px; font-weight: 600; letter-spacing: 0.5px;">Dashboard Pemantauan Spasial</h2>
                                <p style="margin: 5px 0 0 0; font-size: 12px; color: #dceefb;">Integrasi Model Machine Learning dan GIS</p>
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 13px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 8px;">
                            <b>Wilayah:</b> Pekanbaru<br>
                            <span style="color: #e2f0ff;">{tanggal_valid}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 95px); overflow: hidden;">
                        <div style="flex-grow: 1; border: 2px solid #e2e8f0; border-radius: 10px; overflow: hidden; position: relative;">
            """
            
            custom_layout_end = f"""
                        </div> 
                        
                        <div style="width: 300px; display: flex; flex-direction: column; gap: 15px; flex-shrink: 0;">
                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9;">
                                <b style="font-size: 13px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 10px;">Status Prediksi Saat Ini</b>
                                <div style="font-size: 16px; font-weight: bold; color: {marker_color};">{risk_label}</div>
                            </div>

                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; flex-grow:1;">
                                <b style="font-size: 12px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 8px;">Legenda Risiko</b>
                                <div style="font-size: 12px; line-height: 2;">
                                    <div style="display: flex; align-items: center;"><i style="background: blue; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;"></i> Rendah</div>
                                    <div style="display: flex; align-items: center;"><i style="background: green; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;"></i> Sedang</div>
                                    <div style="display: flex; align-items: center;"><i style="background: orange; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;"></i> Tinggi</div>
                                    <div style="display: flex; align-items: center;"><i style="background: red; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;"></i> S. Tinggi</div>
                                </div>
                            </div>
                        </div>

                    </div> 
                </div> 
            </body>
            """
            
            framed_dashboard_html = raw_map_html.replace('<body>', custom_css_and_layout_start).replace('</body>', custom_layout_end)
            folium_static(m, width=450, height=350)

            b64_html = base64.b64encode(framed_dashboard_html.encode('utf-8')).decode('utf-8')
            
            custom_button_html = f"""
            <button onclick="openMap()" style="width: 100%; padding: 8px 16px; background-color: #ffffff; color: #333; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 14px; transition: 0.3s;" onmouseover="this.style.borderColor='#1f77b4'; this.style.color='#1f77b4'" onmouseout="this.style.borderColor='#ccc'; this.style.color='#333'">
                🌐 Buka Dashboard Peta Penuh (Pekanbaru)
            </button>
            <script>
            function openMap() {{
                fetch(`data:text/html;base64,{b64_html}`).then(res => res.blob()).then(blob => {{
                    window.open(URL.createObjectURL(blob), '_blank');
                }});
            }}
            </script>
            """
            components.html(custom_button_html, height=50)


    # === BAGIAN PETA REGIONAL FRAGMENT ===
    @st.fragment(run_every=7)
    def peta_regional_fragment():
        df_raw = load_data()
        res = preprocess_sensor_data(df_raw)
        
        if res[0] is not None and not isinstance(res[0], str):
            df, clean_df, scaled_all, fitur = res
            last_row = df.iloc[-1]
            last_num = clean_df.iloc[-1]
            risk_label_pku = last_row["Prediksi Kebakaran"]
            
            waktu_valid = pd.to_datetime(last_row['Waktu'], errors='coerce')
            if pd.isna(waktu_valid):
                try: waktu_valid = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
                except Exception: pass
                
            tanggal_valid = waktu_valid.strftime('%d %B %Y, %H:%M WIB') if pd.notna(waktu_valid) else str(last_row['Waktu'])
            marker_color_pku = {"Low / Rendah": "blue", "Moderate / Sedang": "green", "High / Tinggi": "orange", "Very High / Sangat Tinggi": "red"}.get(risk_label_pku, "gray")

            logo_b64 = get_image_base64("logo.png")
            logo_upi_b64 = get_image_base64("logo upi yptk.png")

            logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="height: 45px; background: white; padding: 4px; border-radius: 4px;" alt="Logo">' if logo_b64 else ''
            logo_upi_tag = f'<img src="data:image/png;base64,{logo_upi_b64}" style="width: 50px; height: auto;" alt="Logo UPI YPTK">' if logo_upi_b64 else ''
            
            regional_coords = [0.8500, 101.9000] 
            m_regional = folium.Map(location=regional_coords, zoom_start=7.5, control_scale=True, tiles='OpenStreetMap')
            Fullscreen(position='topright').add_to(m_regional)

            riau_geojson_data = load_riau_geojson()
            if riau_geojson_data:
                riau_geojson = copy.deepcopy(riau_geojson_data) 
                filtered_features = []
                
                for feature in riau_geojson['features']:
                    nama_wilayah = feature['properties'].get('nama', '').lower()
                    kab_kota = feature['properties'].get('kab_kota', '').lower()
                    
                    if 'pekanbaru' in nama_wilayah or 'pekanbaru' in kab_kota:
                        feature['properties']['warna_fill'] = marker_color_pku
                        feature['properties']['tooltip_info'] = f"Status: {risk_label_pku} (Real-time)"
                        filtered_features.append(feature)
                    elif 'siak' in nama_wilayah or 'siak' in kab_kota:
                        feature['properties']['warna_fill'] = "#9e9e9e"
                        feature['properties']['tooltip_info'] = "Menunggu Data IoT"
                        filtered_features.append(feature)
                    elif 'pelalawan' in nama_wilayah or 'pelalawan' in kab_kota:
                        feature['properties']['warna_fill'] = "#9e9e9e"
                        feature['properties']['tooltip_info'] = "Menunggu Data IoT"
                        filtered_features.append(feature)
                    elif 'bengkalis' in nama_wilayah or 'bengkalis' in kab_kota:
                        feature['properties']['warna_fill'] = "#9e9e9e"
                        feature['properties']['tooltip_info'] = "Menunggu Data IoT"
                        filtered_features.append(feature)

                riau_geojson['features'] = filtered_features

                folium.GeoJson(
                    riau_geojson,
                    style_function=lambda feature: {
                        'fillColor': feature['properties']['warna_fill'],
                        'color': '#333333', 
                        'weight': 2,
                        'fillOpacity': 0.7 if feature['properties']['warna_fill'] != "#9e9e9e" else 0.4,
                    },
                    tooltip=folium.GeoJsonTooltip(fields=['nama', 'tooltip_info'], aliases=['Kab/Kota:', 'Keterangan:'], style="font-weight: bold; font-size: 13px;")
                ).add_to(m_regional)
                
                popup_pku_html = f"<div style='width: 200px; font-size: 12px;'><b>Pekanbaru:</b> {risk_label_pku}</div>"
                folium.Marker(location=[0.5333, 101.4500], popup=folium.Popup(popup_pku_html), icon=folium.Icon(color=marker_color_pku, icon="info-sign")).add_to(m_regional)

                def get_dummy_popup(nama_daerah): return f"<div style='width: 200px; font-size: 12px;'><b>{nama_daerah}:</b> Menunggu Data</div>"
                folium.Marker(location=[0.7490, 102.0460], popup=folium.Popup(get_dummy_popup("Kab. Siak")), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)
                folium.Marker(location=[0.2662, 101.6917], popup=folium.Popup(get_dummy_popup("Kab. Pelalawan")), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)
                folium.Marker(location=[1.4789, 102.1444], popup=folium.Popup(get_dummy_popup("Kab. Bengkalis")), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)

            raw_map_html = m_regional.get_root().render()

            custom_css_and_layout_start = f"""
            <body style="background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
                <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 100%; max-width: 1450px; height: 95vh; display: flex; flex-direction: column;">
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(30,60,114,0.2);">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            {logo_img_tag}
                            <div>
                                <h2 style="margin: 0; font-size: 20px; font-weight: 600; letter-spacing: 0.5px;">Pemantauan Regional</h2>
                                <p style="margin: 5px 0 0 0; font-size: 12px; color: #d1e8ff;">Tahap Perluasan Integrasi Sensor IoT</p>
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 13px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 8px;">
                            <b>Domain:</b> Sebagian Wilayah Riau<br>
                            <span style="color: #e2f0ff;">{tanggal_valid}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 95px); overflow: hidden;">
                        <div style="flex-grow: 1; border: 2px solid #e2e8f0; border-radius: 10px; overflow: hidden; position: relative;">
            """
            
            custom_layout_end = "</div></div></div></body>"
            framed_dashboard_html = raw_map_html.replace('<body>', custom_css_and_layout_start).replace('</body>', custom_layout_end)

            b64_html = base64.b64encode(framed_dashboard_html.encode('utf-8')).decode('utf-8')
            
            custom_button_html = f"""
            <button onclick="openMapRegional()" style="width: 100%; padding: 8px 16px; background-color: #e67e22; color: #ffffff; border: none; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 14px; margin-top: 10px; font-weight: bold; transition: 0.3s;" onmouseover="this.style.backgroundColor='#d35400'" onmouseout="this.style.backgroundColor='#e67e22'">
                🗺️ Buka Dashboard Pemantauan Regional
            </button>
            <script>
            function openMapRegional() {{
                fetch(`data:text/html;base64,{b64_html}`).then(res => res.blob()).then(blob => {{
                    window.open(URL.createObjectURL(blob), '_blank');
                }});
            }}
            </script>
            """
            components.html(custom_button_html, height=60)

    # === BAGIAN UTAMA DASHBOARD =====================
    def main_dashboard():
        st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)
        
        df_raw = load_data()
        res = preprocess_sensor_data(df_raw)
        
        col_kiri, col_tengah, col_kanan = st.columns([1.1, 1.4, 1.1], gap="medium")
        
        with col_kiri:
            indikator_kiri_realtime()
            
        with col_tengah:
            st.markdown("<div class='stCard' style='padding-top:10px;'>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; margin-top:0;'>Visualisasi Peta Spasial</h5>", unsafe_allow_html=True)
            peta_realtime_fragment()
            peta_regional_fragment()

            # === TOMBOL KE-3: MULTIMODAL NEW TAB ===
            st.markdown("""
            <a href="?page=multimodal" target="_blank" style="text-decoration: none;">
                <button style="width: 100%; padding: 12px 16px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #ffffff; border: none; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 15px; margin-top: 10px; font-weight: bold; transition: 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                    📸 Buka Dashboard Multimodal (YOLO + IoT)
                </button>
            </a>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_kanan:
            st.markdown("<div class='stCard' style='padding-top:10px;'>", unsafe_allow_html=True)
            st.markdown("<h5 style='text-align: center; margin-top:0;'>Area Pantauan</h5>", unsafe_allow_html=True)
            try:
                st.image(Image.open("forestiot4.jpg"), use_container_width=True)
            except Exception:
                st.info("Gambar 'forestiot4.jpg' tidak ditemukan di direktori aplikasi.")
            st.markdown("</div>", unsafe_allow_html=True)
                    
        if res[0] is not None and not isinstance(res[0], str):
            df, clean_df, scaled_all, fitur = res

            st.markdown("<div class='section-title' style='margin-top: 25px;'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="scrollable-table stCard" style="margin-bottom: 25px;">
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>
                    <tr>
                        <th style='background-color:#e0e0e0;'>Warna</th>
                        <th style='background-color:#e0e0e0;'>Tingkat Resiko / Intensitas</th>
                        <th style='background-color:#e0e0e0;'>Keterangan</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='background-color:#228B22; color:white;'>
                        <td>Green</td><td>Low / Rendah</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran rendah. Intensitas api pada kategori rendah. Api mudah dikendalikan.</td>
                    </tr>
                    <tr style='background-color:#FFD700; color:black;'>
                        <td>Yellow</td><td>Moderate / Sedang</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sedang. Intensitas api pada kategori sedang. Api relatif masih cukup mudah dikendalikan.</td>
                    </tr>
                    <tr style='background-color:#FF6347; color:white;'>
                        <td>Orange</td><td>High / Tinggi</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran tinggi. Intensitas api pada kategori tinggi. Api sulit dikendalikan.</td>
                    </tr>
                    <tr style='background-color:#8B0000; color:white;'>
                        <td>Red</td><td>Very High / Sangat Tinggi</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sangat tinggi. Intensitas api pada kategori sangat tinggi. Api sangat sulit dikendalikan.</td>
                    </tr>
                </tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title' style='margin-bottom: 15px;'>Visualisasi Tren Data Sensor</div>", unsafe_allow_html=True)

            df_chart = clean_df.copy()
            waktu_clean = df['Waktu'].astype(str).str.replace(' - ', ' ', regex=False)
            df_chart['Waktu_DT'] = pd.to_datetime(waktu_clean, errors='coerce')
            df_chart = df_chart.dropna(subset=['Waktu_DT'])

            if not df_chart.empty:
                df_chart = df_chart.set_index('Waktu_DT')
                df_daily = df_chart[fitur].resample('D').mean().dropna()
                df_daily = df_daily.tail(15)

                chart_rename = {
                    'Tavg: Temperatur rata-rata (°C)': 'Suhu (°C)',
                    'RH_avg: Kelembapan rata-rata (%)': 'Kelembapan (%)',
                    'RR: Curah hujan (mm)': 'Curah Hujan (mm)',
                    'ff_avg: Kecepatan angin rata-rata (m/s)': 'Kecepatan Angin (m/s)',
                    'Kelembaban Permukaan Tanah': 'Kelembaban Tanah (%)'
                }
                df_daily = df_daily.rename(columns=chart_rename)
                df_vis = df_daily.reset_index()

                x_axis = alt.X('Waktu_DT:T', axis=alt.Axis(format='%d %b %Y', title='Tanggal', labelAngle=-45, grid=False, tickCount=df_vis.shape[0]))

                tab_all, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Semua Data", "🌡️ Suhu Udara", "💧 Kelembapan Udara", "🌧️ Curah Hujan", "💨 Kecepatan Angin", "🌱 Kelembapan Tanah"
                ])

                with tab_all:
                    df_melted = df_vis.melt(id_vars=['Waktu_DT'], var_name='Parameter Sensor', value_name='Nilai')
                    satuan_map = {
                        'Suhu (°C)': '°C', 'Kelembapan (%)': '%', 'Curah Hujan (mm)': 'mm', 'Kecepatan Angin (m/s)': 'm/s', 'Kelembaban Tanah (%)': '%'
                    }
                    df_melted['Satuan'] = df_melted['Parameter Sensor'].map(satuan_map)
                    df_melted['LabelText'] = df_melted.apply(lambda row: f"{row['Nilai']:.1f} {row['Satuan']}", axis=1)
                    
                    selection = alt.selection_point(fields=['Parameter Sensor'], bind='legend')

                    chart_base = alt.Chart(df_melted).mark_line(strokeWidth=3, interpolate='monotone').encode(
                        x=x_axis,
                        y=alt.Y('Nilai:Q', title='Nilai Pembacaan', axis=alt.Axis(grid=True, gridDash=[3,3])),
                        color=alt.Color('Parameter Sensor:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
                        tooltip=['Waktu_DT:T', 'Parameter Sensor:N', alt.Tooltip('Nilai:Q', format='.1f')]
                    )

                    points = chart_base.mark_circle(size=60, opacity=0.8).encode(
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
                    )

                    text_labels = chart_base.mark_text(
                        align='center', baseline='bottom', dy=-10, fontSize=11, fontWeight='bold'
                    ).encode(text=alt.Text('LabelText:N'), opacity=alt.condition(selection, alt.value(1), alt.value(0)))

                    chart_all = (chart_base + points + text_labels).add_params(selection).properties(height=450).interactive()
                    st.altair_chart(chart_all, use_container_width=True)

                with tab1:
                    chart_temp = alt.Chart(df_vis).mark_line(color="#ff5733", strokeWidth=3, point=alt.OverlayMarkDef(color="#ff5733", size=50)).encode(x=x_axis, y=alt.Y('Suhu (°C):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Suhu (°C):Q', format='.1f')]).properties(height=350).interactive()
                    st.altair_chart(chart_temp, use_container_width=True)

                with tab2:
                    chart_hum = alt.Chart(df_vis).mark_line(color="#33d4ff", strokeWidth=3, point=alt.OverlayMarkDef(color="#33d4ff", size=50)).encode(x=x_axis, y=alt.Y('Kelembapan (%):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Kelembapan (%):Q', format='.1f')]).properties(height=350).interactive()
                    st.altair_chart(chart_hum, use_container_width=True)

                with tab3:
                    base = alt.Chart(df_vis).encode(x=x_axis)
                    bar = base.mark_bar(color="#335eff", opacity=0.7, size=25).encode(y=alt.Y('Curah Hujan (mm):Q', title='Curah Hujan (mm)'), tooltip=['Waktu_DT:T', alt.Tooltip('Curah Hujan (mm):Q', format='.1f')])
                    line = base.mark_line(color="#ff0000", strokeWidth=2).encode(y=alt.Y('Curah Hujan (mm):Q'))
                    point = base.mark_circle(color="#ff0000", size=60).encode(y=alt.Y('Curah Hujan (mm):Q'))
                    chart_rain = (bar + line + point).properties(height=350).interactive()
                    st.altair_chart(chart_rain, use_container_width=True)

                with tab4:
                    chart_wind = alt.Chart(df_vis).mark_line(color="#a833ff", strokeWidth=3, point=alt.OverlayMarkDef(color="#a833ff", size=50)).encode(x=x_axis, y=alt.Y('Kecepatan Angin (m/s):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Kecepatan Angin (m/s):Q', format='.1f')]).properties(height=350).interactive()
                    st.altair_chart(chart_wind, use_container_width=True)

                with tab5:
                    chart_soil = alt.Chart(df_vis).mark_line(color="#33ff5e", strokeWidth=3, point=alt.OverlayMarkDef(color="#33ff5e", size=50)).encode(x=x_axis, y=alt.Y('Kelembaban Tanah (%):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Kelembaban Tanah (%):Q', format='.1f')]).properties(height=350).interactive()
                    st.altair_chart(chart_soil, use_container_width=True)
            else:
                st.info("Data tidak dapat diproses untuk grafik. Pastikan format kolom Waktu pada file CSV valid.")

            st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            def to_excel(df_to_save: pd.DataFrame) -> bytes:
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df_to_save.to_excel(writer, index=False, sheet_name='Prediksi')
                writer.close()
                return output.getvalue()

            df_xlsx = to_excel(df)
            st.download_button(
                label="📥 Download Hasil Prediksi Kebakaran sebagai XLSX",
                data=df_xlsx, file_name="hasil_prediksi_kebakaran.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    main_dashboard()

    # === PENGUJIAN MANUAL & TEKS ===
    if "man_suhu" not in st.session_state: st.session_state.man_suhu = 30.0
    if "man_kel" not in st.session_state: st.session_state.man_kel = 65.0
    if "man_curah" not in st.session_state: st.session_state.man_curah = 10.0
    if "man_angin" not in st.session_state: st.session_state.man_angin = 3.0
    if "man_tanah" not in st.session_state: st.session_state.man_tanah = 50.0
    if "manual_result" not in st.session_state: st.session_state.manual_result = None

    def reset_manual():
        st.session_state.man_suhu = 0.0
        st.session_state.man_kel = 0.0
        st.session_state.man_curah = 0.0
        st.session_state.man_angin = 0.0
        st.session_state.man_tanah = 0.0
        st.session_state.manual_result = None

    def do_predict_manual():
        input_df = pd.DataFrame([{
            'Tavg: Temperatur rata-rata (°C)': st.session_state.man_suhu,
            'RH_avg: Kelembapan rata-rata (%)': st.session_state.man_kel,
            'RR: Curah hujan (mm)': st.session_state.man_curah,
            'ff_avg: Kecepatan angin rata-rata (m/s)': st.session_state.man_angin,
            'Kelembaban Permukaan Tanah': st.session_state.man_tanah
        }])
        scaled_manual = scaler.transform(input_df)
        st.session_state.manual_result = convert_to_label(model.predict(scaled_manual)[0])

    @st.fragment
    def manual_prediction_ui():
        st.markdown("<div class='section-title' style='margin-top: 30px;'>Pengujian Menggunakan Data Meteorologi Manual</div>", unsafe_allow_html=True)
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Suhu Udara (°C)", key="man_suhu")
            st.number_input("Kelembapan Udara (%)", key="man_kel")
        with col2:
            st.number_input("Curah Hujan (mm)", key="man_curah")
            st.number_input("Kecepatan Angin (m/s)", key="man_angin")
        with col3:
            st.number_input("Kelembaban Tanah (%)", key="man_tanah")

        btn_pred, btn_reset, _ = st.columns([2, 2, 8])
        with btn_pred: st.button("🔍 Prediksi Manual", on_click=do_predict_manual, use_container_width=True)
        with btn_reset: st.button("🧼 Reset Manual", on_click=reset_manual, use_container_width=True)

        if st.session_state.manual_result:
            hasil = st.session_state.manual_result
            font, bg = risk_styles.get(hasil, ("black", "white"))
            st.markdown(
                f"<div style='color:{font}; background-color:{bg}; padding:15px; border-radius:8px; margin-top:20px; font-size:16px; text-align:center;'>"
                f"Prediksi Risiko Kebakaran: <b>{hasil}</b></div>", unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if "txt_input" not in st.session_state: st.session_state.txt_input = ""
    if "txt_result" not in st.session_state: st.session_state.txt_result = None
    if "txt_preprocessing" not in st.session_state: st.session_state.txt_preprocessing = {}

    def reset_text():
        st.session_state.txt_input = ""
        st.session_state.txt_result = None
        st.session_state.txt_preprocessing = {}

    def do_predict_text():
        if st.session_state.txt_input.strip() == "":
            st.warning("Harap masukkan deskripsi teks terlebih dahulu.")
        elif vectorizer is None or model_text is None:
            st.error("Model teks gagal dimuat.")
        else:
            try:
                raw_text = st.session_state.txt_input
                text_lower = raw_text.lower()
                text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
                text_stopword = stopword_remover.remove(text_clean)
                tokens = text_stopword.split()
                token_display = "[" + ", ".join(tokens) + "]"
                text_stemmed = stemmer.stem(" ".join(tokens))

                X_trans = vectorizer.transform([text_stemmed])
                feature_names = vectorizer.get_feature_names_out()
                dense_vector = X_trans.todense().tolist()[0]

                tfidf_details = [{"Kata (Term)": word, "Skor TF-IDF": round(score, 4)} for word, score in zip(feature_names, dense_vector) if score > 0]
                tfidf_details = sorted(tfidf_details, key=lambda x: x["Skor TF-IDF"], reverse=True)
                df_tfidf = pd.DataFrame(tfidf_details)

                prob_dict = {}
                try:
                    proba = model_text.predict_proba(X_trans)[0]
                    prob_dict = {"Low / Rendah": proba[0], "Moderate / Sedang": proba[1], "High / Tinggi": proba[2], "Very High / Sangat Tinggi": proba[3]}
                except: pass

                pred = model_text.predict(X_trans)[0]
                label_text = convert_to_label(pred)

                st.session_state.txt_preprocessing = {
                    "raw": raw_text, "case_folding": text_lower, "cleansing": text_clean, "stopword": text_stopword, "tokenizing": token_display, "stemming": text_stemmed, "tfidf_df": df_tfidf, "prob_dict": prob_dict
                }
                st.session_state.txt_result = label_text

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses input teks: {e}")

    @st.fragment
    def text_prediction_ui():
        st.markdown("<div class='section-title' style='margin-top: 20px;'>Pengujian Menggunakan Data Teks</div>", unsafe_allow_html=True)
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.text_area("Masukkan deskripsi lingkungan:", key="txt_input", height=120)

        btn_pred_text, btn_reset_text, _ = st.columns([2, 2, 8])
        with btn_pred_text: st.button("🔍 Prediksi Teks", on_click=do_predict_text, use_container_width=True)
        with btn_reset_text: st.button("🧼 Reset Teks", on_click=reset_text, use_container_width=True)
            
        if st.session_state.txt_result:
            with st.expander("🛠️ Klik untuk melihat hasil setiap tahapan Pre-processing & Keputusan Model", expanded=False):
                steps = st.session_state.txt_preprocessing
                if steps:
                    st.markdown("**1. Original Text**")
                    st.info(steps.get("raw", "-"))
                    st.markdown("**2. Cleansing (Penghapusan Karakter Khusus & Angka)**")
                    st.info(steps.get("cleansing", "-"))
                    st.markdown("**3. Stopword (Penghapusan Kata)**")
                    st.info(steps.get("stopword", "-"))
                    st.markdown("**4. Tokenization (Pemenggalan Kata)**")
                    st.info(steps.get("tokenizing", "[]"))
                    st.markdown("**5. Stemming (Pemotongan Imbuhan)**")
                    st.info(steps.get("stemming", "-"))
                    st.markdown("**6. Ekstraksi Fitur (TF-IDF)**")
                    df_tfidf_display = steps.get("tfidf_df")
                    if df_tfidf_display is not None and not df_tfidf_display.empty: st.dataframe(df_tfidf_display, use_container_width=True)
                    else: st.warning("Kata-kata pada input ini tidak dikenali dalam kosakata model Anda.")

                    st.markdown("**7. Analisis Keputusan Model (Probabilitas HSEL)**")
                    prob_dict = steps.get("prob_dict")
                    if prob_dict:
                        for label, prob in prob_dict.items():
                            st.markdown(f"**{label}** ({prob*100:.1f}%)")
                            st.progress(float(prob))
                    else: st.info("Model ini tidak menyediakan metrik probabilitas.")

            hasil = st.session_state.txt_result
            font, bg = risk_styles.get(hasil, ("black", "white"))
            st.markdown(
                f"<div style='color:{font}; background-color:{bg}; padding:15px; border-radius:8px; margin-top:20px; font-size:16px; text-align:center;'>"
                f"Prediksi Risiko Kebakaran: <b>{hasil}</b></div>", unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    manual_prediction_ui()
    text_prediction_ui()

    # === FOOTER ===
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top: 20px; background-color: #2c3e50; padding: 15px 20px; border-radius: 12px; text-align: center; color: white;'>
        <p style='margin: 0; font-size: 26px; font-weight: bold; line-height: 1.2;'>Smart Fire Prediction HSEL Model</p>
        <p style='margin: 0; font-size: 14px; line-height: 1.5; color:#bdc3c7;'>Sistem Cerdas Peringatan Dini Kebakaran Lahan<br>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026</p>
    </div>
    """, unsafe_allow_html=True)
