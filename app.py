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
    }
    .scrollable-table { overflow-x: auto; }
    
    iframe[title="folium_static"] {
        border: 4px solid #444 !important;
        border-radius: 4px !important;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.2) !important;
    }
    
    .yolo-frame {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 25px;
        background-color: #ffffff;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        height: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# === FUNGSI BANTUAN ===
def convert_day_to_indonesian(day_name):
    return {'Monday':'Senin','Tuesday':'Selasa','Wednesday':'Rabu','Thursday':'Kamis','Friday':'Jumat','Saturday':'Sabtu','Sunday':'Minggu'}.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {'January':'Januari','February':'Februari','March':'Maret','April':'April','May':'Mei','June':'Juni','July':'Juli','August':'Agustus','September':'September','October':'Oktober','November':'November','December':'Desember'}.get(month_name, month_name)

def convert_to_label(pred):
    return {0: "Low / Rendah", 1: "Moderate / Sedang", 2: "High / Tinggi", 3: "Very High / Sangat Tinggi"}.get(pred, "Unknown")

risk_styles = {
    "Low / Rendah": ("white", "blue"),
    "Moderate / Sedang": ("white", "green"),
    "High / Tinggi": ("black", "yellow"),
    "Very High / Sangat Tinggi": ("white", "red")
}

def get_image_base64(path):
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

def get_multimodal_decision(visual_label, iot_label):
    if visual_label == 1 and "Very High" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Darurat:</b> Bunyikan alarm darurat dan lakukan evakuasi.</li><li><b>Operasional:</b> Kerahkan seluruh armada Damkar/BPBD ke lokasi.</li><li><b>Koordinasi:</b> Laporkan ke komando pusat provinsi.</li></ul>"
        return "Kebakaran Telah Terjadi", "Indikasi sangat kuat bahwa kebakaran telah terjadi berdasarkan konfirmasi visual dan sensor.", "#B22222", "🔥", tl
    elif visual_label == 1 and "High" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Operasional:</b> Kerahkan regu pemadam terdekat ke titik koordinat.</li><li><b>Isolasi:</b> Lakukan penyekatan area cegah api merembet.</li><li><b>Peringatan:</b> Berikan peringatan waspada kepada warga sekitar.</li></ul>"
        return "Kebakaran Sangat Mungkin", "Gambaran visual api dan kondisi lingkungan berisiko tinggi memperkuat dugaan kebakaran aktif.", "#DC143C", "🚨", tl
    elif visual_label == 1 and "Moderate" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Verifikasi:</b> Kirim tim cepat untuk memverifikasi skala api.</li><li><b>Tindakan:</b> Lakukan pemadaman dini sebelum api membesar.</li><li><b>Pantau:</b> Cek arah angin untuk prediksi perembetan.</li></ul>"
        return "Kemungkinan Kebakaran", "Visual mendeteksi api, namun kondisi sensor menunjukkan tingkat risiko sedang.", "#FF8C00", "⚠️", tl
    elif visual_label == 1 and "Low" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Inspeksi:</b> Pengecekan kemungkinan pembakaran sampah/terkendali.</li><li><b>Hukuman:</b> Tegur pelaku jika aktivitas ilegal.</li><li><b>Pemadaman:</b> Padamkan api agar tidak menimbulkan asap.</li></ul>"
        return "Terdeteksi Api Isolated", "Visual menunjukkan api meskipun kondisi lingkungan kurang mendukung penyebaran. Kemungkinan aktivitas manusia.", "#FFA500", "🟠", tl
    elif visual_label == 0 and "Very High" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Waspada:</b> Tetapkan status siaga darurat tingkat lokal.</li><li><b>Patroli:</b> Sisir area rawan secara fisik dan drone.</li><li><b>Preventif:</b> Larang keras seluruh aktivitas api di area rawan.</li></ul>"
        return "Risiko Kebakaran Sangat Tinggi", "Belum ada deteksi visual api, namun lingkungan sangat rentan kebakaran. Waspada dini diperlukan.", "#8B0000", "🌡️", tl
    elif visual_label == 0 and "High" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Siaga:</b> Siagakan armada pemadam di pos terdekat.</li><li><b>Pemantauan:</b> Tingkatkan frekuensi pantauan dashboard.</li><li><b>Edukasi:</b> Siarkan peringatan ke warga.</li></ul>"
        return "Potensi Kebakaran Tinggi", "Belum ada api terdeteksi, tetapi kondisi sekitar menunjukkan risiko tinggi kebakaran.", "#FF6347", "🔥", tl
    elif visual_label == 0 and "Moderate" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Rutin:</b> Lakukan patroli berkala sesuai jadwal standar.</li><li><b>Pengawasan:</b> Awasi indikasi pembakaran perkebunan.</li><li><b>Analisis:</b> Pantau pergerakan kelembapan harian.</li></ul>"
        return "Kondisi Rentan Kebakaran", "Lingkungan menunjukkan risiko sedang terhadap kebakaran. Pemantauan disarankan.", "#FFD700", "⚠️", tl
    elif visual_label == 0 and "Low" in iot_label: 
        tl = "<ul style='margin:0; padding-left:20px;'><li><b>Rutin:</b> Lanjutkan pemantauan reguler melalui dashboard.</li><li><b>Dokumentasi:</b> Catat dan laporkan kondisi aman.</li><li><b>Teknis:</b> Pelihara fungsi alat IoT secara berkala.</li></ul>"
        return "Tidak Terindikasi Kebakaran", "Tidak ada api terdeteksi dan kondisi lingkungan tergolong aman.", "#228B22", "✅", tl
    else: 
        return "Status Tidak Diketahui", "Data tidak mencukupi untuk menarik kesimpulan.", "#808080", "❓", "Menunggu Data Lengkap"

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
            if cand in df.columns: actual_rename[cand] = target_name; break

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
    
    logo_b64 = get_image_base64("logo upi yptk.png")
    img_tag = f'<img src="data:image/png;base64,{logo_b64}" width="65" style="border-radius:8px; background:white; padding:5px;">' if logo_b64 else '🔥'
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 25px 30px; border-radius: 12px; color: white; display: flex; align-items: center; margin-bottom: 25px; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
        <div style="margin-right: 20px;">
            {img_tag}
        </div>
        <div>
            <h1 style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 1px; color: #ffffff;">COMMAND CENTER MULTIMODAL</h1>
            <p style="margin: 5px 0 0 0; font-size: 15px; color: #b2bec3;">Sistem Peringatan Dini Cerdas: Integrasi Kecerdasan Visual Hybrid YOLO-ViT+GRU dan Sensor Lingkungan HSEL Terpadu</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if yolo_model is None:
        st.error("🚨 Model YOLO (`best.pt`) tidak ditemukan. Pastikan Anda telah mengunggah file `best.pt` ke direktori aplikasi!")
    else:
        col_vis, col_sensor = st.columns([1.15, 1.25], gap="large")
        
        # === KOLOM VISUAL (KIRI) ===
        with col_vis:
            st.markdown("<div class='yolo-frame'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color:#2c3e50; border-bottom:2px solid #e0e0e0; padding-bottom:10px; margin-top:0;'>👁️ AI Visual (Hybrid YOLO-ViT+GRU)</h4>", unsafe_allow_html=True)
            
            input_method = st.radio("Pilih Sumber Pengamatan:", ["📁 Unggah File Citra", "🎥 Kamera Langsung / USB"], horizontal=True)
            img_to_process = None
            
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
                    st.image(res_rgb, caption="Hasil Analisis Visi Komputer Hybrid YOLO-ViT+GRU", use_container_width=True)
                    
                    if st.session_state.yolo_fire_detected:
                        st.error(f"🔥 Sistem mendeteksi keberadaan {len(detections)} titik api aktif!")
                    else:
                        st.success("✅ Tidak terdeteksi adanya anomali api pada citra ini.")
            else:
                st.session_state.yolo_fire_detected = None
                try:
                    st.image(Image.open("hutan.png"), use_container_width=True, caption="Menunggu Input Visual (Kamera/Unggah Citra)")
                except Exception:
                    try: 
                        st.image(Image.open("forestiot4.jpg"), use_container_width=True, caption="Menunggu Input Visual (Kamera/Unggah Citra)")
                    except:
                        st.info("Menunggu input visual...")
                
            st.markdown("</div>", unsafe_allow_html=True)

        # === KOLOM SENSOR & KEPUTUSAN (KANAN) ===
        with col_sensor:
            @st.fragment(run_every=7)
            def sensor_and_decision_fragment():
                df_raw = load_data()
                res = preprocess_sensor_data(df_raw)
                
                st.markdown("<div style='background-color:#fff; padding:25px; border-radius:12px; border:2px solid #e2e8f0; box-shadow:0 6px 12px rgba(0,0,0,0.08); margin-bottom:0;'>", unsafe_allow_html=True)
                
                if res[0] is not None and not isinstance(res[0], str):
                    df, clean_df, scaled_all, fitur = res
                    last_row = df.iloc[-1]
                    last_num = clean_df.iloc[-1]
                    hsel_risk = last_row["Prediksi Kebakaran"]
                    
                    # Definisikan tanggal_valid
                    waktu = pd.to_datetime(last_row['Waktu'], errors='coerce')
                    if pd.isna(waktu):
                        try: waktu = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
                        except Exception: waktu = None

                    if isinstance(waktu, pd.Timestamp):
                        hari = convert_day_to_indonesian(waktu.strftime('%A'))
                        bulan = convert_month_to_indonesian(waktu.strftime('%B'))
                        tanggal = waktu.strftime(f'%d {bulan} %Y')
                        tanggal_valid = waktu.strftime('%d %B %Y, %H:%M WIB')
                    else:
                        hari, tanggal = "-", str(last_row['Waktu'])
                        tanggal_valid = str(last_row['Waktu'])
                        
                    font, bg = risk_styles.get(hsel_risk, ("black", "white"))
                    
                    # === 1. TOP SECTION: MAP & IOT DEVICE ===
                    st.markdown("<h4 style='color:#2c3e50; border-bottom:2px solid #e0e0e0; padding-bottom:10px; margin-top:0;'>🗺️ Peta Konteks & Sensor</h4>", unsafe_allow_html=True)
                    
                    cm1, cm2 = st.columns([1.5, 1], gap="medium")
                    with cm1:
                        st.markdown("<div style='font-size:13px; font-weight:bold; color:#555; margin-bottom:5px;'>Peta Konteks Lokal</div>", unsafe_allow_html=True)
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
                        folium_static(m_mini, width=360, height=260)
                        
                    with cm2:
                        st.markdown("<div style='font-size:13px; font-weight:bold; color:#555; margin-bottom:5px;'>Alat Node IoT</div>", unsafe_allow_html=True)
                        try:
                            st.image(Image.open("alat_iot.png"), use_container_width=True)
                        except Exception:
                            st.info("Gambar alat_iot.png tidak ditemukan.")
                    
                    st.markdown("<hr style='margin:25px 0 15px 0; border: 1px dashed #e0e0e0;'>", unsafe_allow_html=True)
                    
                    # === 2. MIDDLE SECTION: KONTEKS LINGKUNGAN ===
                    st.markdown("<h4 style='color:#2c3e50; border-bottom:2px solid #e0e0e0; padding-bottom:10px; margin-top:0;'>📡 Konteks Lingkungan (IoT)</h4>", unsafe_allow_html=True)
                    
                    st.markdown("<p style='font-size:13px; color:gray; margin-bottom:10px;'>Pembacaan Node Sensor Real-Time:</p>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🌡️ Suhu", f"{float(last_num[fitur[0]]):.1f} °C")
                    m2.metric("💧 Kelembapan", f"{float(last_num[fitur[1]]):.1f} %")
                    m3.metric("🌧️ Curah Hujan", f"{float(last_num[fitur[2]]):.1f} mm")
                    
                    m4, m5, _ = st.columns([1,1,1])
                    m4.metric("💨 Kec. Angin", f"{float(last_num[fitur[3]]):.1f} m/s")
                    m5.metric("🌱 Kel. Tanah", f"{float(last_num[fitur[4]]):.1f} %")
                    
                    st.markdown(
                        f"<div style='background-color:{bg}; color:{font}; padding:15px; border-radius:8px; font-weight:bold; margin-top:15px; font-size:16px; text-align:center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
                        f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran:<br>"
                        f"<span style='text-decoration: underline; font-size: 24px;'>{hsel_risk}</span></div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("<hr style='margin:25px 0 15px 0; border: 1px dashed #e0e0e0;'>", unsafe_allow_html=True)
                    
                    # === 3. BOTTOM SECTION: LOGIKA MULTIMODAL KEPUTUSAN ===
                    st.markdown("<h4 style='color:#2c3e50; margin-bottom:0;'>🧠 Output Keputusan Terpadu</h4>", unsafe_allow_html=True)
                    fire_detected = st.session_state.get("yolo_fire_detected", None)
                    
                    if fire_detected is None:
                        st.info("ℹ️ Menunggu konfirmasi visual (Silakan unggah atau ambil gambar di panel kiri).")
                    else:
                        visual_val = 1 if fire_detected else 0
                        title, desc, color, icon, tindak_lanjut = get_multimodal_decision(visual_val, hsel_risk)
                        font_col = "white" if color not in ["#FFD700", "#FF8C00", "#FFA500"] else "black"
                        
                        st.markdown(f"""
                        <div style='background-color: {color}; color: {font_col}; padding: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.15); margin-top: 10px;'>
                            <h3 style='color: {font_col}; margin-top: 0; font-size:20px;'>{icon} {title}</h3>
                            <p style='font-size: 14px; margin-bottom: 0; line-height:1.4;'>{desc}</p>
                        </div>
                        <div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid {color}; padding: 15px; border-radius: 8px; margin-top: 15px;'>
                            <b style='color: #2c3e50; font-size: 14px;'>📋 Rekomendasi Tindak Lanjut:</b>
                            <div style='font-size: 13px; color: #4a5568; margin-top: 8px; line-height: 1.5;'>
                                {tindak_lanjut}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # === 4. PRODUCED BY (DIPERBESAR & LEBIH PROPORIONAL) ===
                    logo_upi_b64 = get_image_base64("logo upi yptk.png")
                    logo_upi_tag = f'<img src="data:image/png;base64,{logo_upi_b64}" style="width: 80px; height: auto;" alt="Logo">' if logo_upi_b64 else ''
                    
                    st.markdown(f"""
                    <div style="background: #fdfdfd; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-top:30px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
                        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 15px;">
                            {logo_upi_tag}
                            <div style="text-align: left; line-height: 1.5;">
                                <b style="font-size: 12px; color: #718096; text-transform: uppercase; letter-spacing: 1px;">Produced By</b><br>
                                <span style="font-size: 16px; font-weight: bold; color: #2d3748;">Multimodal HSEL & Hybrid YOLO</span><br>
                                <span style="font-size: 13px; font-style: italic; color: #4a5568;">Mahasiswa Doctoral TI UPI YPTK Padang</span>
                            </div>
                        </div>
                        <div style="border-top: 1px dashed #cbd5e0; padding-top: 12px; text-align: center;">
                            <div style="font-size: 13px; color: #3182ce; font-weight: 600; margin-bottom: 5px;">Processed: <span style="color: #2b6cb0; font-weight: normal;">{tanggal_valid}</span></div>
                            <div style="font-size: 12px; color: #718096;">Data Source: Sensor IoT Lokal & Vision AI</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Data IoT Terputus atau Tidak Tersedia.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            sensor_and_decision_fragment()


# -------------------------------------------------------------------------
# HALAMAN 1 (DEFAULT): DASHBOARD UTAMA
# -------------------------------------------------------------------------
else:
    # === HEADER ===
    col1, col2 = st.columns([1, 9])
    with col1:
        try: st.image("logo.png", width=170)
        except: pass
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
            try: waktu = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
            except Exception: waktu = None

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

        st.markdown("<h5 style='text-align: center;'>Data Sensor Realtime</h5>", unsafe_allow_html=True)
        sensor_html = "<table style='width: 100%; border-collapse: collapse;'>"
        sensor_html += "<thead><tr><th>Variabel</th><th>Value</th></tr></thead><tbody>"
        for i in range(len(sensor_df)):
            var = sensor_df.iloc[i, 0]
            val = sensor_df.iloc[i, 1]
            sensor_html += f"<tr><td style='padding:6px;'>{var}</td><td style='padding:6px;'>{val}</td></tr>"
        sensor_html += "</tbody></table>"
        st.markdown(sensor_html, unsafe_allow_html=True)

        st.markdown(
            f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold; margin-top: 15px;'>"
            f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
            f"<span style='text-decoration: underline; font-size: 22px;'>{risk_label}</span></p>",
            unsafe_allow_html=True
        )

        with st.expander("📊 Analisis Keputusan Model (XAI)"):
            st.markdown("<span style='font-size:14px; color:gray;'>Grafik di bawah menunjukkan seberapa besar setiap parameter sensor berkontribusi terhadap prediksi saat ini.</span>", unsafe_allow_html=True)
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
                    kontribusi.append({"fitur": nama_f, "shap_val": float(shap_v), "pct": pct_f})
                kontribusi = sorted(kontribusi, key=lambda x: x["pct"], reverse=True)

                st.markdown("<h4 style='margin-top: 25px;'>Analisis Detail Keputusan Model (XAI)</h4>", unsafe_allow_html=True)
                if risk_label == "Low / Rendah": st.success("Kondisi lingkungan saat ini terpantau **sangat aman dan stabil**. Berdasarkan analisis *Explainable AI* (SHAP), berikut adalah dominasi faktor-faktor alam yang sukses meredam potensi kebakaran:")
                elif risk_label == "Moderate / Sedang": st.info("Kondisi lingkungan saat ini terpantau **cukup stabil namun memerlukan pemantauan berkala**. Berikut adalah rincian faktor yang memengaruhi keseimbangan risiko saat ini:")
                elif risk_label == "High / Tinggi": st.warning("Kondisi lingkungan saat ini terpantau **kritis**. Berdasarkan analisis *Explainable AI* (SHAP), terdapat ancaman bahaya yang dipicu oleh memburuknya faktor-faktor berikut:")
                elif risk_label == "Very High / Sangat Tinggi": st.error("Kondisi lingkungan saat ini berada pada fase **SANGAT EKSTREM**. Faktor-faktor alam berikut secara masif mendorong eskalasi kebakaran lahan ke tingkat bahaya tertinggi:")

                icons = ["🔴", "🟠", "🟡", "🟢", "⚪"]
                for i, factor in enumerate(kontribusi):
                    icon = icons[i] if i < len(icons) else "⚪"
                    nama_fitur = str(factor['fitur']).lower()
                    persen = factor['pct']
                    arah = factor['shap_val']

                    st.markdown(f"**{icon} {factor['fitur'].title()} ({persen:.1f}%)**")
                    if persen < 5.0:
                        if arah > 0: st.write("- Memberikan dorongan minor terhadap potensi risiko. Pengaruhnya saat ini tertutupi oleh faktor dominan lainnya.")
                        else: st.write("- Memiliki efek peredaman yang sangat kecil terhadap prediksi saat ini. Kondisinya belum cukup signifikan untuk memengaruhi status lingkungan secara keseluruhan.")
                    else:
                        if "tanah" in nama_fitur:
                            if arah > 0: st.write("- **Meningkatkan Risiko:** Merupakan faktor pendorong utama. Kelembaban tanah yang rendah menunjukkan kondisi lahan yang teramat kering.")
                            else: st.write("- **Meredam Risiko:** Kelembaban tanah terdeteksi cukup tinggi (basah/lembab). Bertindak sebagai tameng alami.")
                        elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur:
                            if arah > 0: st.write("- **Meningkatkan Risiko:** Udara yang kering mempercepat proses pengeringan bahan bakar alami.")
                            else: st.write("- **Meredam Risiko:** Tingkat kelembapan udara yang tinggi membantu menjaga kebasahan partikel.")
                        elif "angin" in nama_fitur or "ff" in nama_fitur:
                            if arah > 0: st.write("- **Mempercepat Eskalasi:** Kecepatan angin saat ini berisiko memperluas area kebakaran dengan sangat cepat.")
                            else: st.write("- **Kondisi Stabil:** Pergerakan angin yang relatif lambat dan tenang tidak memberikan ancaman berarti.")
                        elif "suhu" in nama_fitur or "temperatur" in nama_fitur or "tavg" in nama_fitur:
                            if arah > 0: st.write("- **Meningkatkan Risiko:** Suhu lingkungan yang sangat panas memicu penguapan air dari vegetasi.")
                            else: st.write("- **Meredam Risiko:** Suhu udara yang tergolong sejuk atau normal menjaga stabilitas termal lingkungan.")
                        elif "hujan" in nama_fitur or "rr" in nama_fitur:
                            if arah > 0: st.write("- **Meningkatkan Risiko:** Ketiadaan curah hujan menghilangkan faktor pendingin alami utama.")
                            else: st.write("- **Meredam Risiko:** Curah hujan yang turun merupakan faktor pendingin krusial.")
                        else:
                            if arah > 0: st.write("- Secara kalkulasi sistem berkontribusi dalam meningkatkan potensi risiko kebakaran.")
                            else: st.write("- Secara kalkulasi sistem berkontribusi menstabilkan potensi risiko kebakaran.")
            except Exception as e:
                st.error(f"Visualisasi XAI belum dapat diproses: {e}")

        with st.expander("Tindak Lanjut Instansi"):
            if risk_label == "Low / Rendah": st.markdown("""<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Monitoring rutin kondisi lingkungan</li><li>Patroli berkala ringan</li><li>Edukasi preventif kepada masyarakat</li><li>Dokumentasi dan pelaporan kondisi normal</li></ul>""", unsafe_allow_html=True)
            elif risk_label == "Moderate / Sedang": st.markdown("""<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Peningkatan frekuensi patroli</li><li>Penyampaian peringatan dini terbatas</li><li>Koordinasi internal BPBD dan aparat desa</li><li>Pengawasan aktivitas pembakaran terbuka</li></ul>""", unsafe_allow_html=True)
            elif risk_label == "High / Tinggi": st.markdown("""<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Aktivasi pos siaga tingkat lokal</li><li>Penempatan personel siaga di titik rawan</li><li>Koordinasi dengan TNI/Polri dan Manggala Agni</li><li>Peringatan dini terbuka masyarakat</li><li>Penyiapan peralatan pemadaman awal</li></ul>""", unsafe_allow_html=True)
            elif risk_label == "Very High / Sangat Tinggi": st.markdown("""<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Status siaga darurat tingkat lokal</li><li>Aktivasi penuh posko tanggap darurat</li><li>Mobilisasi tim pemantauan dan pemadam</li><li>Koordinasi lintas sektor (BPBD, TNI, Polri, DLH)</li><li>Penyebaran peringatan dini melalui media resmi</li><li>Pengetatan larangan pembakaran terbuka</li></ul>""", unsafe_allow_html=True)

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

            xai_html = ""
            try:
                data_realtime_scaled = pd.DataFrame(scaled_all[-1:], columns=fitur)
                background_data = pd.DataFrame(shap.sample(scaled_all, 50), columns=fitur)
                explainer = shap.Explainer(model.predict, background_data)
                shap_values = explainer(data_realtime_scaled)
                
                total_abs_shap = sum(abs(v) for v in shap_values[0].values)
                kontribusi_map = []
                for nama_f, shap_v in zip(fitur, shap_values[0].values):
                    pct_f = (abs(float(shap_v)) / total_abs_shap) * 100 if total_abs_shap > 0 else 0.0
                    kontribusi_map.append({"fitur": nama_f, "shap_val": float(shap_v), "pct": pct_f})
                kontribusi_map = sorted(kontribusi_map, key=lambda x: x["pct"], reverse=True)
                
                for factor in kontribusi_map:
                    nama_fitur = str(factor['fitur']).lower()
                    persen = factor['pct']
                    arah = factor['shap_val']
                    icon = "🔴" if arah > 0 else "🟢"
                    
                    if persen < 5.0:
                        icon = "⚪"
                        desc = "Dorongan minor terhadap potensi risiko." if arah > 0 else "Efek peredaman sangat kecil."
                        bg_col = "#f5f5f5"
                        br_col = "#cccccc"
                    else:
                        if "tanah" in nama_fitur: desc = "Meningkatkan Risiko (Kering)" if arah > 0 else "Meredam Risiko (Lembab)"
                        elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur: desc = "Memperburuk (Udara Kering)" if arah > 0 else "Menjaga Kebasahan (Lembap)"
                        elif "angin" in nama_fitur or "ff" in nama_fitur: desc = "Mempercepat Eskalasi (O2)" if arah > 0 else "Kondisi Stabil (Tenang)"
                        elif "suhu" in nama_fitur or "temperatur" in nama_fitur or "tavg" in nama_fitur: desc = "Memicu Penguapan (Panas)" if arah > 0 else "Stabilitas Termal (Normal)"
                        elif "hujan" in nama_fitur or "rr" in nama_fitur: desc = "Tanpa Hujan (Pendingin Hilang)" if arah > 0 else "Faktor Pendingin (Hujan)"
                        else: desc = "Meningkatkan Potensi" if arah > 0 else "Menstabilkan Potensi"
                        
                        bg_col = "#ffebeb" if arah > 0 else "#ebffef"
                        br_col = "#ff4b4b" if arah > 0 else "#21c354"
                    
                    xai_html += f"""
                    <div style='margin-bottom: 6px; padding: 6px; background: {bg_col}; border-left: 3px solid {br_col}; border-radius: 0 4px 4px 0;'>
                        <b style='color:#333; font-size:12px;'>{icon} {factor['fitur'].title()} ({persen:.1f}%)</b><br>
                        <span style='color:#555; font-size:11px;'>{desc}</span>
                    </div>
                    """
            except Exception:
                xai_html = "<i>Data XAI belum siap dimuat.</i>"

            if risk_label == "Low / Rendah":
                tl_html = "<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Monitoring rutin kondisi lingkungan</li><li>Patroli berkala ringan</li><li>Edukasi preventif kepada masyarakat</li><li>Dokumentasi dan pelaporan kondisi normal</li></ul>"
            elif risk_label == "Moderate / Sedang":
                tl_html = "<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Peningkatan frekuensi patroli</li><li>Penyampaian peringatan dini terbatas</li><li>Koordinasi internal BPBD dan aparat desa</li><li>Pengawasan aktivitas pembakaran terbuka</li></ul>"
            elif risk_label == "High / Tinggi":
                tl_html = "<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Aktivasi pos siaga tingkat lokal</li><li>Penempatan personel siaga di titik rawan</li><li>Koordinasi dengan TNI/Polri dan Manggala Agni</li><li>Peringatan dini terbuka masyarakat</li><li>Penyiapan peralatan pemadaman awal</li></ul>"
            else:
                tl_html = "<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:12px;'><li>Status siaga darurat tingkat lokal</li><li>Aktivasi penuh posko tanggap darurat</li><li>Mobilisasi tim pemantauan dan pemadam</li><li>Koordinasi lintas sektor (BPBD, TNI, Polri, DLH)</li><li>Penyebaran peringatan dini melalui media resmi</li><li>Pengetatan larangan pembakaran terbuka</li></ul>"

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
            except Exception: pekanbaru_geojson = None

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
            MousePosition(position="bottomleft", separator=" | ", empty_string="Koordinat tidak tersedia", lng_first=True, num_digits=20, prefix="Posisi:", lat_formatter=formatter, lng_formatter=formatter).add_to(m)
            Fullscreen(position='topright').add_to(m)

            if pekanbaru_geojson and pekanbaru_geojson["features"]:
                folium.GeoJson(pekanbaru_geojson, style_function=lambda feature, color=marker_color: {'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.4}, tooltip=folium.GeoJsonTooltip(fields=['nama'], aliases=['Wilayah:'], style="font-weight: bold; font-size: 14px;")).add_to(m)

            folium.Marker(location=pekanbaru_coords, popup=popup_text, icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m)

            logo_b64 = get_image_base64("logo.png")
            logo_upi_b64 = get_image_base64("logo upi yptk.png")
            logo_img_tag = f'<img src="data:image/png;base64,{logo_b64}" style="height: 55px; background: white; padding: 4px; border-radius: 4px;" alt="Logo">' if logo_b64 else ''
            logo_upi_tag = f'<img src="data:image/png;base64,{logo_upi_b64}" style="width: 60px; height: auto;" alt="Logo UPI YPTK">' if logo_upi_b64 else ''
            
            raw_map_html = m.get_root().render()

            custom_css_and_layout_start = f"""
            <body style="background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
                <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 100%; max-width: 1450px; height: 95vh; display: flex; flex-direction: column;">
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #1f77b4 0%, #175a8a 100%); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(31,119,180,0.2);">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            {logo_img_tag}
                            <div>
                                <h2 style="margin: 0; font-size: 22px; font-weight: 600; letter-spacing: 0.5px;">Dashboard Prediksi Risiko Kebakaran Lahan</h2>
                                <p style="margin: 5px 0 0 0; font-size: 13px; color: #dceefb;">Integrasi Model Machine Learning, IoT, dan Spatial GIS</p>
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 13px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 10px 15px; border-radius: 8px;">
                            <b style="font-size: 14px; letter-spacing: 0.5px;">Domain/Wilayah:</b> Prov. RIAU - Kota Pekanbaru<br>
                            <span style="color: #e2f0ff;"><b>Valid/Berlaku:</b> {tanggal_valid}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 95px); overflow: hidden;">
                        
                        <div style="width: 280px; display: flex; flex-direction: column; gap: 15px; overflow-y: auto; padding-right: 5px; flex-shrink: 0;">
                            
                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9;">
                                <b style="font-size: 13px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 10px;">Status Prediksi Saat Ini</b>
                                <div style="font-size: 17px; font-weight: bold; color: {marker_color};">{risk_label}</div>
                            </div>

                            <div style="display: flex; gap: 10px;">
                                <div style="flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9; text-align:center;">
                                    <b style="font-size: 12px; color: #333; display: block; margin-bottom: 10px; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Arah Utara</b>
                                    <div style="font-weight: bold; font-size: 15px; color: #333; margin-bottom: 3px;">U</div>
                                    <div style="width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-bottom: 16px solid red; margin: 0 auto;"></div>
                                    <div style="width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 16px solid #555; margin: 0 auto;"></div>
                                </div>
                                <div style="flex: 1.5; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9;">
                                    <b style="font-size: 12px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 8px;">Legenda Risiko</b>
                                    <div style="font-size: 11px; line-height: 1.8;">
                                        <div style="display: flex; align-items: center;"><i style="background: blue; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;"></i> Rendah</div>
                                        <div style="display: flex; align-items: center;"><i style="background: green; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;"></i> Sedang</div>
                                        <div style="display: flex; align-items: center;"><i style="background: orange; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;"></i> Tinggi</div>
                                        <div style="display: flex; align-items: center;"><i style="background: red; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px;"></i> S. Tinggi</div>
                                    </div>
                                </div>
                            </div>

                            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9;">
                                <b style="font-size: 13px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 5px;">Tindak Lanjut Instansi</b>
                                <div style="font-size: 12px; line-height: 1.5;">{tl_html}</div>
                            </div>
                        </div>
                        
                        <div style="flex-grow: 1; border: 2px solid #e2e8f0; border-radius: 10px; overflow: hidden; position: relative;">
            """
            
            custom_layout_end = f"""
                        </div> 
                        
                        <div style="width: 340px; display: flex; flex-direction: column; gap: 18px; flex-shrink: 0; padding-left: 5px;">
                            <div style="flex-grow: 1; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; background: #ffffff;">
                                <b style="font-size: 14px; color: #2d3748; display: block; border-bottom: 2px solid #edf2f7; padding-bottom: 12px; margin-bottom: 15px; text-transform: uppercase;">Faktor Pemicu (XAI SHAP)</b>
                                <div style="font-size: 12px;">{xai_html}</div>
                            </div>

                            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px;">
                                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 18px;">
                                    {logo_upi_tag}
                                    <div style="text-align: left; line-height: 1.5;">
                                        <b style="font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Produced By</b><br>
                                        <span style="font-size: 13px; font-weight: bold; color: #2d3748;">Model HSEL Terintegrasi IoT</span><br>
                                        <span style="font-size: 11px; font-style: italic; color: #4a5568;">Mahasiswa Doctoral TI UPI YPTK Padang</span>
                                    </div>
                                </div>
                                <div style="border-top: 1px dashed #cbd5e0; padding-top: 15px; text-align: center;">
                                    <div style="font-size: 12px; color: #3182ce; font-weight: 600; margin-bottom: 8px;">Processed Date: <span style="color: #2b6cb0; font-weight: normal;">{tanggal_valid}</span></div>
                                    <div style="font-size: 11px; color: #718096;"><b style="color: #4a5568;">Data Source:</b> Sensor IoT Lokal, HSEL Prediction</div>
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
                🌐 Buka Dashboard Pemantauan Terpadu (Pekanbaru)
            </button>
            <script>
            function openMap() {{
                fetch(`data:text/html;base64,{b64_html}`).then(res => res.blob()).then(blob => {{ window.open(URL.createObjectURL(blob), '_blank'); }});
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
                folium.GeoJson(riau_geojson, style_function=lambda feature: {'fillColor': feature['properties']['warna_fill'], 'color': '#333333', 'weight': 2, 'fillOpacity': 0.7 if feature['properties']['warna_fill'] != "#9e9e9e" else 0.4}, tooltip=folium.GeoJsonTooltip(fields=['nama', 'tooltip_info'], aliases=['Kab/Kota:', 'Keterangan:'], style="font-weight: bold; font-size: 13px;")).add_to(m_regional)
                
                popup_pku_html = f"<div style='width: 230px; font-size: 13px; line-height: 1.5;'><b>Wilayah:</b> Kota Pekanbaru<br><b>Prediksi:</b> {risk_label_pku}<br><b>Suhu:</b> {float(last_num[fitur[0]]):.1f} °C<br><b>Kelembapan:</b> {float(last_num[fitur[1]]):.1f} %<br><b>Curah Hujan:</b> {float(last_num[fitur[2]]):.1f} mm<br><b>Kecepatan Angin:</b> {float(last_num[fitur[3]]):.1f} m/s<br><b>Kelembaban Tanah:</b> {float(last_num[fitur[4]]):.1f} %<br><b>Waktu:</b> {last_row['Waktu']}</div>"
                folium.Marker(location=[0.5333, 101.4500], popup=folium.Popup(popup_pku_html, max_width=250), icon=folium.Icon(color=marker_color_pku, icon="info-sign")).add_to(m_regional)

                def get_dummy_popup(nama_daerah): return f"<div style='width: 230px; font-size: 13px; line-height: 1.5;'><b>Wilayah:</b> {nama_daerah}<br><b>Prediksi:</b> Unknown / Menunggu Data<br><b>Suhu:</b> Unknown<br><b>Kelembapan:</b> Unknown<br><b>Curah Hujan:</b> Unknown<br><b>Kecepatan Angin:</b> Unknown<br><b>Kelembaban Tanah:</b> Unknown<br><b>Waktu:</b> -</div>"
                folium.Marker(location=[0.7490, 102.0460], popup=folium.Popup(get_dummy_popup("Kabupaten Siak"), max_width=250), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)
                folium.Marker(location=[0.2662, 101.6917], popup=folium.Popup(get_dummy_popup("Kabupaten Pelalawan"), max_width=250), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)
                folium.Marker(location=[1.4789, 102.1444], popup=folium.Popup(get_dummy_popup("Kabupaten Bengkalis"), max_width=250), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)

            raw_map_html = m_regional.get_root().render()

            custom_css_and_layout_start = f"""
            <body style="background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
                <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 100%; max-width: 1450px; height: 95vh; display: flex; flex-direction: column;">
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(30,60,114,0.2);">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            {logo_img_tag}
                            <div>
                                <h2 style="margin: 0; font-size: 20px; font-weight: 600; letter-spacing: 0.5px;">Pemantauan Regional (Pekanbaru, Siak, Pelalawan, Bengkalis)</h2>
                                <p style="margin: 5px 0 0 0; font-size: 13px; color: #d1e8ff;">Tahap Perluasan Integrasi Sensor IoT</p>
                            </div>
                        </div>
                        <div style="text-align: right; font-size: 13px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 8px 12px; border-radius: 8px;">
                            <b style="font-size: 14px; letter-spacing: 0.5px;">Domain:</b> Sebagian Wilayah Riau<br>
                            <span style="color: #e2f0ff;"><b>Valid/Berlaku:</b> {tanggal_valid}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 95px); overflow: hidden;">
                        
                        <div style="width: 330px; display: flex; flex-direction: column; gap: 18px; overflow-y: auto; padding-right: 5px;">
                            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px;">
                                <b style="font-size: 14px; color: #2d3748; display: block; border-bottom: 2px solid #edf2f7; padding-bottom: 12px; margin-bottom: 15px; text-transform: uppercase;">Legenda Status Kawasan</b>
                                <div style="font-size: 13px; color: #4a5568; line-height: 2.4;">
                                    <div style="display: flex; align-items: center;"><i style="background: blue; width: 14px; height: 14px; border-radius: 50%; margin-right: 12px;"></i> Risiko Rendah</div>
                                    <div style="display: flex; align-items: center;"><i style="background: green; width: 14px; height: 14px; border-radius: 50%; margin-right: 12px;"></i> Risiko Sedang</div>
                                    <div style="display: flex; align-items: center;"><i style="background: orange; width: 14px; height: 14px; border-radius: 50%; margin-right: 12px;"></i> Risiko Tinggi</div>
                                    <div style="display: flex; align-items: center;"><i style="background: red; width: 14px; height: 14px; border-radius: 50%; margin-right: 12px;"></i> Risiko Sangat Tinggi</div>
                                    <div style="height: 1px; background: #edf2f7; margin: 12px 0;"></div>
                                    <div style="display: flex; align-items: center;"><i style="background: #9e9e9e; width: 14px; height: 14px; border-radius: 50%; margin-right: 12px;"></i> <b style="color: #2d3748;">Menunggu Data IoT</b></div>
                                </div>
                            </div>

                            <div style="background: #fffaf0; border: 1px solid #feebc8; border-left: 5px solid #ed8936; border-radius: 8px; padding: 18px; color: #9c4221;">
                                <b style="font-size: 14px; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">💡 Catatan Sistem</b>
                                <span style="font-size: 13px; line-height: 1.6; display: block;">Pemantauan <i>real-time</i> aktif di Pekanbaru. Kab. Siak, Kab. Pelalawan, dan Kab. Bengkalis dalam tahap persiapan.</span>
                            </div>

                            <div style="flex-grow: 1;"></div>

                            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px;">
                                <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 18px;">
                                    {logo_upi_tag}
                                    <div style="text-align: left; line-height: 1.5;">
                                        <b style="font-size: 11px; color: #718096; text-transform: uppercase;">Produced By</b><br>
                                        <span style="font-size: 13px; font-weight: bold; color: #2d3748;">Model HSEL Terintegrasi IoT</span><br>
                                        <span style="font-size: 11px; font-style: italic; color: #4a5568;">Mahasiswa Doctoral TI UPI YPTK Padang</span>
                                    </div>
                                </div>
                                <div style="border-top: 1px dashed #cbd5e0; padding-top: 15px; text-align: center;">
                                    <div style="font-size: 12px; color: #3182ce; font-weight: 600; margin-bottom: 8px;">Processed Date: <span style="color: #2b6cb0; font-weight: normal;">{tanggal_valid}</span></div>
                                    <div style="font-size: 11px; color: #718096;"><b style="color: #4a5568;">Data Source:</b> Sensor IoT Lokal, HSEL Prediction</div>
                                </div>
                            </div>
                        </div>
                        
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
                fetch(`data:text/html;base64,{b64_html}`).then(res => res.blob()).then(blob => {{ window.open(URL.createObjectURL(blob), '_blank'); }});
            }}
            </script>
            """
            components.html(custom_button_html, height=60)

    # === BAGIAN UTAMA DASHBOARD =====================
    def main_dashboard():
        st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)
        
        df_raw = load_data()
        res = preprocess_sensor_data(df_raw)
        
        col_kiri, col_tengah, col_kanan = st.columns([1.2, 1.2, 1.2])
        
        with col_kiri:
            indikator_kiri_realtime()
            
        with col_tengah:
            st.markdown("<h5 style='text-align: center;'>Visualisasi Peta Lokasi Prediksi Kebakaran</h5>", unsafe_allow_html=True)
            peta_realtime_fragment()
            peta_regional_fragment()

            # === TOMBOL KE-3: MULTIMODAL NEW TAB ===
            st.markdown("""
            <a href="?page=multimodal" target="_blank" style="text-decoration: none;">
                <button style="width: 100%; padding: 12px 16px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #ffffff; border: none; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 14px; margin-top: 5px; font-weight: bold; transition: 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                    📸 Buka Dashboard Multimodal (Hybrid YOLO-ViT+GRU & IoT)
                </button>
            </a>
            """, unsafe_allow_html=True)

        with col_kanan:
            st.markdown("<h5 style='text-align: center;'>IoT Smart Fire Prediction</h5>", unsafe_allow_html=True)
            try:
                st.image(Image.open("forestiot4.jpg").resize((480, 360)))
            except Exception:
                st.info("Gambar 'forestiot4.jpg' tidak ditemukan di direktori.")
                    
        if res[0] is not None and not isinstance(res[0], str):
            df, clean_df, scaled_all, fitur = res

            st.markdown("<div class='section-title' style='margin-top: 25px;'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="scrollable-table" style="margin-bottom: 25px;">
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>
                    <tr>
                        <th style='background-color:#e0e0e0;'>Warna</th>
                        <th style='background-color:#e0e0e0;'>Tingkat Resiko / Intensitas</th>
                        <th style='background-color:#e0e0e0;'>Keterangan</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='background-color:blue; color:white;'>
                        <td>Blue</td><td>Low / Rendah</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran rendah. Intensitas api pada kategori rendah. Api mudah dikendalikan.</td>
                    </tr>
                    <tr style='background-color:green; color:white;'>
                        <td>Green</td><td>Moderate / Sedang</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sedang. Intensitas api pada kategori sedang. Api relatif masih cukup mudah dikendalikan.</td>
                    </tr>
                    <tr style='background-color:yellow; color:black;'>
                        <td>Yellow</td><td>High / Tinggi</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran tinggi. Intensitas api pada kategori tinggi. Api sulit dikendalikan.</td>
                    </tr>
                    <tr style='background-color:red; color:white;'>
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
                df_daily = df_chart[fitur].resample('D').mean().dropna().tail(15)

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

                tab_all, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Semua Data", "🌡️ Suhu Udara", "💧 Kelembapan Udara", "🌧️ Curah Hujan", "💨 Kecepatan Angin", "🌱 Kelembapan Tanah"])

                with tab_all:
                    df_melted = df_vis.melt(id_vars=['Waktu_DT'], var_name='Parameter Sensor', value_name='Nilai')
                    satuan_map = {'Suhu (°C)': '°C', 'Kelembapan (%)': '%', 'Curah Hujan (mm)': 'mm', 'Kecepatan Angin (m/s)': 'm/s', 'Kelembaban Tanah (%)': '%'}
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

                    points = chart_base.mark_circle(size=60, opacity=0.8).encode(opacity=alt.condition(selection, alt.value(1), alt.value(0.1)))
                    text_labels = chart_base.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11, fontWeight='bold').encode(text=alt.Text('LabelText:N'), opacity=alt.condition(selection, alt.value(1), alt.value(0)))

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

            st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            def to_excel(df_to_save: pd.DataFrame) -> bytes:
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df_to_save.to_excel(writer, index=False, sheet_name='Prediksi')
                writer.close()
                return output.getvalue()

            df_xlsx = to_excel(df)
            st.download_button(label="📥 Download Hasil Prediksi", data=df_xlsx, file_name="hasil_prediksi_kebakaran.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    main_dashboard()

    # === PENGUJIAN MANUAL & TEKS ===
    if "man_suhu" not in st.session_state: st.session_state.man_suhu = 30.0
    if "man_kel" not in st.session_state: st.session_state.man_kel = 65.0
    if "man_curah" not in st.session_state: st.session_state.man_curah = 10.0
    if "man_angin" not in st.session_state: st.session_state.man_angin = 3.0
    if "man_tanah" not in st.session_state: st.session_state.man_tanah = 50.0
    if "manual_result" not in st.session_state: st.session_state.manual_result = None

    def reset_manual():
        st.session_state.man_suhu, st.session_state.man_kel, st.session_state.man_curah, st.session_state.man_angin, st.session_state.man_tanah = 0.0, 0.0, 0.0, 0.0, 0.0
        st.session_state.manual_result = None

    def do_predict_manual():
        input_df = pd.DataFrame([{'Tavg: Temperatur rata-rata (°C)': st.session_state.man_suhu, 'RH_avg: Kelembapan rata-rata (%)': st.session_state.man_kel, 'RR: Curah hujan (mm)': st.session_state.man_curah, 'ff_avg: Kecepatan angin rata-rata (m/s)': st.session_state.man_angin, 'Kelembaban Permukaan Tanah': st.session_state.man_tanah}])
        scaled_manual = scaler.transform(input_df)
        st.session_state.manual_result = convert_to_label(model.predict(scaled_manual)[0])

    @st.fragment
    def manual_prediction_ui():
        st.markdown("<div class='section-title' style='margin-top: 30px;'>Pengujian Menggunakan Data Meteorologi Manual</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Suhu Udara (°C)", key="man_suhu")
            st.number_input("Kelembapan Udara (%)", key="man_kel")
        with col2:
            st.number_input("Curah Hujan (mm)", key="man_curah")
            st.number_input("Kecepatan Angin (m/s)", key="man_angin")
        with col3:
            st.number_input("Kelembaban Tanah (%)", key="man_tanah")

        btn_pred, btn_reset, _ = st.columns([1, 1, 8])
        if btn_pred.button("🔍 Prediksi Manual", use_container_width=True):
            do_predict_manual()
            st.rerun()
        if btn_reset.button("🧼 Reset Manual", use_container_width=True):
            reset_manual()
            st.rerun()

        if st.session_state.manual_result:
            hasil = st.session_state.manual_result
            font, bg = risk_styles.get(hasil, ("black", "white"))
            st.markdown(f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px; margin-top:15px;'>Prediksi Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True)

    if "txt_result" not in st.session_state: st.session_state.txt_result = None
    if "txt_preprocessing" not in st.session_state: st.session_state.txt_preprocessing = {}

    def reset_text():
        st.session_state.txt_result, st.session_state.txt_preprocessing = None, {}

    def do_predict_text(input_text):
        if input_text.strip() == "": st.warning("Harap masukkan deskripsi teks.")
        elif vectorizer is None or model_text is None: st.error("Model teks gagal dimuat. Pastikan file 'tfidf_vectorizer.joblib' dan 'stacking_text_model.joblib' tersedia di direktori.")
        else:
            try:
                raw_text = input_text
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
                st.session_state.txt_preprocessing = {"raw": raw_text, "case_folding": text_lower, "cleansing": text_clean, "stopword": text_stopword, "tokenizing": token_display, "stemming": text_stemmed, "tfidf_df": df_tfidf, "prob_dict": prob_dict}
                st.session_state.txt_result = convert_to_label(pred)
            except Exception as e: st.error(f"Terjadi kesalahan saat memproses input teks: {e}")

    @st.fragment
    def text_prediction_ui():
        st.markdown("<div class='section-title' style='margin-top: 20px;'>Pengujian Menggunakan Data Teks</div>", unsafe_allow_html=True)
        
        current_text = st.text_area("Masukkan deskripsi lingkungan (misal: Cuaca hari ini sangat panas dan kering tanpa hujan):", value=st.session_state.get("txt_input_val", ""), height=120)
        
        btn_pred_text, btn_reset_text, _ = st.columns([1, 1, 8])
        if btn_pred_text.button("🔍 Prediksi Teks", use_container_width=True):
            st.session_state.txt_input_val = current_text
            do_predict_text(current_text)
            st.rerun()
            
        if btn_reset_text.button("🧼 Reset Teks", use_container_width=True):
            st.session_state.txt_input_val = ""
            reset_text()
            st.rerun()
            
        if st.session_state.txt_result:
            with st.expander("🛠️ Klik untuk melihat hasil setiap tahapan Pre-processing & Keputusan Model", expanded=False):
                steps = st.session_state.txt_preprocessing
                if steps:
                    st.markdown("**1. Original Text**"); st.info(steps.get("raw", "-"))
                    st.markdown("**2. Cleansing**"); st.info(steps.get("cleansing", "-"))
                    st.markdown("**3. Stopword**"); st.info(steps.get("stopword", "-"))
                    st.markdown("**4. Tokenization**"); st.info(steps.get("tokenizing", "[]"))
                    st.markdown("**5. Stemming**"); st.info(steps.get("stemming", "-"))
                    st.markdown("**6. Ekstraksi Fitur (TF-IDF)**")
                    df_tfidf_display = steps.get("tfidf_df")
                    if df_tfidf_display is not None and not df_tfidf_display.empty: st.dataframe(df_tfidf_display, use_container_width=True)
                    else: st.warning("Kata-kata pada input ini tidak dikenali dalam model vocabulary.")

                    st.markdown("**7. Probabilitas Model HSEL**")
                    prob_dict = steps.get("prob_dict")
                    if prob_dict:
                        for label, prob in prob_dict.items():
                            st.markdown(f"**{label}** ({prob*100:.1f}%)")
                            st.progress(float(prob))

            hasil = st.session_state.txt_result
            font, bg = risk_styles.get(hasil, ("black", "white"))
            st.markdown(f"<p style='color:{font}; background-color:{bg}; padding:15px; border-radius:8px; margin-top: 15px; font-size: 16px; font-weight:bold; text-align:center;'>Hasil Prediksi Tingkat Risiko Kebakaran: <span style='text-decoration:underline;'>{hasil}</span></p>", unsafe_allow_html=True)

    manual_prediction_ui()
    text_prediction_ui()

    # === FOOTER ===
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top: 20px; background-color: black; padding: 10px 20px; border-radius: 10px; text-align: center; color: white;'>
        <p style='margin: 0; font-size: 30px; font-weight: bold; line-height: 1.2;'>Smart Fire Prediction HSEL Model</p>
        <p style='margin: 0; font-size: 13px; line-height: 1.2;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026</p>
    </div>
    """, unsafe_allow_html=True)
