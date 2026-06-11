import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from streamlit_folium import folium_static
import folium
from PIL import Image, ImageDraw
import re
import altair as alt

# === TAMBAHAN LIBRARY UNTUK XAI ===
import shap
import matplotlib.pyplot as plt

# === TAMBAHAN LIBRARY SASTRAWI ===
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === TAMBAHAN LIBRARY UNTUK LOAD DATA ===
import requests
import time
from io import StringIO, BytesIO

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Fire Prediction HSEL", page_icon="favicon.ico", layout="wide")

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

# === FUNGSI EKSPOR GAMBAR ===
def create_report_image(risk_label, last_row, last_num, fitur):
    img = Image.new('RGB', (600, 450), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((50, 30), "LAPORAN PREDIKSI KEBAKARAN HSEL", fill=(0, 0, 0))
    d.text((50, 60), f"Tanggal: {last_row['Waktu']}", fill=(100, 100, 100))
    d.text((50, 100), f"STATUS RISIKO: {risk_label}", fill=(200, 0, 0))
    y = 150
    for f in fitur:
        val = last_num[f]
        d.text((50, y), f"- {f}: {val:.2f}", fill=(0, 0, 0))
        y += 30
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# === LOAD MODEL, SCALER, DAN SASTRAWI ===
@st.cache_resource
def load_model(): return joblib.load("HSEL_IoT_Model.joblib")
@st.cache_resource
def load_scaler(): return joblib.load("scaler.joblib")
@st.cache_resource
def load_sastrawi():
    return StopWordRemoverFactory().create_stop_word_remover(), StemmerFactory().create_stemmer()

model = load_model()
scaler = load_scaler()
stopword_remover, stemmer = load_sastrawi()

SHEET_ID = "1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_EDIT_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
SHEET_CSV_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def load_data():
    try:
        url_no_cache = f"{SHEET_CSV_LINK}&t={int(time.time())}"
        response = requests.get(url_no_cache, timeout=5)
        return pd.read_csv(StringIO(response.text)) if response.status_code == 200 else None
    except: return None

def preprocess_sensor_data(df):
    if df is None or df.empty: return None, None, None, None
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        'Waktu': ['Waktu', 'Timestamp'],
        'Tavg: Temperatur rata-rata (°C)': ['Suhu', 'Suhu Udara'],
        'RH_avg: Kelembapan rata-rata (%)': ['Kelembapan', 'RH (%)'],
        'RR: Curah hujan (mm)': ['Curah Hujan', 'RR'],
        'ff_avg: Kecepatan angin rata-rata (m/s)': ['Kecepatan Angin', 'Angin'],
        'Kelembaban Permukaan Tanah': ['Kelembapan Tanah', 'Soil Moisture']
    }
    
    actual_rename = {}
    for target, candidates in rename_map.items():
        for cand in candidates:
            if cand in df.columns: actual_rename[cand] = target
    
    df = df.rename(columns=actual_rename)
    fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)', 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Permukaan Tanah']
    
    if any(c not in df.columns for c in fitur + ['Waktu']): return "error", [], None, None
    
    clean_df = df[fitur].apply(lambda x: x.astype(str).str.replace(',', '.').astype(float)).fillna(0)
    scaled_all = scaler.transform(clean_df)
    df["Prediksi Kebakaran"] = [convert_to_label(p) for p in model.predict(scaled_all)]
    return df, clean_df, scaled_all, fitur

# === UI ===
st.header("Smart Fire Prediction HSEL Model")

@st.fragment(run_every=7)
def indikator_kiri_realtime():
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    if res[0] is None: return
    if isinstance(res[0], str) and res[0] == "error": return
        
    df, clean_df, scaled_all, fitur = res
    risk_label = df.iloc[-1]["Prediksi Kebakaran"]
    font, bg = risk_styles.get(risk_label, ("black", "white"))

    st.markdown(f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; text-align:center;'>Risiko: {risk_label}</p>", unsafe_allow_html=True)

def main_dashboard():
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1: indikator_kiri_realtime()
        
    if res[0] is not None and not isinstance(res[0], str):
        df, clean_df, scaled_all, fitur = res
        
        with col2:
            upi_yptk_coords = [-0.8953, 100.3957]
            m = folium.Map(location=upi_yptk_coords, zoom_start=15)
            folium.Marker(upi_yptk_coords, popup="UPI YPTK Padang", icon=folium.Icon(color="red")).add_to(m)
            folium_static(m, width=450, height=350)
            
        with col3:
            st.markdown("### Ekspor Laporan")
            report_png = create_report_image(df.iloc[-1]["Prediksi Kebakaran"], df.iloc[-1], clean_df.iloc[-1], fitur)
            st.download_button("📸 Download Laporan (PNG)", report_png, "laporan.png", "image/png")

        # Visualisasi Tren Premium
        df_chart = clean_df.copy()
        df_chart['Waktu_DT'] = pd.to_datetime(df['Waktu'], errors='coerce')
        df_melted = df_chart.melt(id_vars=['Waktu_DT'], var_name='Parameter', value_name='Nilai')
        
        selection = alt.selection_point(fields=['Parameter'], bind='legend')
        
        base = alt.Chart(df_melted).encode(
            x='Waktu_DT:T', y='Nilai:Q', color='Parameter:N',
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
        )
        
        chart = (base.mark_line(interpolate='monotone', strokeWidth=3) + 
                 base.mark_circle(size=60) + 
                 base.mark_text(dy=-10, fontWeight='bold').encode(text=alt.Text('Nilai:Q', format='.1f'))
                ).add_params(selection).properties(height=400).interactive()
        
        st.altair_chart(chart, use_container_width=True)

main_dashboard()
