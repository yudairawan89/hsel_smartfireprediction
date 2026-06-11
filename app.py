import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from streamlit_folium import folium_static
import folium
from folium.plugins import MousePosition, Fullscreen
from PIL import Image
import re
import altair as alt

# === TAMBAHAN LIBRARY UNTUK BACA GEOJSON & KOMPONEN PETA ===
import json

# === TAMBAHAN LIBRARY UNTUK XAI ===
import shap
import matplotlib.pyplot as plt

# === TAMBAHAN LIBRARY SASTRAWI ===
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# === TAMBAHAN LIBRARY UNTUK LOAD DATA & WEB KOMPONEN ===
import requests
import time
from io import StringIO, BytesIO
import base64
import streamlit.components.v1 as components

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Fire Prediction HSEL", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# === STYLE KUSTOM (Digabungkan dengan Style Profesional) ===
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
        margin-top: 30px;
    }
    .scrollable-table { overflow-x: auto; }
    
    /* Frame khusus untuk iframe Folium agar terlihat seperti Peta GIS Statis di Streamlit */
    iframe[title="folium_static"] {
        border: 4px solid #444 !important;
        border-radius: 4px !important;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.2) !important;
    }

    /* --- STYLING DASHBOARD PROFESIONAL BARU --- */
    .main-header {
        background-color: #1f77b4;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 25px;
    }
    .main-header h1 { margin: 0; font-size: 24px; font-weight: bold; }
    .main-header p { margin: 5px 0 0 0; font-size: 14px; font-weight: normal; color: #dceefb; }

    /* Styling Kotak Sidebar Panels */
    [data-testid="stSidebar"] { padding-top: 1rem; }
    .panel-frame {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    .panel-header {
        font-size: 15px;
        font-weight: bold;
        color: #333;
        display: block;
        border-bottom: 1px solid #ccc;
        padding-bottom: 8px;
        margin-bottom: 10px;
    }

    /* Lingkaran Legenda Risiko di Sidebar */
    .risk-dot { height: 12px; width: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
    .risk-rendah { background-color: blue; }
    .risk-sedang { background-color: green; }
    .risk-tinggi { background-color: orange; }
    .risk-sangattinggi { background-color: red; }

    /* Indikator SHAP/XAI Warna di Sidebar */
    .shap-factor-row { display: flex; align-items: flex-start; margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    .shap-dot-small { height: 10px; width: 10px; border-radius: 50%; margin-top: 4px; margin-right: 8px; flex-shrink: 0; }
    .shap-good { background-color: #21c354; } /* Hijau (Meredam risiko) */
    .shap-bad { background-color: #ff4b4b; }  /* Merah (Memicu risiko) */
    .shap-factor-details p { margin: 0; font-size: 13px; line-height: 1.4; }
    .shap-factor-title { font-weight: bold; color: #333; }
    .shap-factor-contribution { font-weight: normal; color: #555; }
    .shap-factor-desc { color: #777; font-size: 12px; font-style: italic; }

    /* Teks Metadata Kecil */
    .metadata-text { font-size: 12px; color: #666; margin-top: 10px; }
    .produced-by-block { border-top: 2px solid #333; margin-top: 20px; padding-top: 15px; font-family: sans-serif; }
    .produced-by-header { font-weight: bold; font-size: 14px; text-transform: uppercase; margin-bottom: 5px; }
    .produced-by-item { margin: 0 0 5px 0; font-size: 13px; line-height: 1.4; }
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

# === LOAD MODEL, SCALER, SASTRAWI, DAN TEKS MODEL SECARA GLOBAL ===
@st.cache_resource
def load_model():
    return joblib.load("HSEL_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

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

# Eksekusi instansiasi model
model = load_model()
scaler = load_scaler()
stopword_remover, stemmer = load_sastrawi()

try:
    vectorizer, model_text = load_text_models()
except Exception:
    vectorizer, model_text = None, None

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
        else:
            return None
    except Exception:
        return None

def preprocess_sensor_data(df):
    if df is None or df.empty:
        return None, None, None, None

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
                found = cand
                break
        if found is not None:
            actual_rename[found] = target_name

    df = df.rename(columns=actual_rename)
    fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)', 'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Permukaan Tanah']
    
    missing = [c for c in fitur + ['Waktu'] if c not in df.columns]
    if missing: return "error", missing, None, None

    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = clean_df[col].astype(str).str.replace(',', '.', regex=False).astype(float).fillna(0)
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaled_all = scaler.transform(clean_df)
    predictions = [convert_to_label(p) for p in model.predict(scaled_all)]
    df["Prediksi Kebakaran"] = predictions

    return df, clean_df, scaled_all, fitur


# =========================================================================
# === BAGIAN DASHBOARD REALTIME (TERPUSAT DALAM SATU FRAGMENT) ============
# =========================================================================
@st.fragment(run_every=7)
def realtime_top_dashboard():
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
    risk_label = last_row["Prediksi Kebakaran"]
    waktu_valid = str(last_row['Waktu']) # Menarik Waktu Asli dari Sensor

    # --- HITUNG SHAP (XAI) SEKALI UNTUK FRAME & SIDEBAR ---
    data_realtime_scaled = pd.DataFrame(scaled_all[-1:], columns=fitur)
    background_data = pd.DataFrame(shap.sample(scaled_all, 50), columns=fitur)
    explainer = shap.Explainer(model.predict, background_data)
    shap_values = explainer(data_realtime_scaled)
    
    total_abs_shap = sum(abs(v) for v in shap_values[0].values)
    kontribusi = []
    for nama_f, shap_v in zip(fitur, shap_values[0].values):
        pct_f = (abs(float(shap_v)) / total_abs_shap) * 100 if total_abs_shap > 0 else 0.0
        kontribusi.append({"fitur": nama_f, "shap_val": float(shap_v), "pct": pct_f})
    kontribusi = sorted(kontribusi, key=lambda x: x["pct"], reverse=True)

    # ================= FRAME 1: SIDEBAR =================
    with st.sidebar:
        # Logo dan Header Sidebar
        logo_col_left, logo_col_right = st.columns([1, 4])
        try:
            with logo_col_left:
                st.image("logo.png", width=60)
        except Exception:
            pass 
        with logo_col_right:
            st.markdown(f"**Smart Fire Prediction**\nKota Pekanbaru\n\n*HSEL IoT Model*")
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        # Panel Legenda
        st.markdown("<div class='panel-frame'>", unsafe_allow_html=True)
        st.markdown("<span class='panel-header'>Arah Utara & Legenda</span>", unsafe_allow_html=True)
        col_north, col_legend = st.columns([1, 1.5])
        with col_north:
            st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Comics_north_arrow.svg", caption="U", width=60)
        with col_legend:
            st.markdown(f'<span class="risk-dot risk-rendah"></span> Rendah<br>'
                        f'<span class="risk-dot risk-sedang"></span> Sedang<br>'
                        f'<span class="risk-dot risk-tinggi"></span> Tinggi<br>'
                        f'<span class="risk-dot risk-sangattinggi"></span> Sangat Tinggi', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Panel XAI SHAP Lengkap
        st.markdown("<div class='panel-frame'>", unsafe_allow_html=True)
        st.markdown("<span class='panel-header'>Faktor Pemicu (XAI SHAP)</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size:12px; color:gray; display:block; margin-bottom:10px;'>Grafik kontribusi parameter terhadap status risiko: <b>{risk_label.split('/')[0].strip()}</b></span>", unsafe_allow_html=True)
        
        for factor in kontribusi:
            nama_fitur = str(factor['fitur']).lower()
            persen = factor['pct']
            arah = factor['shap_val']
            
            # Tentukan Deskripsi dan Warna Berdasarkan Arah (Positif=Bad, Negatif=Good)
            if arah > 0:
                indicator_class = "shap-bad"
                if "tanah" in nama_fitur: desc = "Kelembaban tanah yang rendah menunjukkan kondisi lahan yang teramat kering."
                elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur: desc = "Udara yang kering mempercepat proses pengeringan bahan bakar."
                elif "angin" in nama_fitur or "ff" in nama_fitur: desc = "Kecepatan angin saat ini berisiko memperluas area kebakaran."
                elif "suhu" in nama_fitur or "temperatur" in nama_fitur: desc = "Suhu panas memicu penguapan air dari vegetasi."
                elif "hujan" in nama_fitur or "rr" in nama_fitur: desc = "Ketiadaan curah hujan menghilangkan faktor pendingin alami."
                else: desc = "Meningkatkan potensi risiko kebakaran."
            else:
                indicator_class = "shap-good"
                if "tanah" in nama_fitur: desc = "Kelembaban tanah terdeteksi basah, bertindak sebagai tameng alami."
                elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur: desc = "Kelembapan udara tinggi menjaga kebasahan partikel."
                elif "angin" in nama_fitur or "ff" in nama_fitur: desc = "Pergerakan angin relatif lambat dan tidak mengancam."
                elif "suhu" in nama_fitur or "temperatur" in nama_fitur: desc = "Suhu udara sejuk menjaga stabilitas termal."
                elif "hujan" in nama_fitur or "rr" in nama_fitur: desc = "Curah hujan yang turun merupakan faktor pendingin krusial."
                else: desc = "Menstabilkan potensi risiko kebakaran."

            simbol_arah = "+" if arah > 0 else "-"
            st.markdown(f"""
                <div class="shap-factor-row">
                    <div class="shap-dot-small {indicator_class}"></div>
                    <div class="shap-factor-details">
                        <p><span class="shap-factor-title">{factor['fitur']}</span> <span class="shap-factor-contribution">({simbol_arah}{persen:.1f}%)</span></p>
                        <p class="shap-factor-desc">{desc}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Panel Tindak Lanjut
        st.markdown("<div class='panel-frame'>", unsafe_allow_html=True)
        st.markdown("<span class='panel-header'>Tindak Lanjut Instansi</span>", unsafe_allow_html=True)
        if risk_label == "Low / Rendah":
            st.markdown("<ul style='padding-left: 20px; font-size:13px;'><li>Monitoring rutin kondisi lingkungan</li><li>Patroli berkala ringan</li><li>Edukasi preventif kepada masyarakat</li><li>Dokumentasi dan pelaporan kondisi normal</li></ul>", unsafe_allow_html=True)
        elif risk_label == "Moderate / Sedang":
            st.markdown("<ul style='padding-left: 20px; font-size:13px;'><li>Peningkatan frekuensi patroli</li><li>Penyampaian peringatan dini terbatas</li><li>Koordinasi internal BPBD dan aparat desa</li><li>Pengawasan aktivitas pembakaran terbuka</li></ul>", unsafe_allow_html=True)
        elif risk_label == "High / Tinggi":
            st.markdown("<ul style='padding-left: 20px; font-size:13px;'><li>Aktivasi pos siaga tingkat lokal</li><li>Penempatan personel siaga di titik rawan</li><li>Koordinasi dengan TNI/Polri dan Manggala Agni</li><li>Peringatan dini terbuka kepada masyarakat</li><li>Penyiapan peralatan pemadaman awal</li></ul>", unsafe_allow_html=True)
        else:
            st.markdown("<ul style='padding-left: 20px; font-size:13px;'><li>Penetapan status siaga darurat tingkat lokal</li><li>Aktivasi penuh posko tanggap darurat</li><li>Mobilisasi tim pemantauan dan pemadam</li><li>Koordinasi lintas sektor</li><li>Penyebaran peringatan dini</li><li>Pengetatan larangan pembakaran terbuka</li></ul>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    
    # ================= HEADER & KOLOM UTAMA =================
    st.markdown(f"""
    <div class="main-header">
        <h1>Dashboard Prediksi Risiko Kebakaran Lahan</h1>
        <p>Wilayah Administratif Kota Pekanbaru | Observasi Terbaru: {waktu_valid}</p>
    </div>
    """, unsafe_allow_html=True)

    body_col_main, body_col_produced = st.columns([4, 1.2])

    # ================= FRAME 2: PETA INTERAKTIF =================
    with body_col_main:
        st.markdown("<div style='text-align: center; margin-bottom: 10px; font-weight: bold;'>Visualisasi Peta Lokasi Prediksi Kebakaran</div>", unsafe_allow_html=True)
        
        pekanbaru_coords = [0.5333, 101.4500] 
        color_map = {"Low / Rendah": "blue", "Moderate / Sedang": "green", "High / Tinggi": "orange", "Very High / Sangat Tinggi": "red"}
        marker_color = color_map.get(risk_label, "gray")

        m = folium.Map(location=pekanbaru_coords, zoom_start=9.5, control_scale=True, tiles='OpenStreetMap')

        formatter = "function(num) {return L.Util.formatNum(num, 5) + ' &deg;';};"
        MousePosition(position="bottomleft", separator=" | ", empty_string="Koordinat tidak tersedia", lng_first=True, num_digits=20, prefix="Posisi:", lat_formatter=formatter, lng_formatter=formatter).add_to(m)
        Fullscreen(position='topright').add_to(m)

        try:
            with open("Provinsi Riau-KAB_KOTA.geojson", "r") as f:
                riau_geojson = json.load(f)
            pekanbaru_feature = None
            for feature in riau_geojson['features']:
                nama_wilayah = feature['properties'].get('nama', '').lower()
                kab_kota = feature['properties'].get('kab_kota', '').lower()
                if 'pekanbaru' in nama_wilayah or 'pekanbaru' in kab_kota:
                    pekanbaru_feature = feature
                    break
            pekanbaru_geojson = {"type": "FeatureCollection", "features": [pekanbaru_feature] if pekanbaru_feature else []}
            if pekanbaru_geojson["features"]:
                folium.GeoJson(
                    pekanbaru_geojson,
                    style_function=lambda feature, color=marker_color: {'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.4},
                    tooltip=folium.GeoJsonTooltip(fields=['nama'], aliases=['Wilayah:'], style="font-weight: bold; font-size: 14px;")
                ).add_to(m)
        except Exception:
            pass

        popup_text = folium.Popup(f"""
            <div style='width: 230px; font-size: 13px; line-height: 1.5;'>
            <b>Wilayah:</b> Kota Pekanbaru<br>
            <b>Prediksi:</b> {risk_label}<br>
            <b>Suhu:</b> {float(last_num[fitur[0]]):.1f} °C<br>
            <b>Kelembapan:</b> {float(last_num[fitur[1]]):.1f} %<br>
            <b>Curah Hujan:</b> {float(last_num[fitur[2]]):.1f} mm<br>
            <b>Kecepatan Angin:</b> {float(last_num[fitur[3]]):.1f} m/s<br>
            <b>Kelembaban Tanah:</b> {float(last_num[fitur[4]]):.1f} %<br>
            <b>Waktu:</b> {waktu_valid}
            </div>
        """, max_width=250)

        folium.Marker(location=pekanbaru_coords, popup=popup_text, icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m)

        raw_map_html = m.get_root().render()
        folium_static(m, width=950, height=450)

        # Layout HTML Spesifik untuk Download Offline
        custom_css_and_layout_start = f"""
        <body style="background-color: #eef2f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 25px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
            <div style="background-color: white; padding: 25px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.15); width: 100%; max-width: 1300px; height: 90vh; display: flex; flex-direction: column;">
                <div style="background-color: #1f77b4; color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px; flex-shrink: 0;">
                    <h2 style="margin: 0; font-size: 22px;">Dashboard Prediksi Risiko Kebakaran Lahan (Offline)</h2>
                    <p style="margin: 5px 0 0 0; font-size: 14px; font-weight: normal; color: #dceefb;">Wilayah Administratif Kota Pekanbaru</p>
                </div>
                <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 90px); overflow: hidden;">
                    <div style="width: 300px; display: flex; flex-direction: column; gap: 15px; overflow-y: auto; padding-right: 5px; flex-shrink: 0;">
                        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9;">
                            <b style="font-size: 14px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 10px;">Status Prediksi Terakhir</b>
                            <div style="font-size: 18px; font-weight: bold; color: {marker_color};">{risk_label}</div>
                            <div style="font-size: 11px; margin-top:5px;">Valid: {waktu_valid}</div>
                        </div>
                    </div>
                    <div style="flex-grow: 1; border: 3px solid #555; border-radius: 8px; overflow: hidden; position: relative; box-shadow: inset 0 0 10px rgba(0,0,0,0.1);">
                        """
        custom_layout_end = "</div> </div> </div> </body>"
        framed_dashboard_html = raw_map_html.replace('<body>', custom_css_and_layout_start).replace('</body>', custom_layout_end)

        # Tombol Buka Map Tab Baru
        b64_html = base64.b64encode(framed_dashboard_html.encode('utf-8')).decode('utf-8')
        custom_button_html = f"""
        <button onclick="openMap()" style="width: 100%; padding: 8px 16px; background-color: #ffffff; color: #333; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 14px; transition: 0.3s;" onmouseover="this.style.borderColor='#1f77b4'; this.style.color='#1f77b4'" onmouseout="this.style.borderColor='#ccc'; this.style.color='#333'">
            🌐 Buka Peta Interaktif (Tab Baru)
        </button>
        <script>
        function openMap() {{
            const b64Data = "{b64_html}";
            const byteCharacters = atob(b64Data);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {{ byteNumbers[i] = byteCharacters.charCodeAt(i); }}
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], {{type: 'text/html;charset=utf-8'}});
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank');
        }}
        </script>
        """
        components.html(custom_button_html, height=50)

    # ================= FRAME 3: IoT & METADATA =================
    with body_col_produced:
        st.markdown("<div style='text-align: center; margin-bottom: 10px; font-weight: bold;'>IoT Smart Fire Prediction</div>", unsafe_allow_html=True)
        try:
            image = Image.open("forestiot4.jpg")
            st.image(image, use_column_width=True)
        except Exception:
            st.info("Gambar 'forestiot4.jpg' tidak ditemukan.")

        st.markdown("<div class='produced-by-block'>", unsafe_allow_html=True)
        st.markdown(f"**Domain/Wilayah:** Kota Pekanbaru\n\n**Valid/Berlaku:** {waktu_valid}")
        st.markdown(f"<p class='metadata-text' style='margin-bottom:15px;'>(Observation time ditarik dari timestamp sensor IoT terbaru)</p>", unsafe_allow_html=True)

        st.markdown("<p class='produced-by-header'>Produced By / Diproduksi Oleh:</p>", unsafe_allow_html=True)
        st.markdown("""
            <p class='produced-by-item'>● Model HSEL Terintegrasi IoT</p>
            <p class='produced-by-item'>● Mahasiswa Doctoral Teknologi Informasi</p>
            <p class='produced-by-item'>● Universitas Putra Indonesia YPTK Padang</p>
        """, unsafe_allow_html=True)
        st.markdown(f"<p class='metadata-text'>Processed Date: {datetime.utcnow().strftime('%d/%m/%Y')}</p>", unsafe_allow_html=True)
        st.markdown("<p class='metadata-text'>© BMKG & Team Peneliti UPI YPTK, 2026</p></div>", unsafe_allow_html=True)


# =========================================================================
# === BAGIAN UTAMA APLIKASI ===============================================
# =========================================================================
def main_dashboard():
    # Render UI Dashboard Realtime di atas
    realtime_top_dashboard()
    
    # Load Data untuk Chart & Tabel bagian bawah
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    
    if res[0] is not None and not isinstance(res[0], str):
        df, clean_df, scaled_all, fitur = res

        st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="scrollable-table" style="margin-bottom: 25px; margin-top:15px;">
        <table style='width: 100%; border-collapse: collapse;'>
            <thead>
                <tr>
                    <th style='background-color:#e0e0e0;'>Warna</th>
                    <th style='background-color:#e0e0e0;'>Tingkat Resiko / Intensitas</th>
                    <th style='background-color:#e0e0e0;'>Keterangan</th>
                </tr>
            </thead>
            <tbody>
                <tr style='background-color:blue; color:white;'><td>Blue</td><td>Low / Rendah</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran rendah. Intensitas api pada kategori rendah. Api mudah dikendalikan.</td></tr>
                <tr style='background-color:green; color:white;'><td>Green</td><td>Moderate / Sedang</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sedang. Intensitas api pada kategori sedang. Api relatif masih cukup mudah dikendalikan.</td></tr>
                <tr style='background-color:yellow; color:black;'><td>Yellow</td><td>High / Tinggi</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran tinggi. Intensitas api pada kategori tinggi. Api sulit dikendalikan.</td></tr>
                <tr style='background-color:red; color:white;'><td>Red</td><td>Very High / Sangat Tinggi</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran sangat tinggi. Intensitas api pada kategori sangat tinggi. Api sangat sulit dikendalikan.</td></tr>
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
                text_labels = chart_base.mark_text(align='center', baseline='bottom', dy=-10, fontSize=11, fontWeight='bold').encode(
                    text=alt.Text('LabelText:N'), opacity=alt.condition(selection, alt.value(1), alt.value(0))
                )

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
                st.altair_chart((bar + line + point).properties(height=350).interactive(), use_container_width=True)
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
        st.download_button(
            label="📥 Download Hasil Prediksi Kebakaran sebagai XLSX",
            data=df_xlsx, file_name="hasil_prediksi_kebakaran.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

main_dashboard()


# =========================================================================
# === BAGIAN PENGUJIAN MANUAL & TEKS DENGAN FRAGMENT KHUSUS ===============
# =========================================================================
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
    st.markdown("<div class='section-title'>Pengujian Menggunakan Data Meteorologi Manual</div>", unsafe_allow_html=True)
    
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
    with btn_pred:
        st.button("🔍 Prediksi Manual", on_click=do_predict_manual)
    with btn_reset:
        st.button("🧼 Reset Manual", on_click=reset_manual)

    if st.session_state.manual_result:
        hasil = st.session_state.manual_result
        font, bg = risk_styles.get(hasil, ("black", "white"))
        st.markdown(f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px; margin-top:15px;'>Prediksi Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True)

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
            except Exception: pass

            pred = model_text.predict(X_trans)[0]
            st.session_state.txt_preprocessing = {"raw": raw_text, "case_folding": text_lower, "cleansing": text_clean, "stopword": text_stopword, "tokenizing": token_display, "stemming": text_stemmed, "tfidf_df": df_tfidf, "prob_dict": prob_dict}
            st.session_state.txt_result = convert_to_label(pred)
        except Exception as e:
            st.error(f"Kesalahan memproses input teks: {e}")

@st.fragment
def text_prediction_ui():
    st.markdown("<div class='section-title'>Pengujian Menggunakan Data Teks</div>", unsafe_allow_html=True)
    st.text_area("Masukkan deskripsi lingkungan:", key="txt_input", height=120)

    btn_pred_text, btn_reset_text, _ = st.columns([1, 1, 8])
    with btn_pred_text:
        st.button("🔍 Prediksi Teks", on_click=do_predict_text)
    with btn_reset_text:
        st.button("🧼 Reset Teks", on_click=reset_text)
        
    if st.session_state.txt_result:
        with st.expander("🛠️ Klik untuk melihat hasil setiap tahapan Pre-processing & Keputusan Model", expanded=False):
            steps = st.session_state.txt_preprocessing
            if steps:
                st.markdown("**1. Original Text**"); st.info(steps.get("raw", "-"))
                st.markdown("**2. Case Folding**"); st.info(steps.get("case_folding", "-"))
                st.markdown("**3. Cleansing**"); st.info(steps.get("cleansing", "-"))
                st.markdown("**4. Stopword**"); st.info(steps.get("stopword", "-"))
                st.markdown("**5. Tokenization**"); st.info(steps.get("tokenizing", "[]"))
                st.markdown("**6. Stemming**"); st.info(steps.get("stemming", "-"))
                st.markdown("**7. Ekstraksi Fitur (TF-IDF)**")
                if steps.get("tfidf_df") is not None and not steps.get("tfidf_df").empty: st.dataframe(steps.get("tfidf_df"), use_container_width=True)
                else: st.warning("Kata-kata pada input tidak dikenali.")

                st.markdown("**8. Analisis Keputusan Model (Probabilitas)**")
                if steps.get("prob_dict"):
                    for label, prob in steps.get("prob_dict").items():
                        st.markdown(f"**{label}** ({prob*100:.1f}%)"); st.progress(float(prob))

        hasil = st.session_state.txt_result
        font, bg = risk_styles.get(hasil, ("black", "white"))
        st.markdown(f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px; margin-top: 15px; font-size: 16px;'>Hasil Prediksi Tingkat Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True)

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
