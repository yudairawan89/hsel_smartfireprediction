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
    
    /* Frame khusus untuk iframe Folium agar terlihat seperti Peta GIS Statis di Streamlit */
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

# === LOAD MODEL, SCALER, SASTRAWI ===
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

# Penanganan Error Model Teks
try:
    vectorizer, model_text = load_text_models()
except Exception:
    vectorizer, model_text = None, None

# === LOAD GEOJSON DENGAN CACHE (Mencegah Lag) ===
@st.cache_data
def load_riau_geojson():
    try:
        with open("Provinsi Riau-KAB_KOTA.geojson", "r") as f:
            return json.load(f)
    except Exception as e:
        return None

# === KONFIG GOOGLE SHEET ===
SHEET_ID = "1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_EDIT_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
SHEET_CSV_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# === LOAD DATA TANPA CACHE DAN ANTI-CRASH ===
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

# === FUNGSI PEMBANTU PRE-PROCESSING SENSOR ===
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

    fitur = [
        'Tavg: Temperatur rata-rata (°C)',
        'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)',
        'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
    ]

    missing = [c for c in fitur + ['Waktu'] if c not in df.columns]
    if missing:
        return "error", missing, None, None

    clean_df = df[fitur].copy()
    for col in fitur:
        clean_df[col] = (
            clean_df[col].astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
            .fillna(0)
        )
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    scaled_all = scaler.transform(clean_df)
    predictions = [convert_to_label(p) for p in model.predict(scaled_all)]
    df["Prediksi Kebakaran"] = predictions

    return df, clean_df, scaled_all, fitur

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
            <button style='padding: 6px 16px; background-color: #1f77b4; color: white; border: none; border-radius: 4px; cursor: pointer;'>Data Cloud</button>
            </a>
            """,
            unsafe_allow_html=True
        )

st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# =========================================================================
# === BAGIAN REALTIME FRAGMENT (KOLOM KIRI YANG REFRESH 7 DETIK) ==========
# =========================================================================
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
        f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
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
                except ValueError:
                    pass

            st.pyplot(fig, bbox_inches='tight', dpi=300)
            plt.close(fig) 
            plt.clf()
            plt.rcParams.update({'font.size': 10})

            shap_vals_arr = shap_values[0].values
            kontribusi = []
            for nama_f, shap_v in zip(fitur, shap_vals_arr):
                pct_f = (abs(float(shap_v)) / total_abs_shap) * 100 if total_abs_shap > 0 else 0.0
                kontribusi.append({
                    "fitur": nama_f,
                    "shap_val": float(shap_v),
                    "pct": pct_f
                })
            kontribusi = sorted(kontribusi, key=lambda x: x["pct"], reverse=True)

            st.markdown("<h4 style='margin-top: 25px;'>Analisis Detail Keputusan Model (XAI)</h4>", unsafe_allow_html=True)

            if risk_label == "Low / Rendah":
                st.success("Kondisi lingkungan saat ini terpantau **sangat aman dan stabil**. Berdasarkan analisis *Explainable AI* (SHAP), berikut adalah dominasi faktor-faktor alam yang sukses meredam potensi kebakaran:")
            elif risk_label == "Moderate / Sedang":
                st.info("Kondisi lingkungan saat ini terpantau **cukup stabil namun memerlukan pemantauan berkala**. Berikut adalah rincian faktor yang memengaruhi keseimbangan risiko saat ini:")
            elif risk_label == "High / Tinggi":
                st.warning("Kondisi lingkungan saat ini terpantau **kritis**. Berdasarkan analisis *Explainable AI* (SHAP), terdapat ancaman bahaya yang dipicu oleh memburuknya faktor-faktor berikut:")
            elif risk_label == "Very High / Sangat Tinggi":
                st.error("Kondisi lingkungan saat ini berada pada fase **SANGAT EKSTREM**. Faktor-faktor alam berikut secara masif mendorong eskalasi kebakaran lahan ke tingkat bahaya tertinggi:")

            icons = ["🔴", "🟠", "🟡", "🟢", "⚪"]

            for i, factor in enumerate(kontribusi):
                icon = icons[i] if i < len(icons) else "⚪"
                nama_fitur = str(factor['fitur']).lower()
                persen = factor['pct']
                arah = factor['shap_val']

                st.markdown(f"**{icon} {factor['fitur'].title()} ({persen:.1f}%)**")

                if persen < 5.0:
                    if arah > 0:
                        st.write("- Memberikan dorongan minor terhadap potensi risiko. Pengaruhnya saat ini tertutupi oleh faktor dominan lainnya.")
                    else:
                        st.write("- Memiliki efek peredaman yang sangat kecil terhadap prediksi saat ini. Kondisinya belum cukup signifikan untuk memengaruhi status lingkungan secara keseluruhan.")
                else:
                    if "tanah" in nama_fitur:
                        if arah > 0:
                            st.write("- **Meningkatkan Risiko:** Merupakan faktor pendorong utama. Kelembaban tanah yang rendah menunjukkan kondisi lahan yang teramat kering.")
                        else:
                            st.write("- **Meredam Risiko:** Kelembaban tanah terdeteksi cukup tinggi (basah/lembab). Bertindak sebagai tameng alami.")
                    elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur:
                        if arah > 0:
                            st.write("- **Meningkatkan Risiko:** Udara yang kering mempercepat proses pengeringan bahan bakar alami.")
                        else:
                            st.write("- **Meredam Risiko:** Tingkat kelembapan udara yang tinggi membantu menjaga kebasahan partikel.")
                    elif "angin" in nama_fitur or "ff" in nama_fitur:
                        if arah > 0:
                            st.write("- **Mempercepat Eskalasi:** Kecepatan angin saat ini berisiko memperluas area kebakaran dengan sangat cepat.")
                        else:
                            st.write("- **Kondisi Stabil:** Pergerakan angin yang relatif lambat dan tenang tidak memberikan ancaman berarti.")
                    elif "suhu" in nama_fitur or "temperatur" in nama_fitur or "tavg" in nama_fitur:
                        if arah > 0:
                            st.write("- **Meningkatkan Risiko:** Suhu lingkungan yang sangat panas memicu penguapan air dari vegetasi.")
                        else:
                            st.write("- **Meredam Risiko:** Suhu udara yang tergolong sejuk atau normal menjaga stabilitas termal lingkungan.")
                    elif "hujan" in nama_fitur or "rr" in nama_fitur:
                        if arah > 0:
                            st.write("- **Meningkatkan Risiko:** Ketiadaan curah hujan menghilangkan faktor pendingin alami utama.")
                        else:
                            st.write("- **Meredam Risiko:** Curah hujan yang turun merupakan faktor pendingin krusial.")
                    else:
                        if arah > 0:
                            st.write("- Secara kalkulasi sistem berkontribusi dalam meningkatkan potensi risiko kebakaran.")
                        else:
                            st.write("- Secara kalkulasi sistem berkontribusi menstabilkan potensi risiko kebakaran.")

        except Exception as e:
            st.error(f"Visualisasi XAI belum dapat diproses: {e}")

    with st.expander("Tindak Lanjut Instansi"):
        if risk_label == "Low / Rendah":
            st.markdown("""
**Tindakan Instansi:**
1. Monitoring rutin kondisi lingkungan
2. Patroli berkala ringan
3. Edukasi preventif kepada masyarakat
4. Dokumentasi dan pelaporan kondisi normal
""")
        elif risk_label == "Moderate / Sedang":
            st.markdown("""
**Tindakan Instansi:**
1. Peningkatan frekuensi patroli
2. Penyampaian peringatan dini terbatas
3. Koordinasi internal BPBD dan aparat desa
4. Pengawasan aktivitas pembakaran terbuka
""")
        elif risk_label == "High / Tinggi":
            st.markdown("""
**Tindakan Instansi:**
1. Aktivasi pos siaga tingkat lokal
2. Penempatan personel siaga di titik rawan
3. Koordinasi dengan TNI/Polri dan Manggala Agni
4. Peringatan dini terbuka kepada masyarakat
5. Penyiapan peralatan pemadaman awal
""")
        elif risk_label == "Very High / Sangat Tinggi":
            st.markdown("""
**Tindakan Instansi:**
1. Penetapan status siaga darurat tingkat lokal
2. Aktivasi penuh posko tanggap darurat
3. Mobilisasi tim pemantauan dan pemadam
4. Koordinasi lintas sektor (BPBD, TNI, Polri, DLH, Manggala Agni)
5. Penyebaran peringatan dini melalui media resmi
6. Pengetatan larangan pembakaran terbuka
""")


# =========================================================================
# === BAGIAN PETA REALTIME PEKANBARU FRAGMENT (REFRESH 7 DETIK) ===========
# =========================================================================
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
            try:
                waktu_valid = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
            except Exception: pass
            
        if pd.notna(waktu_valid):
            tanggal_valid = waktu_valid.strftime('%d %B %Y, %H:%M WIB')
        else:
            tanggal_valid = str(last_row['Waktu'])
        
        pekanbaru_coords = [0.5333, 101.4500] 
        color_map = {
            "Low / Rendah": "blue",
            "Moderate / Sedang": "green",
            "High / Tinggi": "orange",
            "Very High / Sangat Tinggi": "red"
        }
        marker_color = color_map.get(risk_label, "gray")

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
                kontribusi_map.append({
                    "fitur": nama_f,
                    "shap_val": float(shap_v),
                    "pct": pct_f
                })
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
                    if "tanah" in nama_fitur:
                        desc = "Meningkatkan Risiko (Kering)" if arah > 0 else "Meredam Risiko (Lembab)"
                    elif "udara" in nama_fitur or "rh" in nama_fitur or "kelembapan" in nama_fitur:
                        desc = "Memperburuk (Udara Kering)" if arah > 0 else "Menjaga Kebasahan (Lembap)"
                    elif "angin" in nama_fitur or "ff" in nama_fitur:
                        desc = "Mempercepat Eskalasi (O2)" if arah > 0 else "Kondisi Stabil (Tenang)"
                    elif "suhu" in nama_fitur or "temperatur" in nama_fitur or "tavg" in nama_fitur:
                        desc = "Memicu Penguapan (Panas)" if arah > 0 else "Stabilitas Termal (Normal)"
                    elif "hujan" in nama_fitur or "rr" in nama_fitur:
                        desc = "Tanpa Hujan (Pendingin Hilang)" if arah > 0 else "Faktor Pendingin (Hujan)"
                    else:
                        desc = "Meningkatkan Potensi" if arah > 0 else "Menstabilkan Potensi"
                    
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
            tl_html = """<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:11px;'>
                <li>Monitoring rutin kondisi lingkungan</li>
                <li>Patroli berkala ringan</li>
                <li>Edukasi preventif kepada masyarakat</li>
                <li>Dokumentasi dan pelaporan kondisi normal</li>
            </ul>"""
        elif risk_label == "Moderate / Sedang":
            tl_html = """<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:11px;'>
                <li>Peningkatan frekuensi patroli</li>
                <li>Penyampaian peringatan dini terbatas</li>
                <li>Koordinasi internal BPBD dan aparat desa</li>
                <li>Pengawasan aktivitas pembakaran terbuka</li>
            </ul>"""
        elif risk_label == "High / Tinggi":
            tl_html = """<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:11px;'>
                <li>Aktivasi pos siaga tingkat lokal</li>
                <li>Penempatan personel siaga di titik rawan</li>
                <li>Koordinasi dengan TNI/Polri dan Manggala Agni</li>
                <li>Peringatan dini terbuka masyarakat</li>
                <li>Penyiapan peralatan pemadaman awal</li>
            </ul>"""
        else:
            tl_html = """<ul style='margin: 4px 0 0 0; padding-left: 18px; color:#333; font-size:11px;'>
                <li>Status siaga darurat tingkat lokal</li>
                <li>Aktivasi penuh posko tanggap darurat</li>
                <li>Mobilisasi tim pemantauan & pemadam</li>
                <li>Koordinasi lintas sektor (BPBD, TNI, Polri, DLH)</li>
                <li>Penyebaran peringatan dini media resmi</li>
                <li>Pengetatan larangan pembakaran terbuka</li>
            </ul>"""

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
            <b>Kelembaban Tanah:</b> {float(last_num[fitur[4]]):.1f} %<br>
            <b>Waktu:</b> {last_row['Waktu']}
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
                style_function=lambda feature, color=marker_color: {
                    'fillColor': color, 'color': color, 'weight': 2, 'fillOpacity': 0.4,   
                },
                tooltip=folium.GeoJsonTooltip(fields=['nama'], aliases=['Wilayah:'], style="font-weight: bold; font-size: 14px;")
            ).add_to(m)

        folium.Marker(location=pekanbaru_coords, popup=popup_text, icon=folium.Icon(color=marker_color, icon="info-sign")).add_to(m)

        logo_base64 = ""
        logo_upi_base64 = ""
        try:
            if os.path.exists("logo.png"):
                with open("logo.png", "rb") as image_file:
                    logo_base64 = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
                    
            if os.path.exists("logo upi yptk.png"):
                with open("logo upi yptk.png", "rb") as upi_file:
                    logo_upi_base64 = f"data:image/png;base64,{base64.b64encode(upi_file.read()).decode()}"
        except Exception:
            pass
            
        logo_img_tag = f'<img src="{logo_base64}" style="height: 55px; background: white; padding: 4px; border-radius: 4px;" alt="Logo">' if logo_base64 else ''
        logo_upi_tag = f'<img src="{logo_upi_base64}" style="width: 60px; height: auto;" alt="Logo UPI YPTK">' if logo_upi_base64 else ''
        
        raw_map_html = m.get_root().render()

        custom_css_and_layout_start = f"""
        <body style="background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
            <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 100%; max-width: 1450px; height: 95vh; display: flex; flex-direction: column;">
                
                <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #1f77b4 0%, #175a8a 100%); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(31,119,180,0.2);">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        {logo_img_tag}
                        <div>
                            <h2 style="margin: 0; font-size: 22px; font-weight: 600; letter-spacing: 0.5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">Dashboard Prediksi Risiko Kebakaran Lahan</h2>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #dceefb;">Integrasi Model Machine Learning, IoT, dan Spatial GIS</p>
                        </div>
                    </div>
                    <div style="text-align: right; font-size: 13px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 10px 15px; border-radius: 8px;">
                        <b style="font-size: 14px; letter-spacing: 0.5px;">Domain/Wilayah:</b> Prov. RIAU - Kota Pekanbaru<br>
                        <span style="color: #e2f0ff;"><b>Valid/Berlaku:</b> {tanggal_valid} (Observation)</span>
                    </div>
                </div>
                
                <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 95px); overflow: hidden;">
                    
                    <div style="width: 280px; display: flex; flex-direction: column; gap: 15px; overflow-y: auto; padding-right: 5px; flex-shrink: 0;">
                        
                        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                            <b style="font-size: 13px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 10px;">Status Prediksi Saat Ini</b>
                            <div style="font-size: 17px; font-weight: bold; color: {marker_color};">{risk_label}</div>
                        </div>

                        <div style="display: flex; gap: 10px;">
                            <div style="flex: 1; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9; display: flex; flex-direction: column; align-items: center; justify-content: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                <b style="font-size: 12px; color: #333; display: block; margin-bottom: 10px; width: 100%; text-align: left; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Arah Utara</b>
                                <div style="text-align: center;">
                                    <div style="font-weight: bold; font-size: 15px; color: #333; margin-bottom: 3px;">U</div>
                                    <div style="width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-bottom: 16px solid red; margin: 0 auto;"></div>
                                    <div style="width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 16px solid #555; margin: 0 auto;"></div>
                                </div>
                            </div>
                            <div style="flex: 1.5; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                                <b style="font-size: 12px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 8px;">Legenda Risiko</b>
                                <div style="font-size: 11px; line-height: 1.8;">
                                    <div style="display: flex; align-items: center;"><i style="background: blue; width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> Rendah</div>
                                    <div style="display: flex; align-items: center;"><i style="background: green; width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> Sedang</div>
                                    <div style="display: flex; align-items: center;"><i style="background: orange; width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> Tinggi</div>
                                    <div style="display: flex; align-items: center;"><i style="background: red; width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> S. Tinggi</div>
                                </div>
                            </div>
                        </div>

                        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                            <b style="font-size: 13px; color: #333; display: block; border-bottom: 1px solid #ccc; padding-bottom: 8px; margin-bottom: 5px;">Tindak Lanjut Instansi</b>
                            <div style="font-size: 12px; line-height: 1.5;">{tl_html}</div>
                        </div>
                    </div>
                    
                    <div style="flex-grow: 1; border: 2px solid #e2e8f0; border-radius: 10px; overflow: hidden; position: relative; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                        """
        
        custom_layout_end = f"""
                    </div> 
                    
                    <div style="width: 340px; display: flex; flex-direction: column; gap: 18px; flex-shrink: 0; padding-left: 5px;">
                        
                        <div style="flex-grow: 1; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; background: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                            <b style="font-size: 14px; color: #2d3748; display: block; border-bottom: 2px solid #edf2f7; padding-bottom: 12px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.5px;">Faktor Pemicu (XAI SHAP)</b>
                            <div style="font-size: 12px;">{xai_html}</div>
                        </div>

                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                            <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 18px;">
                                {logo_upi_tag}
                                <div style="text-align: left; line-height: 1.5;">
                                    <b style="font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Produced By</b><br>
                                    <span style="font-size: 13px; font-weight: bold; color: #2d3748;">Model HSEL Terintegrasi IoT</span><br>
                                    <span style="font-size: 11px; font-style: italic; color: #4a5568;">Mahasiswa Doctoral Teknologi Informasi<br>Universitas Putra Indonesia YPTK Padang</span>
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

        # Menggunakan Fetch API Blob untuk menghindari Blank Page Browser
        b64_html = base64.b64encode(framed_dashboard_html.encode('utf-8')).decode('utf-8')
        
        custom_button_html = f"""
        <button onclick="openMap()" style="width: 100%; padding: 8px 16px; background-color: #ffffff; color: #333; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 14px; transition: 0.3s;" onmouseover="this.style.borderColor='#1f77b4'; this.style.color='#1f77b4'" onmouseout="this.style.borderColor='#ccc'; this.style.color='#333'">
            🌐 Buka Dashboard Pemantauan Terpadu (Pekanbaru)
        </button>
        
        <script>
        function openMap() {{
            const b64Data = "{b64_html}";
            fetch(`data:text/html;base64,${{b64Data}}`)
            .then(res => res.blob())
            .then(blob => {{
                const url = URL.createObjectURL(blob);
                window.open(url, '_blank');
            }})
            .catch(err => console.error("Error rendering map:", err));
        }}
        </script>
        """
        components.html(custom_button_html, height=50)


# =========================================================================
# === BAGIAN PETA REGIONAL FRAGMENT (PEKANBARU, SIAK, PELALAWAN, BENGKALIS)
# =========================================================================
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
            try:
                waktu_valid = pd.to_datetime(str(last_row['Waktu']), dayfirst=False, errors='coerce')
            except Exception: pass
            
        tanggal_valid = waktu_valid.strftime('%d %B %Y, %H:%M WIB') if pd.notna(waktu_valid) else str(last_row['Waktu'])
        
        color_map = {
            "Low / Rendah": "blue",
            "Moderate / Sedang": "green",
            "High / Tinggi": "orange",
            "Very High / Sangat Tinggi": "red"
        }
        marker_color_pku = color_map.get(risk_label_pku, "gray")

        # LOAD GAMBAR BASE64 UNTUK DASHBOARD REGIONAL
        logo_base64 = ""
        logo_upi_base64 = ""
        try:
            if os.path.exists("logo.png"):
                with open("logo.png", "rb") as image_file:
                    logo_base64 = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
            if os.path.exists("logo upi yptk.png"):
                with open("logo upi yptk.png", "rb") as upi_file:
                    logo_upi_base64 = f"data:image/png;base64,{base64.b64encode(upi_file.read()).decode()}"
        except Exception:
            pass

        logo_img_tag = f'<img src="{logo_base64}" style="height: 55px; background: white; padding: 4px; border-radius: 4px;" alt="Logo">' if logo_base64 else ''
        logo_upi_tag = f'<img src="{logo_upi_base64}" style="width: 60px; height: auto;" alt="Logo UPI YPTK">' if logo_upi_base64 else ''
        
        # 1. PERSIAPAN PETA 
        regional_coords = [0.8500, 101.9000] # Digeser sedikit agar Bengkalis dan lainnya terlihat pas
        m_regional = folium.Map(location=regional_coords, zoom_start=7.5, control_scale=True, tiles='OpenStreetMap')
        Fullscreen(position='topright').add_to(m_regional)

        # 2. BACA GEOJSON DARI CACHE & FILTER HANYA 4 WILAYAH (Termasuk Bengkalis)
        riau_geojson_data = load_riau_geojson()
        
        if riau_geojson_data:
            riau_geojson = copy.deepcopy(riau_geojson_data) 
            filtered_features = []
            
            for feature in riau_geojson['features']:
                nama_wilayah = feature['properties'].get('nama', '').lower()
                kab_kota = feature['properties'].get('kab_kota', '').lower()
                
                # Cek Pekanbaru (Real-time)
                if 'pekanbaru' in nama_wilayah or 'pekanbaru' in kab_kota:
                    feature['properties']['warna_fill'] = marker_color_pku
                    feature['properties']['tooltip_info'] = f"Status: {risk_label_pku} (Real-time)"
                    filtered_features.append(feature)
                
                # Cek Siak (Dummy)
                elif 'siak' in nama_wilayah or 'siak' in kab_kota:
                    feature['properties']['warna_fill'] = "#9e9e9e" # Abu-abu
                    feature['properties']['tooltip_info'] = "Status: Menunggu Data IoT"
                    filtered_features.append(feature)
                
                # Cek Pelalawan (Dummy)
                elif 'pelalawan' in nama_wilayah or 'pelalawan' in kab_kota:
                    feature['properties']['warna_fill'] = "#9e9e9e" # Abu-abu
                    feature['properties']['tooltip_info'] = "Status: Menunggu Data IoT"
                    filtered_features.append(feature)
                
                # Cek Bengkalis (Dummy)
                elif 'bengkalis' in nama_wilayah or 'bengkalis' in kab_kota:
                    feature['properties']['warna_fill'] = "#9e9e9e" # Abu-abu
                    feature['properties']['tooltip_info'] = "Status: Menunggu Data IoT"
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
                tooltip=folium.GeoJsonTooltip(
                    fields=['nama', 'tooltip_info'], 
                    aliases=['Kab/Kota:', 'Keterangan:'], 
                    style="font-weight: bold; font-size: 13px;"
                )
            ).add_to(m_regional)
            
            # --- PEMBUATAN MARKER / POIN LOKASI ---
            # Popup Khusus Pekanbaru (Real Data)
            popup_pku_html = f"""<div style='width: 230px; font-size: 13px; line-height: 1.5;'>
                <b>Wilayah:</b> Kota Pekanbaru<br>
                <b>Prediksi:</b> {risk_label_pku}<br>
                <b>Suhu:</b> {float(last_num[fitur[0]]):.1f} °C<br>
                <b>Kelembapan:</b> {float(last_num[fitur[1]]):.1f} %<br>
                <b>Curah Hujan:</b> {float(last_num[fitur[2]]):.1f} mm<br>
                <b>Kecepatan Angin:</b> {float(last_num[fitur[3]]):.1f} m/s<br>
                <b>Kelembaban Tanah:</b> {float(last_num[fitur[4]]):.1f} %<br>
                <b>Waktu:</b> {last_row['Waktu']}
                </div>"""
            pekanbaru_coords = [0.5333, 101.4500]
            folium.Marker(location=pekanbaru_coords, popup=folium.Popup(popup_pku_html, max_width=250), icon=folium.Icon(color=marker_color_pku, icon="info-sign")).add_to(m_regional)

            # Fungsi Pembuat Popup Dummy untuk Wilayah Lain
            def get_dummy_popup(nama_daerah):
                return f"""<div style='width: 230px; font-size: 13px; line-height: 1.5;'>
                <b>Wilayah:</b> {nama_daerah}<br>
                <b>Prediksi:</b> Unknown / Menunggu Data<br>
                <b>Suhu:</b> Unknown<br>
                <b>Kelembapan:</b> Unknown<br>
                <b>Curah Hujan:</b> Unknown<br>
                <b>Kecepatan Angin:</b> Unknown<br>
                <b>Kelembaban Tanah:</b> Unknown<br>
                <b>Waktu:</b> -
                </div>"""

            # Marker Siak
            siak_coords = [0.7490, 102.0460]
            folium.Marker(location=siak_coords, popup=folium.Popup(get_dummy_popup("Kabupaten Siak"), max_width=250), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)
            
            # Marker Pelalawan
            pelalawan_coords = [0.2662, 101.6917]
            folium.Marker(location=pelalawan_coords, popup=folium.Popup(get_dummy_popup("Kabupaten Pelalawan"), max_width=250), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)
            
            # Marker Bengkalis
            bengkalis_coords = [1.4789, 102.1444]
            folium.Marker(location=bengkalis_coords, popup=folium.Popup(get_dummy_popup("Kabupaten Bengkalis"), max_width=250), icon=folium.Icon(color="gray", icon="info-sign")).add_to(m_regional)

        # 3. RENDER PETA KE HTML
        raw_map_html = m_regional.get_root().render()

        # 4. INJEKSI LAYOUT HTML DASHBOARD BARU (Versi Profesional)
        custom_css_and_layout_start = f"""
        <body style="background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; box-sizing: border-box;">
            <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); width: 100%; max-width: 1450px; height: 95vh; display: flex; flex-direction: column;">
                
                <div style="display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; flex-shrink: 0; box-shadow: 0 4px 15px rgba(30,60,114,0.2);">
                    <div style="display: flex; align-items: center; gap: 15px;">
                        {logo_img_tag}
                        <div>
                            <h2 style="margin: 0; font-size: 22px; font-weight: 600; letter-spacing: 0.5px;">Pemantauan Regional (Pekanbaru, Siak, Pelalawan, Bengkalis)</h2>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #d1e8ff;">Tahap Perluasan Integrasi Sensor IoT</p>
                        </div>
                    </div>
                    <div style="text-align: right; font-size: 13px; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 10px 15px; border-radius: 8px;">
                        <b style="font-size: 14px; letter-spacing: 0.5px;">Domain:</b> Sebagian Wilayah Riau<br>
                        <span style="color: #e2f0ff;"><b>Valid/Berlaku:</b> {tanggal_valid}</span>
                    </div>
                </div>
                
                <div style="display: flex; gap: 20px; flex-grow: 1; height: calc(100% - 95px); overflow: hidden;">
                    
                    <div style="width: 330px; display: flex; flex-direction: column; gap: 18px; overflow-y: auto; padding-right: 5px;">
                        
                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                            <b style="font-size: 14px; color: #2d3748; display: block; border-bottom: 2px solid #edf2f7; padding-bottom: 12px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.5px;">Legenda Status Kawasan</b>
                            <div style="font-size: 13px; color: #4a5568; line-height: 2.4;">
                                <div style="display: flex; align-items: center;"><i style="background: blue; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 12px; box-shadow: 0 0 0 3px rgba(0,0,255,0.1);"></i> Risiko Rendah</div>
                                <div style="display: flex; align-items: center;"><i style="background: green; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 12px; box-shadow: 0 0 0 3px rgba(0,128,0,0.1);"></i> Risiko Sedang</div>
                                <div style="display: flex; align-items: center;"><i style="background: orange; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 12px; box-shadow: 0 0 0 3px rgba(255,165,0,0.1);"></i> Risiko Tinggi</div>
                                <div style="display: flex; align-items: center;"><i style="background: red; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 12px; box-shadow: 0 0 0 3px rgba(255,0,0,0.1);"></i> Risiko S. Tinggi</div>
                                <div style="height: 1px; background: #edf2f7; margin: 12px 0;"></div>
                                <div style="display: flex; align-items: center;"><i style="background: #9e9e9e; width: 14px; height: 14px; display: inline-block; border-radius: 50%; margin-right: 12px; box-shadow: 0 0 0 3px rgba(158,158,158,0.2);"></i> <b style="color: #2d3748;">Menunggu Data IoT</b></div>
                            </div>
                        </div>

                        <div style="background: #fffaf0; border: 1px solid #feebc8; border-left: 5px solid #ed8936; border-radius: 8px; padding: 18px; color: #9c4221; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                            <b style="font-size: 14px; display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">💡 Catatan Sistem</b>
                            <span style="font-size: 13px; line-height: 1.6; display: block; color: #7b341e;">Pemantauan <i>real-time</i> aktif di Pekanbaru. Kab. Siak, Kab. Pelalawan, dan Kab. Bengkalis dalam tahap persiapan pemasangan perangkat sensor.</span>
                        </div>

                        <div style="flex-grow: 1;"></div>

                        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                            <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 18px;">
                                {logo_upi_tag}
                                <div style="text-align: left; line-height: 1.5;">
                                    <b style="font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.8px;">Produced By</b><br>
                                    <span style="font-size: 13px; font-weight: bold; color: #2d3748;">Model HSEL Terintegrasi IoT</span><br>
                                    <span style="font-size: 11px; font-style: italic; color: #4a5568;">Mahasiswa Doctoral Teknologi Informasi<br>Universitas Putra Indonesia YPTK Padang</span>
                                </div>
                            </div>
                            <div style="border-top: 1px dashed #cbd5e0; padding-top: 15px; text-align: center;">
                                <div style="font-size: 12px; color: #3182ce; font-weight: 600; margin-bottom: 8px;">Processed Date: <span style="color: #2b6cb0; font-weight: normal;">{tanggal_valid}</span></div>
                                <div style="font-size: 11px; color: #718096;"><b style="color: #4a5568;">Data Source:</b> Sensor IoT Lokal, HSEL Prediction</div>
                            </div>
                        </div>

                    </div>
                    
                    <div style="flex-grow: 1; border: 2px solid #e2e8f0; border-radius: 10px; overflow: hidden; position: relative; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
        """
        
        custom_layout_end = "</div></div></div></body>"
        framed_dashboard_html = raw_map_html.replace('<body>', custom_css_and_layout_start).replace('</body>', custom_layout_end)

        # 5. RENDER TOMBOL DENGAN JAVASCRIPT FETCH
        b64_html = base64.b64encode(framed_dashboard_html.encode('utf-8')).decode('utf-8')
        
        custom_button_html = f"""
        <button onclick="openMapRegional()" style="width: 100%; padding: 8px 16px; background-color: #e67e22; color: #ffffff; border: none; border-radius: 4px; cursor: pointer; font-family: sans-serif; font-size: 14px; margin-top: 10px; font-weight: bold; transition: 0.3s;" onmouseover="this.style.backgroundColor='#d35400'" onmouseout="this.style.backgroundColor='#e67e22'">
            🗺️ Buka Dashboard Pemantauan Regional
        </button>
        
        <script>
        function openMapRegional() {{
            const b64Data = "{b64_html}";
            fetch(`data:text/html;base64,${{b64Data}}`)
            .then(res => res.blob())
            .then(blob => {{
                const url = URL.createObjectURL(blob);
                window.open(url, '_blank');
            }})
            .catch(err => console.error("Error rendering map:", err));
        }}
        </script>
        """
        components.html(custom_button_html, height=60)


# =========================================================================
# === BAGIAN UTAMA DASHBOARD (TIDAK REFRESH OTOMATIS) =====================
# =========================================================================
def main_dashboard():
    st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)
    
    df_raw = load_data()
    res = preprocess_sensor_data(df_raw)
    
    col_kiri, col_tengah, col_kanan = st.columns([1.2, 1.2, 1.2])
    
    with col_kiri:
        indikator_kiri_realtime()
        
    with col_tengah:
        st.markdown("<h5 style='text-align: center;'>Visualisasi Peta Lokasi Prediksi Kebakaran</h5>", unsafe_allow_html=True)
        # Eksekusi Map Pekanbaru dan Regional (Tombol akan muncul berurutan)
        peta_realtime_fragment()
        peta_regional_fragment()

    with col_kanan:
        st.markdown("<h5 style='text-align: center;'>IoT Smart Fire Prediction</h5>", unsafe_allow_html=True)
        try:
            image = Image.open("forestiot4.jpg")
            st.image(image.resize((480, 360)))
        except Exception:
            st.info("Gambar 'forestiot4.jpg' tidak ditemukan di direktori aplikasi.")
                
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
                    'Suhu (°C)': '°C',
                    'Kelembapan (%)': '%',
                    'Curah Hujan (mm)': 'mm',
                    'Kecepatan Angin (m/s)': 'm/s',
                    'Kelembaban Tanah (%)': '%'
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
                ).encode(
                    text=alt.Text('LabelText:N'), 
                    opacity=alt.condition(selection, alt.value(1), alt.value(0))
                )

                chart_all = (chart_base + points + text_labels).add_params(selection).properties(height=450).interactive()
                st.altair_chart(chart_all, use_container_width=True)

            with tab1:
                chart_temp = alt.Chart(df_vis).mark_line(color="#ff5733", strokeWidth=3, point=alt.OverlayMarkDef(color="#ff5733", size=50)).encode(
                    x=x_axis, y=alt.Y('Suhu (°C):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Suhu (°C):Q', format='.1f')]
                ).properties(height=350).interactive()
                st.altair_chart(chart_temp, use_container_width=True)

            with tab2:
                chart_hum = alt.Chart(df_vis).mark_line(color="#33d4ff", strokeWidth=3, point=alt.OverlayMarkDef(color="#33d4ff", size=50)).encode(
                    x=x_axis, y=alt.Y('Kelembapan (%):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Kelembapan (%):Q', format='.1f')]
                ).properties(height=350).interactive()
                st.altair_chart(chart_hum, use_container_width=True)

            with tab3:
                base = alt.Chart(df_vis).encode(x=x_axis)
                bar = base.mark_bar(color="#335eff", opacity=0.7, size=25).encode(
                    y=alt.Y('Curah Hujan (mm):Q', title='Curah Hujan (mm)'),
                    tooltip=['Waktu_DT:T', alt.Tooltip('Curah Hujan (mm):Q', format='.1f')]
                )
                line = base.mark_line(color="#ff0000", strokeWidth=2).encode(y=alt.Y('Curah Hujan (mm):Q'))
                point = base.mark_circle(color="#ff0000", size=60).encode(y=alt.Y('Curah Hujan (mm):Q'))
                chart_rain = (bar + line + point).properties(height=350).interactive()
                st.altair_chart(chart_rain, use_container_width=True)

            with tab4:
                chart_wind = alt.Chart(df_vis).mark_line(color="#a833ff", strokeWidth=3, point=alt.OverlayMarkDef(color="#a833ff", size=50)).encode(
                    x=x_axis, y=alt.Y('Kecepatan Angin (m/s):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Kecepatan Angin (m/s):Q', format='.1f')]
                ).properties(height=350).interactive()
                st.altair_chart(chart_wind, use_container_width=True)

            with tab5:
                chart_soil = alt.Chart(df_vis).mark_line(color="#33ff5e", strokeWidth=3, point=alt.OverlayMarkDef(color="#33ff5e", size=50)).encode(
                    x=x_axis, y=alt.Y('Kelembaban Tanah (%):Q'), tooltip=['Waktu_DT:T', alt.Tooltip('Kelembaban Tanah (%):Q', format='.1f')]
                ).properties(height=350).interactive()
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
            data=df_xlsx,
            file_name="hasil_prediksi_kebakaran.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
    with btn_pred:
        st.button("🔍 Prediksi Manual", on_click=do_predict_manual)
    with btn_reset:
        st.button("🧼 Reset Manual", on_click=reset_manual)

    if st.session_state.manual_result:
        hasil = st.session_state.manual_result
        font, bg = risk_styles.get(hasil, ("black", "white"))
        st.markdown(
            f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px; margin-top:15px;'>"
            f"Prediksi Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True
        )

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
        st.error("Model teks gagal dimuat. Pastikan file 'tfidf_vectorizer.joblib' dan 'stacking_text_model.joblib' berada di direktori aplikasi.")
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

            tfidf_details = [{"Kata (Term)": word, "Skor TF-IDF": round(score, 4)}
                             for word, score in zip(feature_names, dense_vector) if score > 0]
            tfidf_details = sorted(tfidf_details, key=lambda x: x["Skor TF-IDF"], reverse=True)
            df_tfidf = pd.DataFrame(tfidf_details)

            prob_dict = {}
            try:
                proba = model_text.predict_proba(X_trans)[0]
                prob_dict = {
                    "Low / Rendah": proba[0],
                    "Moderate / Sedang": proba[1],
                    "High / Tinggi": proba[2],
                    "Very High / Sangat Tinggi": proba[3]
                }
            except Exception:
                pass

            pred = model_text.predict(X_trans)[0]
            label_text = convert_to_label(pred)

            st.session_state.txt_preprocessing = {
                "raw": raw_text, "case_folding": text_lower, "cleansing": text_clean,
                "stopword": text_stopword, "tokenizing": token_display, "stemming": text_stemmed,
                "tfidf_df": df_tfidf, "prob_dict": prob_dict
            }
            st.session_state.txt_result = label_text

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses input teks: {e}")

@st.fragment
def text_prediction_ui():
    st.markdown("<div class='section-title' style='margin-top: 20px;'>Pengujian Menggunakan Data Teks</div>", unsafe_allow_html=True)

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
                st.markdown("**1. Original Text**")
                st.info(steps.get("raw", "-"))
                st.markdown("**2. Case Folding (Pengecilan Huruf)**")
                st.info(steps.get("case_folding", "-"))
                st.markdown("**3. Cleansing (Penghapusan Karakter Khusus & Angka)**")
                st.info(steps.get("cleansing", "-"))
                st.markdown("**4. Stopword (Penghapusan Kata)**")
                st.info(steps.get("stopword", "-"))
                st.markdown("**5. Tokenization (Pemenggalan Kata)**")
                st.info(steps.get("tokenizing", "[]"))
                st.markdown("**6. Stemming (Pemotongan Imbuhan)**")
                st.info(steps.get("stemming", "-"))
                st.markdown("**7. Ekstraksi Fitur (TF-IDF)**")
                df_tfidf_display = steps.get("tfidf_df")
                if df_tfidf_display is not None and not df_tfidf_display.empty:
                    st.dataframe(df_tfidf_display, use_container_width=True)
                else:
                    st.warning("Kata-kata pada input ini tidak dikenali dalam kosakata (vocabulary) model Anda.")

                st.markdown("**8. Analisis Keputusan Model (Probabilitas HSEL)**")
                prob_dict = steps.get("prob_dict")
                if prob_dict:
                    for label, prob in prob_dict.items():
                        st.markdown(f"**{label}** ({prob*100:.1f}%)")
                        st.progress(float(prob))
                else:
                    st.info("Model ini tidak menyediakan metrik probabilitas.")

        hasil = st.session_state.txt_result
        font, bg = risk_styles.get(hasil, ("black", "white"))
        st.markdown(
            f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px; margin-top: 15px; font-size: 16px;'>"
            f"Hasil Prediksi Tingkat Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True
        )

manual_prediction_ui()
text_prediction_ui()

# === FOOTER ===
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='
    margin-top: 20px;
    background-color: black;
    padding: 10px 20px;
    border-radius: 10px;
    text-align: center;
    color: white;
'>
    <p style='margin: 0; font-size: 30px; font-weight: bold; line-height: 1.2;'>Smart Fire Prediction HSEL Model</p>
    <p style='margin: 0; font-size: 13px; line-height: 1.2;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026</p>
</div>
""", unsafe_allow_html=True)
