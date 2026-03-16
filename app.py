import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO
from streamlit_folium import folium_static
import folium
from PIL import Image

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
        'Monday': 'Senin','Tuesday': 'Selasa','Wednesday': 'Rabu',
        'Thursday': 'Kamis','Friday': 'Jumat','Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {
        'January': 'Januari','February': 'Februari','March': 'Maret',
        'April': 'April','May': 'Mei','June': 'Juni','July': 'Juli',
        'August': 'Agustus','September': 'September','October': 'Oktober',
        'November': 'November','December': 'Desember'
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

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("HSEL_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === GOOGLE SHEETS ===
SHEET_ID = "1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_CSV_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def load_data():
    return pd.read_csv(SHEET_CSV_LINK)

# === HEADER ===
col1, col2 = st.columns([1,9])

with col1:
    st.image("logo.png", width=170)

with col2:
    st.markdown("""
    <h2>Smart Fire Prediction HSEL Model</h2>
    Sistem ini menggunakan Hybrid Stacking Ensemble Learning (HSEL) untuk memprediksi risiko kebakaran hutan secara real-time.
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# === REALTIME ===
st_autorefresh(interval=7000)

df = load_data()

st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)

if df is not None and not df.empty:

    fitur = [
        'Tavg: Temperatur rata-rata (°C)',
        'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)',
        'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
    ]

    clean_df = df[fitur].astype(float).fillna(0)

    scaled = scaler.transform(clean_df)

    df["Prediksi Kebakaran"] = [convert_to_label(p) for p in model.predict(scaled)]

    last_row = df.iloc[-1]

    risk_label = last_row["Prediksi Kebakaran"]

    font, bg = risk_styles.get(risk_label, ("black","white"))

    waktu = pd.to_datetime(last_row['Waktu'])

    hari = convert_day_to_indonesian(waktu.strftime('%A'))
    bulan = convert_month_to_indonesian(waktu.strftime('%B'))

    tanggal = waktu.strftime(f'%d {bulan} %Y')

    col1,col2,col3 = st.columns([1.2,1.2,1.2])

# === DATA SENSOR ===
    with col1:

        sensor_df = pd.DataFrame({
            "Variabel":fitur,
            "Value":[f"{last_row[f]:.1f}" for f in fitur]
        })

        st.subheader("Data Sensor Realtime")

        st.table(sensor_df)

        st.markdown(
        f"""
        <p style='background-color:{bg};color:{font};padding:10px;border-radius:8px;font-weight:bold;'>
        Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran:
        <span style='font-size:22px;text-decoration:underline'>{risk_label}</span>
        </p>
        """,
        unsafe_allow_html=True
        )

# === FITUR TINDAK LANJUT ===
        with st.expander("Tindak Lanjut Instansi"):

            if risk_label == "Low / Rendah":

                st.markdown("""
**Kondisi**

Risiko kebakaran rendah dan api relatif mudah dikendalikan.

**Tindakan**

• Monitoring rutin kondisi lingkungan  
• Patroli berkala ringan  
• Edukasi preventif kepada masyarakat  
• Dokumentasi kondisi normal
""")

            elif risk_label == "Moderate / Sedang":

                st.markdown("""
**Kondisi**

Risiko kebakaran sedang dan terdapat indikasi peningkatan potensi kebakaran.

**Tindakan**

• Peningkatan frekuensi patroli  
• Peringatan dini terbatas kepada masyarakat  
• Koordinasi internal BPBD dan aparat desa  
• Pengawasan aktivitas pembakaran terbuka
""")

            elif risk_label == "High / Tinggi":

                st.markdown("""
**Kondisi**

Risiko kebakaran tinggi dan api berpotensi meluas.

**Tindakan**

• Aktivasi pos siaga tingkat lokal  
• Penempatan personel siaga di titik rawan  
• Koordinasi dengan TNI, Polri, dan Manggala Agni  
• Penyiapan peralatan pemadaman awal
""")

            elif risk_label == "Very High / Sangat Tinggi":

                st.markdown("""
**Kondisi**

Risiko kebakaran sangat tinggi dengan potensi eskalasi cepat.

**Tindakan**

• Aktivasi penuh posko tanggap darurat  
• Mobilisasi tim pemantauan dan pemadam  
• Koordinasi lintas sektor  
• Penyiapan logistik darurat  
• Rekomendasi Operasi Modifikasi Cuaca
""")

# === MAP ===
    with col2:

        st.subheader("Visualisasi Peta")

        coords = [-0.959240,100.396000]

        m = folium.Map(location=coords, zoom_start=11)

        folium.Marker(coords).add_to(m)

        folium_static(m,width=450,height=350)

# === GAMBAR IOT ===
    with col3:

        st.subheader("IoT Smart Fire Prediction")

        image = Image.open("forestiot4.jpg")

        st.image(image)

# === DATA TABLE ===
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)

st.dataframe(df,use_container_width=True)

# === DOWNLOAD EXCEL ===
def to_excel(df):

    output = BytesIO()

    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    df.to_excel(writer,index=False)

    writer.close()

    return output.getvalue()

df_xlsx = to_excel(df)

st.download_button(
    label="Download Data Prediksi",
    data=df_xlsx,
    file_name="hasil_prediksi.xlsx"
)

# === FOOTER ===
st.markdown("""
<hr>
<center>
<b>Smart Fire Prediction HSEL Model</b><br>
Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026
</center>
""",unsafe_allow_html=True)
