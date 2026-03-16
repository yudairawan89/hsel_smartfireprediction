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
.main {background-color:#F9F9F9;}
table {width:100%;border-collapse:collapse;}
th,td{border:1px solid #ddd;padding:8px;}
th{background-color:#e0e0e0;text-align:center;}
td{text-align:center;}
.section-title{
background-color:#1f77b4;
color:white;
padding:10px;
border-radius:6px;
font-weight:bold;
}
.scrollable-table{overflow-x:auto;}
</style>
""",unsafe_allow_html=True)

# === FUNGSI BANTUAN ===
def convert_day_to_indonesian(day_name):
    return {
        'Monday':'Senin','Tuesday':'Selasa','Wednesday':'Rabu',
        'Thursday':'Kamis','Friday':'Jumat','Saturday':'Sabtu','Sunday':'Minggu'
    }.get(day_name,day_name)

def convert_month_to_indonesian(month_name):
    return {
        'January':'Januari','February':'Februari','March':'Maret',
        'April':'April','May':'Mei','June':'Juni','July':'Juli',
        'August':'Agustus','September':'September','October':'Oktober',
        'November':'November','December':'Desember'
    }.get(month_name,month_name)

def convert_to_label(pred):
    return {
        0:"Low / Rendah",
        1:"Moderate / Sedang",
        2:"High / Tinggi",
        3:"Very High / Sangat Tinggi"
    }.get(pred,"Unknown")

risk_styles={
"Low / Rendah":("white","blue"),
"Moderate / Sedang":("white","green"),
"High / Tinggi":("black","yellow"),
"Very High / Sangat Tinggi":("white","red")
}

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("HSEL_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model=load_model()
scaler=load_scaler()

# === GOOGLE SHEETS ===
SHEET_ID="1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_EDIT_LINK=f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
SHEET_CSV_LINK=f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def load_data():
    return pd.read_csv(SHEET_CSV_LINK)

# === HEADER ===
col1,col2=st.columns([1,9])

with col1:
    st.image("logo.png",width=170)

with col2:

    st.markdown("""
<div style='margin-left:20px;'>
<h2>Smart Fire Prediction HSEL Model</h2>
<p>
Sistem ini menggunakan Hybrid Stacking Ensemble Learning untuk memprediksi
risiko kebakaran hutan secara realtime berdasarkan data sensor IoT.
</p>
</div>
""",unsafe_allow_html=True)

    col_btn=st.columns([10,1])[1]

    with col_btn:
        st.markdown(
        f"""
<a href='{SHEET_EDIT_LINK}' target='_blank'>
<button style='padding:6px 16px;background-color:#1f77b4;color:white;border:none;border-radius:4px;'>Data Cloud</button>
</a>
""",unsafe_allow_html=True)

st.markdown("<hr>",unsafe_allow_html=True)

# === REALTIME ===
realtime=st.container()

with realtime:

    st_autorefresh(interval=7000,key="refresh")

    df=load_data()

    st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>",unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Data belum tersedia")
    else:

        df.columns=[c.strip() for c in df.columns]

        fitur=[
        'Tavg: Temperatur rata-rata (°C)',
        'RH_avg: Kelembapan rata-rata (%)',
        'RR: Curah hujan (mm)',
        'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembaban Permukaan Tanah'
        ]

        clean_df=df[fitur].copy()

        for col in fitur:
            clean_df[col]=clean_df[col].astype(str).str.replace(',','.').astype(float).fillna(0)

        scaled=scaler.transform(clean_df)

        predictions=[convert_to_label(p) for p in model.predict(scaled)]

        df["Prediksi Kebakaran"]=predictions

        last_row=df.iloc[-1]
        last_num=clean_df.iloc[-1]

        waktu=pd.to_datetime(last_row["Waktu"])

        hari=convert_day_to_indonesian(waktu.strftime("%A"))
        bulan=convert_month_to_indonesian(waktu.strftime("%B"))
        tanggal=waktu.strftime(f"%d {bulan} %Y")

        risk_label=last_row["Prediksi Kebakaran"]

        font,bg=risk_styles.get(risk_label,("black","white"))

        sensor_df=pd.DataFrame({
        "Variabel":fitur,
        "Value":[f"{float(last_num[col]):.1f}" for col in fitur]
        })

        col_kiri,col_tengah,col_kanan=st.columns([1.2,1.2,1.2])

        # ================= LEFT =================
        with col_kiri:

            st.markdown("<h5 style='text-align:center;'>Data Sensor Realtime</h5>",unsafe_allow_html=True)

            st.dataframe(sensor_df,use_container_width=True)

            st.markdown(
            f"<p style='background-color:{bg};color:{font};padding:10px;border-radius:8px;font-weight:bold;'>"
            f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
            f"<span style='text-decoration:underline;font-size:22px;'>{risk_label}</span></p>",
            unsafe_allow_html=True)

            # ================= TINDAK LANJUT =================
            risk=risk_label

            with st.expander("Tindak Lanjut Instansi"):

                if risk=="Low / Rendah":

                    st.markdown("""
**Kondisi**

Risiko kebakaran rendah.

**Tindakan**

• Monitoring rutin kondisi lingkungan  
• Patroli berkala ringan  
• Edukasi preventif kepada masyarakat  
• Dokumentasi kondisi normal
""")

                elif risk=="Moderate / Sedang":

                    st.markdown("""
**Kondisi**

Risiko kebakaran sedang.

**Tindakan**

• Peningkatan frekuensi patroli  
• Peringatan dini terbatas kepada masyarakat  
• Koordinasi BPBD dan aparat desa  
• Pengawasan aktivitas pembakaran terbuka
""")

                elif risk=="High / Tinggi":

                    st.markdown("""
**Kondisi**

Risiko kebakaran tinggi.

**Tindakan**

• Aktivasi pos siaga lokal  
• Penempatan personel siaga  
• Koordinasi TNI/Polri dan Manggala Agni  
• Penyiapan alat pemadaman awal
""")

                elif risk=="Very High / Sangat Tinggi":

                    st.markdown("""
**Kondisi**

Risiko kebakaran sangat tinggi.

**Tindakan**

• Aktivasi posko tanggap darurat  
• Mobilisasi tim pemadam  
• Koordinasi lintas sektor  
• Penyiapan logistik darurat  
• Rekomendasi Operasi Modifikasi Cuaca
""")

        # ================= MAP =================
        with col_tengah:

            st.markdown("<h5 style='text-align:center;'>Visualisasi Peta Lokasi Prediksi Kebakaran</h5>",unsafe_allow_html=True)

            pekanbaru_coords=[-0.959240,100.396000]

            m=folium.Map(location=pekanbaru_coords,zoom_start=11)

            folium.Circle(
            location=pekanbaru_coords,
            radius=3000,
            color="green",
            fill=True,
            fill_opacity=0.3
            ).add_to(m)

            folium.Marker(location=pekanbaru_coords).add_to(m)

            folium_static(m,width=450,height=350)

        # ================= IMAGE =================
        with col_kanan:

            st.markdown("<h5 style='text-align:center;'>IoT Smart Fire Prediction</h5>",unsafe_allow_html=True)

            image=Image.open("forestiot4.jpg")
            st.image(image.resize((480,360)))

# === FOOTER ===
st.markdown("<br><hr>",unsafe_allow_html=True)

st.markdown("""
<div style='background-color:black;padding:10px;border-radius:10px;text-align:center;color:white;'>
<p style='font-size:30px;font-weight:bold;'>Smart Fire Prediction HSEL Model</p>
<p style='font-size:13px;'>Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026</p>
</div>
""",unsafe_allow_html=True)
