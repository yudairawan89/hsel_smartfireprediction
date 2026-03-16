import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from io import BytesIO
from streamlit_folium import folium_static
import folium
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Fire Prediction HSEL",
    page_icon="favicon.ico",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
.main {background-color:#F9F9F9;}
.section-title{
background-color:#1f77b4;
color:white;
padding:10px;
border-radius:6px;
font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ================= HELPER =================
def convert_day(day):
    return {
        "Monday":"Senin","Tuesday":"Selasa","Wednesday":"Rabu",
        "Thursday":"Kamis","Friday":"Jumat","Saturday":"Sabtu",
        "Sunday":"Minggu"
    }.get(day,day)

def convert_month(month):
    return {
        "January":"Januari","February":"Februari","March":"Maret",
        "April":"April","May":"Mei","June":"Juni",
        "July":"Juli","August":"Agustus","September":"September",
        "October":"Oktober","November":"November","December":"Desember"
    }.get(month,month)

def convert_label(pred):
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

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return joblib.load("HSEL_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model=load_model()
scaler=load_scaler()

# ================= GOOGLE SHEETS =================
SHEET_ID="1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
CSV_LINK=f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

def load_data():
    return pd.read_csv(CSV_LINK)

# ================= HEADER =================
col1,col2=st.columns([1,9])

with col1:
    st.image("logo.png",width=160)

with col2:
    st.markdown("""
    <h2>Smart Fire Prediction HSEL Model</h2>
    Sistem prediksi risiko kebakaran hutan berbasis Hybrid Stacking Ensemble Learning (HSEL)
    yang terintegrasi dengan data sensor IoT secara real-time.
    """,unsafe_allow_html=True)

st.markdown("<hr>",unsafe_allow_html=True)

# ================= AUTO REFRESH =================
st_autorefresh(interval=7000)

# ================= LOAD DATA =================
df=load_data()

st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>",unsafe_allow_html=True)

if df is None or df.empty:
    st.warning("Data sensor belum tersedia.")
    st.stop()

# ================= NORMALISASI KOLOM =================
df.columns=[c.strip() for c in df.columns]

rename_map={
"Suhu Udara":"Tavg: Temperatur rata-rata (°C)",
"Suhu":"Tavg: Temperatur rata-rata (°C)",
"Temperatur":"Tavg: Temperatur rata-rata (°C)",

"Kelembapan Udara":"RH_avg: Kelembapan rata-rata (%)",
"Kelembapan":"RH_avg: Kelembapan rata-rata (%)",

"Curah Hujan":"RR: Curah hujan (mm)",
"Curah Hujan/Jam":"RR: Curah hujan (mm)",

"Kecepatan Angin":"ff_avg: Kecepatan angin rata-rata (m/s)",
"Kecepatan Angin (ms)":"ff_avg: Kecepatan angin rata-rata (m/s)",

"Kelembapan Tanah":"Kelembaban Permukaan Tanah",
"Soil Moisture":"Kelembaban Permukaan Tanah"
}

df=df.rename(columns=rename_map)

fitur=[
"Tavg: Temperatur rata-rata (°C)",
"RH_avg: Kelembapan rata-rata (%)",
"RR: Curah hujan (mm)",
"ff_avg: Kecepatan angin rata-rata (m/s)",
"Kelembaban Permukaan Tanah"
]

missing=[c for c in fitur if c not in df.columns]

if missing:
    st.error("Kolom berikut tidak ditemukan di Google Sheets:")
    st.write(missing)
    st.write("Kolom yang tersedia:",df.columns)
    st.stop()

# ================= PREPROCESS =================
clean_df=df[fitur].apply(pd.to_numeric,errors="coerce").fillna(0)

scaled=scaler.transform(clean_df)

df["Prediksi Kebakaran"]=[convert_label(p) for p in model.predict(scaled)]

last=df.iloc[-1]

risk=last["Prediksi Kebakaran"]

font,bg=risk_styles.get(risk,("black","white"))

waktu=pd.to_datetime(last["Waktu"],errors="coerce")

hari=convert_day(waktu.strftime("%A"))
bulan=convert_month(waktu.strftime("%B"))

tanggal=waktu.strftime(f"%d {bulan} %Y")

# ================= DASHBOARD =================
col1,col2,col3=st.columns([1.2,1.2,1.2])

# ================= SENSOR =================
with col1:

    st.subheader("Data Sensor Realtime")

    sensor_df=pd.DataFrame({
    "Variabel":fitur,
    "Value":[f"{last[f]:.1f}" for f in fitur]
    })

    st.table(sensor_df)

    st.markdown(
    f"""
    <p style='background-color:{bg};color:{font};padding:10px;border-radius:8px;font-weight:bold;'>
    Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran:
    <span style='font-size:22px;text-decoration:underline'>{risk}</span>
    </p>
    """,
    unsafe_allow_html=True
    )

    # ================= TINDAK LANJUT =================
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
with col2:

    st.subheader("Visualisasi Peta Lokasi")

    coords=[-0.959240,100.396000]

    m=folium.Map(location=coords,zoom_start=11)

    folium.Circle(
    location=coords,
    radius=3000,
    color="green",
    fill=True
    ).add_to(m)

    folium.Marker(coords).add_to(m)

    folium_static(m,width=450,height=350)

# ================= IOT IMAGE =================
with col3:

    st.subheader("IoT Smart Fire Prediction")

    try:
        image=Image.open("forestiot4.jpg")
        st.image(image)
    except:
        st.info("Gambar tidak ditemukan")

# ================= DATA TABLE =================
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>",unsafe_allow_html=True)

st.dataframe(df,use_container_width=True)

# ================= DOWNLOAD =================
def to_excel(data):

    output=BytesIO()

    writer=pd.ExcelWriter(output,engine="xlsxwriter")

    data.to_excel(writer,index=False)

    writer.close()

    return output.getvalue()

st.download_button(
"Download Hasil Prediksi",
to_excel(df),
"hasil_prediksi.xlsx"
)

# ================= FOOTER =================
st.markdown("""
<hr>
<center>
<b>Smart Fire Prediction HSEL Model</b><br>
Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026
</center>
""",unsafe_allow_html=True)
