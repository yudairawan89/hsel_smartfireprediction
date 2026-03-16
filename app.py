import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from io import BytesIO
from streamlit_folium import folium_static
import folium
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Smart Fire Prediction HSEL", page_icon="favicon.ico", layout="wide")

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
margin-top:15px;
}

table {width:100%;border-collapse:collapse}
th,td {border:1px solid #ddd;padding:8px;text-align:center}

</style>
""", unsafe_allow_html=True)

# ================= HELPER =================
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

rename_map_candidates = {
'Tavg: Temperatur rata-rata (°C)': ['Suhu Udara','Suhu','Temperatur','Suhu (°C)'],
'RH_avg: Kelembapan rata-rata (%)': ['Kelembapan Udara','Kelembapan','RH (%)'],
'RR: Curah hujan (mm)': ['Curah Hujan','Curah Hujan/Jam','RR'],
'ff_avg: Kecepatan angin rata-rata (m/s)': ['Kecepatan Angin','Kecepatan Angin (ms)','Angin (m/s)'],
'Kelembaban Permukaan Tanah': ['Kelembapan Tanah','Soil Moisture']
}

actual_rename={}

for target, candidates in rename_map_candidates.items():
    for cand in candidates:
        if cand in df.columns:
            actual_rename[cand]=target
            break

df=df.rename(columns=actual_rename)

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
Tingkat Risiko Kebakaran:
<span style='font-size:22px;text-decoration:underline'>{risk}</span>
</p>
""",
    unsafe_allow_html=True
    )

    with st.expander("Tindak Lanjut Instansi"):

        if risk=="Low / Rendah":
            st.write("Monitoring rutin kondisi lingkungan dan patroli ringan")

        elif risk=="Moderate / Sedang":
            st.write("Peningkatan patroli dan pengawasan aktivitas pembakaran")

        elif risk=="High / Tinggi":
            st.write("Aktivasi pos siaga dan koordinasi pemadaman")

        elif risk=="Very High / Sangat Tinggi":
            st.write("Mobilisasi tim pemadam dan status siaga darurat")

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

# ================= TABEL RISIKO =================
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>",unsafe_allow_html=True)

st.markdown("""
<table>
<tr>
<th>Warna</th>
<th>Tingkat Risiko</th>
<th>Keterangan</th>
</tr>

<tr style='background-color:blue;color:white'>
<td>Blue</td>
<td>Low / Rendah</td>
<td>Risiko kebakaran rendah dan api mudah dikendalikan.</td>
</tr>

<tr style='background-color:green;color:white'>
<td>Green</td>
<td>Moderate / Sedang</td>
<td>Risiko kebakaran sedang dan masih dapat dikendalikan.</td>
</tr>

<tr style='background-color:yellow;color:black'>
<td>Yellow</td>
<td>High / Tinggi</td>
<td>Risiko kebakaran tinggi dan api sulit dikendalikan.</td>
</tr>

<tr style='background-color:red;color:white'>
<td>Red</td>
<td>Very High / Sangat Tinggi</td>
<td>Risiko kebakaran sangat tinggi dan api sangat sulit dikendalikan.</td>
</tr>

</table>
""",unsafe_allow_html=True)

# ================= DATA SENSOR =================
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>",unsafe_allow_html=True)

st.dataframe(df,use_container_width=True)

# ================= DOWNLOAD =================
def to_excel(data):
    output=BytesIO()
    writer=pd.ExcelWriter(output,engine="xlsxwriter")
    data.to_excel(writer,index=False)
    writer.close()
    return output.getvalue()

st.download_button("Download Hasil Prediksi",to_excel(df),"hasil_prediksi.xlsx")

# ================= FOOTER =================
st.markdown("""
<hr>
<center>
<b>Smart Fire Prediction HSEL Model</b><br>
Dikembangkan oleh Mahasiswa Universitas Putera Indonesia YPTK Padang Tahun 2026
</center>
""",unsafe_allow_html=True)
