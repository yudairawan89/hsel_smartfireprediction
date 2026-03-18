import streamlit as st
import pandas as pd
import joblib
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
from io import BytesIO
from streamlit_folium import folium_static
import folium
from PIL import Image
import re
import altair as alt # LIBRARY WAJIB UNTUK GRAFIK KUSTOM (BAR + LINE)

# === TAMBAHAN LIBRARY SASTRAWI ===
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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

# === LOAD MODEL, SCALER, DAN SASTRAWI ===
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

model = load_model()
scaler = load_scaler()
stopword_remover, stemmer = load_sastrawi()

# === KONFIG GOOGLE SHEET ===
SHEET_ID = "1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM"
SHEET_EDIT_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit?usp=sharing"
SHEET_CSV_LINK = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# === LOAD DATA TANPA CACHE ===
def load_data():
    return pd.read_csv(SHEET_CSV_LINK)

# === HEADER ===
col1, col2 = st.columns([1, 9])
with col1:
    st.image("logo.png", width=170)
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

# === PREDIKSI REALTIME DENGAN AUTOREFRESH ===
realtime = st.container()
with realtime:
    st_autorefresh(interval=7000, key="refresh_realtime")
    df = load_data()

    st.markdown("<div class='section-title'>Hasil Prediksi Data Realtime</div>", unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Data belum tersedia atau kosong di Google Sheets.")
    else:
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
            st.error("Kolom wajib tidak ditemukan di Sheets: " + ", ".join(missing))
            st.dataframe(df.head(), use_container_width=True)
            st.stop()

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

        col_kiri, col_tengah, col_kanan = st.columns([1.2, 1.2, 1.2])

        with col_kiri:
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

            with st.expander("Tindak Lanjut Instansi"):
                if risk_label == "Low / Rendah":
                    st.markdown("""
**Kondisi**\nRisiko kebakaran rendah.\n
**Tindakan**\n• Monitoring rutin kondisi lingkungan\n• Patroli berkala ringan\n• Edukasi preventif kepada masyarakat\n• Dokumentasi kondisi normal
""")
                elif risk_label == "Moderate / Sedang":
                    st.markdown("""
**Kondisi**\nRisiko kebakaran sedang.\n
**Tindakan**\n• Peningkatan frekuensi patroli\n• Peringatan dini terbatas kepada masyarakat\n• Koordinasi BPBD dan aparat desa\n• Pengawasan aktivitas pembakaran terbuka
""")
                elif risk_label == "High / Tinggi":
                    st.markdown("""
**Kondisi**\nRisiko kebakaran tinggi.\n
**Tindakan**\n• Aktivasi pos siaga lokal\n• Penempatan personel siaga\n• Koordinasi TNI/Polri dan Manggala Agni\n• Penyiapan alat pemadaman awal
""")
                elif risk_label == "Very High / Sangat Tinggi":
                    st.markdown("""
**Kondisi**\nRisiko kebakaran sangat tinggi.\n
**Tindakan**\n• Aktivasi posko tanggap darurat\n• Mobilisasi tim pemadam\n• Koordinasi lintas sektor\n• Penyiapan logistik darurat\n• Rekomendasi Operasi Modifikasi Cuaca
""")

        with col_tengah:
            st.markdown("<h5 style='text-align: center;'>Visualisasi Peta Lokasi Prediksi Kebakaran</h5>", unsafe_allow_html=True)
            pekanbaru_coords = [-0.959240, 100.396000]
            color_map = {
                "Low / Rendah": "blue",
                "Moderate / Sedang": "green",
                "High / Tinggi": "orange",
                "Very High / Sangat Tinggi": "red"
            }
            marker_color = color_map.get(risk_label, "gray")

            popup_text = folium.Popup(f"""
                <div style='width: 230px; font-size: 13px; line-height: 1.5;'>
                <b>Prediksi:</b> {risk_label}<br>
                <b>Suhu:</b> {last_num[fitur[0]]:.1f} °C<br>
                <b>Kelembapan:</b> {last_num[fitur[1]]:.1f} %<br>
                <b>Curah Hujan:</b> {last_num[fitur[2]]:.1f} mm<br>
                <b>Kecepatan Angin:</b> {last_num[fitur[3]]:.1f} m/s<br>
                <b>Kelembaban Tanah:</b> {last_num[fitur[4]]:.1f} %<br>
                <b>Waktu:</b> {last_row['Waktu']}
                </div>
            """, max_width=250)

            m = folium.Map(location=pekanbaru_coords, zoom_start=11)
            folium.Circle(
                location=pekanbaru_coords,
                radius=3000,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.3
            ).add_to(m)
            folium.Marker(
                location=pekanbaru_coords,
                popup=popup_text,
                icon=folium.Icon(color=marker_color, icon="info-sign")
            ).add_to(m)

            folium_static(m, width=450, height=350)

        with col_kanan:
            st.markdown("<h5 style='text-align: center;'>IoT Smart Fire Prediction</h5>", unsafe_allow_html=True)
            try:
                image = Image.open("forestiot4.jpg")
                st.image(image.resize((480, 360)))
            except Exception:
                st.info("Gambar 'forestiot4.jpg' tidak ditemukan di direktori aplikasi.")

# === TABEL TINGKAT RISIKO ===
st.markdown("<div class='section-title'>Tabel Tingkat Resiko dan Intensitas Kebakaran</div>", unsafe_allow_html=True)
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
            <td>Blue</td><td>Low / Rendah</td><td style='text-align:left; padding-left: 20px;'>Tingkat resiko kebakaran rendah. Intensitas api pada kategori rendah. Api mudah dikendalikan, cenderung akan padam dengan sendirinya.</td>
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

# === TAMBAHAN VISUALISASI TREN (ALTAIR KUSTOM) ===
if 'clean_df' in locals() and 'df' in locals() and not df.empty:
    st.markdown("<div class='section-title' style='margin-bottom: 15px;'>Visualisasi Tren Data Sensor (15 Hari Terakhir)</div>", unsafe_allow_html=True)
    
    df_chart = clean_df.copy()
    
    # 1. Bersihkan string waktu dan paksa menjadi format Datetime
    waktu_clean = df['Waktu'].astype(str).str.replace(' - ', ' ', regex=False)
    df_chart['Waktu_DT'] = pd.to_datetime(waktu_clean, errors='coerce')
    df_chart = df_chart.dropna(subset=['Waktu_DT'])
    
    if not df_chart.empty:
        # 2. Jadikan Datetime sebagai Index dan hitung rata-rata per hari ('D')
        df_chart = df_chart.set_index('Waktu_DT')
        df_daily = df_chart[fitur].resample('D').mean().dropna()
        
        # 3. Ambil 15 hari terakhir
        df_daily = df_daily.tail(15)
        
        # 4. Ganti nama kolom untuk label grafik
        chart_rename = {
            'Tavg: Temperatur rata-rata (°C)': 'Suhu (°C)',
            'RH_avg: Kelembapan rata-rata (%)': 'Kelembapan (%)',
            'RR: Curah hujan (mm)': 'Curah Hujan (mm)',
            'ff_avg: Kecepatan angin rata-rata (m/s)': 'Kecepatan Angin (m/s)',
            'Kelembaban Permukaan Tanah': 'Kelembaban Tanah (%)'
        }
        df_daily = df_daily.rename(columns=chart_rename)
        
        # 5. Reset index agar Waktu_DT bisa dipakai oleh Altair
        df_vis = df_daily.reset_index()
        
        # --- PENGATURAN SUMBU X KUSTOM (Format "%d %b" -> "01 Mar") ---
        # Sumbu T (Temporal) memastikan tanggal diurutkan berdasarkan kalender, bukan abjad
        x_axis = alt.X('Waktu_DT:T', axis=alt.Axis(format='%d %b', title='Tanggal', labelAngle=-45, grid=False))

        # 6. Render Tab Grafik
        tab_all, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Semua Sensor",
            "🌡️ Suhu Udara",
            "💧 Kelembapan Udara",
            "🌧️ Curah Hujan",
            "💨 Kecepatan Angin",
            "🌱 Kelembapan Tanah"
        ])
        
        with tab_all:
            # Transform data ke format 'long' agar bisa dibuat Multi-Line Legend
            df_melted = df_vis.melt(id_vars=['Waktu_DT'], var_name='Parameter Sensor', value_name='Nilai')
            
            chart_all = alt.Chart(df_melted).mark_line(strokeWidth=2, point=True).encode(
                x=x_axis,
                y=alt.Y('Nilai:Q', title='Nilai Pembacaan'),
                color=alt.Color('Parameter Sensor:N', legend=alt.Legend(orient="bottom", title=None)),
                tooltip=['Waktu_DT:T', 'Parameter Sensor:N', alt.Tooltip('Nilai:Q', format='.1f')]
            ).properties(height=400).interactive()
            
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
            # --- GRAFIK TUMPUK (BAR + LINE) UNTUK CURAH HUJAN ---
            base = alt.Chart(df_vis).encode(x=x_axis)
            
            # Diagram Batang
            bar = base.mark_bar(color="#335eff", opacity=0.7, size=25).encode(
                y=alt.Y('Curah Hujan (mm):Q', title='Curah Hujan (mm)'),
                tooltip=['Waktu_DT:T', alt.Tooltip('Curah Hujan (mm):Q', format='.1f')]
            )
            
            # Diagram Garis + Titik
            line = base.mark_line(color="#ff0000", strokeWidth=2).encode(
                y=alt.Y('Curah Hujan (mm):Q')
            )
            point = base.mark_circle(color="#ff0000", size=60).encode(
                y=alt.Y('Curah Hujan (mm):Q')
            )
            
            # Tumpuk ketiga layer tersebut
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

# === TAMPILKAN DATA LENGKAP ===
st.markdown("<div class='section-title'>Data Sensor Lengkap</div>", unsafe_allow_html=True)
st.dataframe(df if 'df' in locals() else pd.DataFrame(), use_container_width=True)

def to_excel(df_to_save: pd.DataFrame) -> bytes:
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_to_save.to_excel(writer, index=False, sheet_name='Prediksi')
    writer.close()
    return output.getvalue()

if 'df' in locals() and not df.empty:
    df_xlsx = to_excel(df)
    st.download_button(
        label="📥 Download Hasil Prediksi Kebakaran sebagai XLSX",
        data=df_xlsx,
        file_name="hasil_prediksi_kebakaran.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# === PREDIKSI MANUAL ===
st.markdown("<div class='section-title'>Pengujian Menggunakan Data Meteorologi Manual</div>", unsafe_allow_html=True)

if "manual_input" not in st.session_state:
    st.session_state.manual_input = {"suhu": 30.0, "kelembapan": 65.0, "curah": 10.0, "angin": 3.0, "tanah": 50.0}
if "manual_result" not in st.session_state:
    st.session_state.manual_result = None

col1, col2, col3 = st.columns(3)
with col1:
    suhu = st.number_input("Suhu Udara (°C)", value=st.session_state.manual_input["suhu"])
    kelembapan = st.number_input("Kelembapan Udara (%)", value=st.session_state.manual_input["kelembapan"])
with col2:
    curah = st.number_input("Curah Hujan (mm)", value=st.session_state.manual_input["curah"])
    angin = st.number_input("Kecepatan Angin (m/s)", value=st.session_state.manual_input["angin"])
with col3:
    tanah = st.number_input("Kelembaban Tanah (%)", value=st.session_state.manual_input["tanah"])

btn_pred, btn_reset, _ = st.columns([1, 1, 8])
with btn_pred:
    if st.button("🔍 Prediksi Manual"):
        input_df = pd.DataFrame([{
            'Tavg: Temperatur rata-rata (°C)': suhu,
            'RH_avg: Kelembapan rata-rata (%)': kelembapan,
            'RR: Curah hujan (mm)': curah,
            'ff_avg: Kecepatan angin rata-rata (m/s)': angin,
            'Kelembaban Permukaan Tanah': tanah
        }])
        scaled_manual = scaler.transform(input_df)
        st.session_state.manual_result = convert_to_label(model.predict(scaled_manual)[0])
        st.session_state.manual_input.update({"suhu": suhu, "kelembapan": kelembapan, "curah": curah, "angin": angin, "tanah": tanah})

with btn_reset:
    if st.button("🧼 Reset Manual"):
        st.session_state.manual_input = {"suhu": 0.0, "kelembapan": 0.0, "curah": 0.0, "angin": 0.0, "tanah": 0.0}
        st.session_state.manual_result = None
        st.rerun()

if st.session_state.manual_result:
    hasil = st.session_state.manual_result
    font, bg = risk_styles.get(hasil, ("black", "white"))
    st.markdown(
        f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px;'>"
        f"Prediksi Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True
    )

# === PREDIKSI TEKS ===
st.markdown("<div class='section-title'>Pengujian Menggunakan Data Teks</div>", unsafe_allow_html=True)

if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "text_result" not in st.session_state:
    st.session_state.text_result = None
if "text_preprocessing" not in st.session_state:
    st.session_state.text_preprocessing = {}

input_text = st.text_area("Masukkan deskripsi lingkungan:", value=st.session_state.text_input, height=120)

btn_pred_text, btn_reset_text, _ = st.columns([1, 1, 8])
with btn_pred_text:
    if st.button("🔍 Prediksi Teks"):
        if input_text.strip() == "":
            st.warning("Harap masukkan deskripsi teks terlebih dahulu.")
        else:
            try:
                vectorizer = joblib.load("tfidf_vectorizer.joblib")
                model_text = joblib.load("stacking_text_model.joblib")

                # --- PROSES PRE-PROCESSING TEKS HSEL ---
                raw_text = input_text
                text_lower = raw_text.lower()
                text_clean = re.sub(r'[^a-zA-Z\s]', '', text_lower)
                text_stopword = stopword_remover.remove(text_clean)
                tokens = text_stopword.split()
                token_display = "[" + ", ".join(tokens) + "]"
                text_stemmed = stemmer.stem(" ".join(tokens))
                
                # --- TF-IDF TRANSFORM ---
                X_trans = vectorizer.transform([text_stemmed])

                # Ekstrak Bobot TF-IDF
                feature_names = vectorizer.get_feature_names_out()
                dense_vector = X_trans.todense().tolist()[0]
                
                tfidf_details = [{"Kata (Term)": word, "Skor TF-IDF": round(score, 4)} 
                                 for word, score in zip(feature_names, dense_vector) if score > 0]
                tfidf_details = sorted(tfidf_details, key=lambda x: x["Skor TF-IDF"], reverse=True)
                df_tfidf = pd.DataFrame(tfidf_details)
                
                # Ekstrak Probabilitas Model HSEL (jika model mendukung predict_proba)
                prob_dict = {}
                try:
                    proba = model_text.predict_proba(X_trans)[0]
                    # Asumsi model.classes_ mengembalikan urutan [0, 1, 2, 3]
                    prob_dict = {
                        "Low / Rendah": proba[0],
                        "Moderate / Sedang": proba[1],
                        "High / Tinggi": proba[2],
                        "Very High / Sangat Tinggi": proba[3]
                    }
                except Exception:
                    pass # Abaikan jika tidak ada predict_proba

                # Prediksi Label
                pred = model_text.predict(X_trans)[0]
                label_text = convert_to_label(pred)

                # Simpan ke session state
                st.session_state.text_preprocessing = {
                    "raw": raw_text,
                    "case_folding": text_lower,
                    "cleansing": text_clean,
                    "stopword": text_stopword,
                    "tokenizing": token_display,
                    "stemming": text_stemmed,
                    "tfidf_df": df_tfidf,
                    "prob_dict": prob_dict
                }
                
                st.session_state.text_input = input_text
                st.session_state.text_result = label_text

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memuat model atau memproses input: {e}")

with btn_reset_text:
    if st.button("🧼 Reset Teks"):
        st.session_state.text_input = ""
        st.session_state.text_result = None
        st.session_state.text_preprocessing = {}
        st.rerun()

if st.session_state.text_result:
    with st.expander("🛠️ Klik untuk melihat hasil setiap tahapan Pre-processing & Keputusan Model", expanded=False):
        steps = st.session_state.text_preprocessing
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
                st.markdown(f"<span style='font-size:14px; color:gray;'>Ditemukan <b>{len(df_tfidf_display)} kata</b> yang dikenali dari kamus vocabulary model. Berikut rincian bobotnya:</span>", unsafe_allow_html=True)
                st.dataframe(df_tfidf_display, use_container_width=True)
            else:
                st.warning("Kata-kata pada input ini tidak dikenali dalam kosakata (vocabulary) model Anda.")

            # Menampilkan Alasan Prediksi (Probabilitas)
            st.markdown("**8. Analisis Keputusan Model (Probabilitas HSEL)**")
            st.markdown("<span style='font-size:14px; color:gray;'>Berdasarkan kombinasi bobot TF-IDF di atas, berikut adalah tingkat keyakinan model Stacking Ensemble untuk setiap kelas:</span>", unsafe_allow_html=True)
            
            prob_dict = steps.get("prob_dict")
            if prob_dict:
                for label, prob in prob_dict.items():
                    # Menentukan warna bar berdasarkan label
                    bar_color = "blue"
                    if "Moderate" in label: bar_color = "green"
                    elif "Sangat Tinggi" in label: bar_color = "red"
                    elif "High" in label: bar_color = "orange"
                    
                    st.markdown(f"**{label}** ({prob*100:.1f}%)")
                    st.progress(float(prob))
            else:
                st.info("Model ini tidak menyediakan metrik probabilitas (predict_proba).")

    hasil = st.session_state.text_result
    font, bg = risk_styles.get(hasil, ("black", "white"))
    st.markdown(
        f"<p style='color:{font}; background-color:{bg}; padding:10px; border-radius:5px; margin-top: 15px; font-size: 16px;'>"
        f"Hasil Prediksi Tingkat Risiko Kebakaran: <b>{hasil}</b></p>", unsafe_allow_html=True
    )

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
