import streamlit as st
import requests

# Konfigurasi Halaman
st.set_page_config(page_title="Insurance Lapse Risk EWS", layout="wide")

st.title("🛡️ Life Insurance Lapse Early Warning System")
st.markdown("Masukkan data profil polis nasabah untuk mengevaluasi probabilitas gagal bayar (*churn/lapse*) secara *real-time* via Backend API.")

# Konfigurasi URL API Backend Lokal
API_URL = "http://127.0.0.1:8000/predict"

with st.form("prediction_form"):
    st.subheader("Data Input Profil Polis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Profil Tertanggung**")
        entry_age = st.number_input("Entry Age (Usia)", min_value=0, max_value=100, value=35)
        sex = st.selectbox("Sex (Kelamin)", ["M", "F"])
        substandard_risk = st.number_input("Substandard Risk (%)", min_value=0.0, value=0.0, step=10.0)
        
        st.markdown("**2. Identitas Produk & Kanal**")
        channel1 = st.text_input("Channel 1", value="1")
        channel2 = st.text_input("Channel 2", value="1")
        channel3 = st.text_input("Channel 3", value="1")
        
    with col2:
        st.markdown("**3. Keuangan & Pembayaran**")
        payment_mode = st.selectbox("Payment Mode", ["Monthly", "Annually", "Quaterly", "Semiannually", "Single Premium"])
        premium_amount = st.number_input("Premium Amount (Premi)", min_value=0.0, value=250.0, step=50.0)
        benefit_amount = st.number_input("Benefit Amount (UP)", min_value=0.0, value=100000.0, step=10000.0)
        initial_benefit = st.number_input("Initial Benefit", min_value=0.0, value=0.0)
        advance_premium = st.number_input("Advance Premium Count", min_value=0, value=0)
        
    with col3:
        st.markdown("**4. Ketentuan & Durasi Polis**")
        policy_year = st.number_input("Policy Year (Tahun)", min_value=1, value=2)
        policy_year_decimal = st.number_input("Policy Year (Desimal)", min_value=0.0, value=2.0)
        full_benefit = st.selectbox("Full Benefit?", ["N", "Y"])
        non_lapse_guaranteed = st.selectbox("Non Lapse Guaranteed", ["NO NLG", "NLG Suspend", "NLG Not Active", "NLG Active"])
        policy_type_1 = st.text_input("Policy Type 1", value="3")
        policy_type_2 = st.text_input("Policy Type 2", value="5")
        policy_type_3 = st.text_input("Policy Type 3", value="A")

    # Tombol Eksekusi API
    submitted = st.form_submit_button("Analisis Risiko Lapse", type="primary")

# Logika Pemrosesan saat tombol ditekan
if submitted:
    # Merakit payload JSON sesuai Pydantic Schema di FastAPI
    payload = {
        "entry_age": entry_age,
        "sex": sex,
        "payment_mode": payment_mode,
        "non_lapse_guaranteed": non_lapse_guaranteed,
        "substandard_risk": float(substandard_risk),
        "advance_premium_count": advance_premium,
        "initial_benefit": float(initial_benefit),
        "full_benefit": full_benefit,
        "policy_year_decimal": float(policy_year_decimal),
        "policy_year": policy_year,
        "channel1": channel1,
        "channel2": channel2,
        "channel3": channel3,
        "policy_type_1": policy_type_1,
        "policy_type_2": policy_type_2,
        "policy_type_3": policy_type_3,
        "benefit_amount": float(benefit_amount),
        "premium_amount": float(premium_amount)
    }

    try:
        with st.spinner("Mengirim data ke Backend API..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            data = result.get("prediction", {})
            
            st.success("Berhasil menerima respons dari model Aktuaria.")
            
            # Membagi hasil ke dalam metrik visual
            res_col1, res_col2, res_col3 = st.columns(3)
            
            # Pewarnaan metrik berdasarkan status lapse
            is_lapse = data.get("is_lapse_predicted")
            lapse_color = "🔴" if is_lapse else "🟢"
            
            res_col1.metric(label="Prediksi Status", value=f"{lapse_color} {'Lapse (Gugur)' if is_lapse else 'Inforce (Aktif)'}")
            res_col2.metric(label="Probabilitas Lapse", value=data.get("lapse_probability_percentage"))
            res_col3.metric(label="Kategori Risiko (Tier)", value=data.get("risk_tier"))
            
            st.info(f"**Rekomendasi Tindakan Operasional:** {data.get('recommended_action')}")
            
        else:
            st.error(f"Gagal memproses data. Backend mengembalikan status {response.status_code}.")
            st.write(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("Koneksi ditolak. Pastikan Backend FastAPI sedang berjalan di http://127.0.0.1:8000.")
