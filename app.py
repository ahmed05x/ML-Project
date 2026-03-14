import streamlit as st
import pandas as pd
import joblib

# --- إعدادات الصفحة ---
st.set_page_config(page_title="Waze Churn Prediction", layout="wide")

# --- تنسيق CSS مخصص لشكل عصري ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stNumberInput, .stSelectbox {
        border-radius: 10px;
    }
    .predict-btn {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    h1 {
        color: #FFD700;
        text-align: center;
        font-family: 'Arial';
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Waze User Retention Analytics 🚗")
st.markdown("<h4 style='text-align: center; color: #999;'>توقع احتمالية بقاء أو مغادرة المستخدم بناءً على سلوكه</h4>", unsafe_allow_html=True)
st.write("---")

# --- تنظيم المدخلات في 3 أعمدة لشكل مختلف ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 النشاط العام")
    sessions = st.number_input("Sessions", value=100)
    total_sessions = st.number_input("Total Sessions", value=150.0)
    n_days_after_onboarding = st.number_input("Days since onboarding", value=1000)

with col2:
    st.subheader("🛣️ سلوك القيادة")
    drives = st.number_input("Drives", value=80)
    driving_days = st.number_input("Driving Days", value=10)
    driven_km_drives = st.number_input("Total Kilometers", value=3000.0)
    duration_minutes_drives = st.number_input("Total Duration (min)", value=1500.0)

with col3:
    st.subheader("📍 التفضيلات والجهاز")
    total_navigations_fav1 = st.number_input("Navigations to Fav 1", value=50)
    total_navigations_fav2 = st.number_input("Navigations to Fav 2", value=10)
    activity_days = st.number_input("Activity Days", value=15)
    device = st.selectbox("Device Type", options=["iPhone", "Android"])

st.write("---")

# --- زر التوقع في المنتصف ---
if st.button("Generate Prediction / توقع الحالة", use_container_width=True):
    
    # 1. حساب الـ Feature الإضافية اللي ظهرت في الـ head عندك
    # لتجنب القسمة على صفر
    km_per_driving_day = driven_km_drives / driving_days if driving_days > 0 else 0
    
    # 2. تحويل الـ Device لـ Boolean (iPhone = True) حسب الصورة
    device_iphone = 1 if device == "iPhone" else 0
    
    # 3. تجهيز الـ DataFrame بنفس ترتيب الـ Features في الصورة
    input_features = pd.DataFrame([[
        sessions, drives, total_sessions, n_days_after_onboarding,
        total_navigations_fav1, total_navigations_fav2, driven_km_drives,
        duration_minutes_drives, activity_days, driving_days, 
        km_per_driving_day, device_iphone
    ]], columns=[
        'sessions', 'drives', 'total_sessions', 'n_days_after_onboarding',
        'total_navigations_fav1', 'total_navigations_fav2', 'driven_km_drives',
        'duration_minutes_drives', 'activity_days', 'driving_days',
        'km_per_driving_day', 'device_iPhone'
    ])

    # 4. التوقع (تأكد من وجود ملف الموديل)
    try:
        model = joblib.load('rf_model.pkl') # غير الاسم لاسم ملفك
        prediction = model.predict(input_features)
        
        if prediction[0] == 1:
            st.error("⚠️ المتوقع: Churned (المستخدم قد يغادر)")
        else:
            st.success("✅ المتوقع: Retained (المستخدم سيبقى)")
            
    except FileNotFoundError:
        st.warning("لم يتم العثور على ملف الموديل. هذه بيانات تجريبية:")
        st.table(input_features)