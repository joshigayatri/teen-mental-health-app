import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Teen Mental Health & Depression Risk Screener",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom Styling (Calm & Soft Mental Health Palette)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #F8FAF9;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #E2F1ED 0%, #D3E7E1 100%);
        border-left: 6px solid #3B7A57;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .header-title {
        color: #1B4332 !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .header-subtitle {
        color: #2D5F4D !important;
        font-size: 15px;
        font-weight: 500;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 24px;
        color: #92400E;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Result Cards */
    .result-low {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #065F46;
    }
    .result-moderate {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #92400E;
    }
    .result-high {
        background-color: #FEF2F2;
        border: 2px solid #F87171;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        color: #991B1B;
    }
    
    /* Section Cards */
    .card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 17px;
        font-weight: 600;
        color: #2D3748;
        margin-bottom: 15px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
    }
    
    /* Helpline Box */
    .helpline-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 16px;
        margin-top: 20px;
        color: #1E40AF;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Model & Scaler Artifact Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    possible_paths = [
        'models',
        '../models',
        os.path.join(os.path.dirname(__file__), 'models'),
        os.path.join(os.path.dirname(__file__), '../models')
    ]
    
    target_dir = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, 'best_model.pkl')):
            target_dir = p
            break
            
    if not target_dir:
        st.error("⚠️ Model artifacts missing. Please ensure `best_model.pkl`, `scaler.pkl`, and `feature_names.pkl` exist in the `models/` directory.")
        st.stop()
        
    model = joblib.load(os.path.join(target_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(target_dir, 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(target_dir, 'feature_names.pkl'))
    
    return model, scaler, feature_names

model, scaler, feature_names = load_ml_artifacts()


# -----------------------------------------------------------------------------
# 4. Header & Educational Disclaimer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <h1 class="header-title" style="color: #1B4332 !important;">🌱 Teen Mental Health & Depression Screener</h1>
    <p class="header-subtitle" style="color: #2D5F4D !important;">An interactive machine learning tool for evaluating behavioral indicators and depression risk factors in teens.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>📌 Important Educational & Academic Disclaimer:</strong><br>
    This screening tool is built purely for academic and educational demonstration purposes using machine learning algorithms. 
    It is <strong>NOT</strong> a clinical diagnostic instrument, medical assessment, or substitute for professional psychiatric care. 
    If you or a teenager you know is experiencing emotional distress, low mood, or mental health concerns, please consult a qualified healthcare professional or contact a mental health crisis line immediately.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. User Input Form
# -----------------------------------------------------------------------------
st.markdown("### 📋 Input Student Habits & Wellbeing Metrics")

with st.form("screener_form"):
    col1, col2, col3 = st.columns(3)
    
    # --- Column 1: Demographics & Social Media ---
    with col1:
        st.markdown('<div class="card"><div class="card-title">👤 Demographics & Social Media</div>', unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=12, max_value=20, value=16, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        daily_social_media_hours = st.slider("Daily Social Media (Hours)", 0.0, 16.0, 4.5, step=0.5)
        platform_usage = st.selectbox("Primary Social Platform", ["Both", "Instagram", "TikTok"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Column 2: Sleep & Physical Habits ---
    with col2:
        st.markdown('<div class="card"><div class="card-title">🌙 Sleep & Physical Health</div>', unsafe_allow_html=True)
        sleep_hours = st.slider("Nightly Sleep (Hours)", 0.0, 12.0, 6.5, step=0.5)
        screen_time_before_sleep = st.slider("Pre-bed Screen Time (Hours)", 0.0, 6.0, 1.5, step=0.25)
        physical_activity = st.slider("Daily Physical Activity (Hours)", 0.0, 5.0, 1.0, step=0.25)
        academic_performance = st.slider("Academic GPA (US 4.0 scale, 2.0 = C average, 4.0 = A average)", 2.0, 4.0, 3.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Column 3: Stress & Emotional Metrics ---
    with col3:
        st.markdown('<div class="card"><div class="card-title">🧠 Stress & Emotional Rating</div>', unsafe_allow_html=True)
        social_interaction_level = st.select_slider("Social Interaction Level", options=["Low", "Medium", "High"], value="Medium")
        stress_level = st.slider("Perceived Stress Level (1-10)", 1, 10, 5)
        anxiety_level = st.slider("Perceived Anxiety Level (1-10)", 1, 10, 5)
        addiction_level = st.slider("Social Media / Phone Addiction (1-10)", 1, 10, 4)
        st.markdown('</div>', unsafe_allow_html=True)
        
    submit_btn = st.form_submit_button("🔍 Run Mental Health Risk Screening", use_container_width=True)


# -----------------------------------------------------------------------------
# 6. Model Inference & Results Display
# -----------------------------------------------------------------------------
if submit_btn:
    # Build dictionary matching original feature schema
    input_data = {
        'age': age,
        'daily_social_media_hours': daily_social_media_hours,
        'sleep_hours': sleep_hours,
        'screen_time_before_sleep': screen_time_before_sleep,
        'academic_performance': academic_performance,
        'physical_activity': physical_activity,
        'stress_level': stress_level,
        'anxiety_level': anxiety_level,
        'addiction_level': addiction_level,
        'gender_male': 1 if gender == "Male" else 0,
        'platform_usage_Instagram': 1 if platform_usage == "Instagram" else 0,
        'platform_usage_TikTok': 1 if platform_usage == "TikTok" else 0,
        'social_interaction_level_low': 1 if social_interaction_level == "Low" else 0,
        'social_interaction_level_medium': 1 if social_interaction_level == "Medium" else 0
    }
    
    # Convert to DataFrame in exact feature column order
    input_df = pd.DataFrame([input_data])[feature_names]
    
    # Scale input using saved scaler
    scaled_input = scaler.transform(input_df)
    
    # Run model prediction
    pred_class = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    dep_probability = float(probabilities[1]) * 100.0
    
    st.markdown("---")
    st.markdown("### 📊 Screening Assessment Results")
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if dep_probability < 35.0:
            st.markdown(f"""
            <div class="result-low">
                <h2 style="margin:0;">🟢 Low Risk Indicator</h2>
                <h3 style="margin-top:10px;">Predicted Probability: {dep_probability:.1f}%</h3>
                <p>The behavioral metrics suggest a lower probability of depressive symptoms.</p>
            </div>
            """, unsafe_allow_html=True)
        elif dep_probability < 65.0:
            st.markdown(f"""
            <div class="result-moderate">
                <h2 style="margin:0;">🟡 Moderate Risk / Watchful Indicator</h2>
                <h3 style="margin-top:10px;">Predicted Probability: {dep_probability:.1f}%</h3>
                <p>Metrics indicate moderate stress/anxiety patterns. Monitoring wellness and lifestyle habits is recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-high">
                <h2 style="margin:0;">🔴 Elevated Risk Indicator</h2>
                <h3 style="margin-top:10px;">Predicted Probability: {dep_probability:.1f}%</h3>
                <p>Elevated levels of stress, anxiety, or screen habits indicate heightened risk. Professional evaluation is encouraged.</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 🎯 Risk Probability Gauge")
        st.progress(min(int(dep_probability), 100))
        st.write(f"**Depression Risk Probability:** `{dep_probability:.2f}%`")
        st.write(f"**Non-Depression Probability:** `{100 - dep_probability:.2f}%`")
        
        st.info("💡 **Key Protective Factors:** Prioritizing 8+ hours of sleep, regular physical activity, and setting healthy social media screen time boundaries.")

    # -------------------------------------------------------------------------
    # 7. Crisis & Mental Health Resources
    # -----------------------------------------------------------------------------
    st.markdown("""
    <div class="helpline-box">
        <h4 style="margin-top:0;">💚 Support & Crisis Helplines</h4>
        <p style="margin-bottom:6px;">If you or someone you know needs support, compassionate help is available 24/7:</p>
        <ul>
            <li><strong>Suicide & Crisis Lifeline (US):</strong> Call or text <strong>988</strong></li>
            <li><strong>Crisis Text Line:</strong> Text <strong>HOME</strong> to <strong>741741</strong></li>
            <li><strong>The Trevor Project (LGBTQ Youth):</strong> Call <strong>1-866-488-7386</strong> or text <strong>START</strong> to <strong>678-678</strong></li>
            <li><strong>International Resources:</strong> Visit <a href="https://findahelpline.com" target="_blank">findahelpline.com</a> for global crisis lines</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
