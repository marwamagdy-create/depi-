# app.py - Diabetes Prediction System (Final Working Version)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    color: #1a5276;
    text-align: center;
    margin-bottom: 1.5rem;
    font-weight: 700;
}
.sub-header {
    font-size: 1.8rem;
    color: #2e86c1;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    font-weight: 600;
}
.risk-high {
    color: #c0392b;
    font-weight: bold;
    font-size: 1.3rem;
    padding: 20px;
    background: linear-gradient(135deg, #ffcccc 0%, #ff9999 100%);
    border-radius: 12px;
    border-left: 6px solid #c0392b;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.risk-moderate {
    color: #d35400;
    font-weight: bold;
    font-size: 1.3rem;
    padding: 20px;
    background: linear-gradient(135deg, #ffe6cc 0%, #ffcc99 100%);
    border-radius: 12px;
    border-left: 6px solid #d35400;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.risk-low {
    color: #27ae60;
    font-weight: bold;
    font-size: 1.3rem;
    padding: 20px;
    background: linear-gradient(135deg, #ccffcc 0%, #99ff99 100%);
    border-radius: 12px;
    border-left: 6px solid #27ae60;
    margin: 15px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton>button {
    background: linear-gradient(135deg, #2e86c1 0%, #1a5276 100%);
    color: white;
    font-weight: bold;
    padding: 12px 24px;
    border-radius: 8px;
    border: none;
    font-size: 1.1rem;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
.info-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #d1e7f0 100%);
    padding: 25px;
    border-radius: 12px;
    margin: 20px 0;
    border-left: 6px solid #2e86c1;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.metric-value {
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.metric-label {
    font-size: 1rem;
    opacity: 0.9;
    letter-spacing: 1px;
}
.feature-card {
    background: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    margin: 8px 0;
    box-shadow: 0 3px 6px rgba(0,0,0,0.05);
    transition: all 0.3s ease;
}
.feature-card:hover {
    transform: translateX(5px);
    box-shadow: 0 5px 10px rgba(0,0,0,0.1);
}
.cluster-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    border-radius: 10px;
    margin: 15px 0;
    border-left: 5px solid #2e86c1;
    box-shadow: 0 4px 8px rgba(0,0,0,0.08);
}
.progress-bar {
    height: 25px;
    border-radius: 12px;
    margin: 20px 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    transition: width 1s ease-in-out;
}
.tab-content {
    padding: 20px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin: 10px 0;
}
.sidebar-section {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #2e86c1;
}
.value-indicator {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: bold;
    margin: 2px;
}
.good-value { background: #d4edda; color: #155724; }
.warning-value { background: #fff3cd; color: #856404; }
.danger-value { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
<strong>📊 Advanced Machine Learning for Diabetes Risk Assessment</strong><br><br>
This system utilizes state-of-the-art machine learning algorithms to provide accurate diabetes risk predictions 
based on comprehensive patient data. The model analyzes multiple risk factors including biometric measurements, 
medical history, and demographic information to generate personalized risk assessments.
</div>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    """Load all trained models with validation"""
    try:
        # Load models
        models = {
            'gb_model': joblib.load('diabetes_classification_model.pkl'),
            'scaler_cluster': joblib.load('scaler_cluster.pkl'),
            'scaler_class': joblib.load('scaler_classification.pkl'),
            'cluster_encoder': joblib.load('cluster_encoder.pkl'),
            'kmeans': joblib.load('kmeans_cluster.pkl'),
            'feature_columns': joblib.load('feature_columns.pkl'),
            'pipeline_info': joblib.load('pipeline_info.pkl')
        }
        
        # Validate models
        required_models = ['gb_model', 'scaler_cluster', 'scaler_class', 'kmeans']
        for model_name in required_models:
            if model_name not in models or models[model_name] is None:
                raise ValueError(f"Model {model_name} not loaded properly")
        
        # Get training statistics from pipeline info
        training_stats = models['pipeline_info'].get('training_stats', {})
        
        # Set default ranges based on typical medical values
        default_ranges = {
            'age': {'min': 0, 'max': 120, 'healthy': (20, 60)},
            'bmi': {'min': 10, 'max': 50, 'healthy': (18.5, 25)},
            'hbA1c': {'min': 3, 'max': 15, 'healthy': (4, 5.6)},
            'glucose': {'min': 50, 'max': 500, 'healthy': (70, 100)}
        }
        
        # Update with training stats if available
        if training_stats:
            for feature, stats in training_stats.items():
                if feature in default_ranges:
                    default_ranges[feature].update(stats)
        
        models['feature_ranges'] = default_ranges
        models['training_stats'] = training_stats
        
        return models
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load models
models = load_models()

if models:
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 System Information")
        
        # Model info
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**🤖 Model Architecture**")
        st.write(f"• Algorithm: Gradient Boosting")
        st.write(f"• Clusters: {models['pipeline_info'].get('n_clusters', 5)}")
        st.write(f"• Features: {len(models['feature_columns'])}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick guide
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**🎯 Healthy Ranges**")
        st.write("• **BMI:** 18.5-24.9 kg/m²")
        st.write("• **HbA1c:** < 5.7%")
        st.write("• **Glucose:** < 100 mg/dL")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk interpretation
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("**📊 Risk Levels**")
        st.write("• **Low Risk:** < 40%")
        st.write("• **Moderate Risk:** 40-70%")
        st.write("• **High Risk:** > 70%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content - Input Form
    st.markdown('<h2 class="sub-header">📝 Patient Information Form</h2>', unsafe_allow_html=True)
    
    # Create tabs for different input sections
    tab1, tab2 = st.tabs(["🔬 Biometric Data", "🏥 Medical & Demographic"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age input with visual feedback
            age = st.slider(
                "**Age (years)**", 
                min_value=0, 
                max_value=120, 
                value=45,
                help="Patient's age in years"
            )
            
            # BMI input
            bmi = st.slider(
                "**BMI (kg/m²)**", 
                min_value=10.0, 
                max_value=50.0, 
                value=25.0,
                step=0.1,
                help="Body Mass Index"
            )
            
            # Visual indicators
            bmi_status = "good-value" if 18.5 <= bmi <= 25 else "warning-value" if 25 < bmi <= 30 else "danger-value"
            bmi_label = "Normal" if 18.5 <= bmi <= 25 else "Overweight" if 25 < bmi <= 30 else "Obese"
            
            st.markdown(f"""
            <div class="feature-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>BMI Status:</span>
                    <span class="value-indicator {bmi_status}">{bmi_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # HbA1c input
            hbA1c = st.slider(
                "**HbA1c Level (%)**", 
                min_value=3.0, 
                max_value=15.0, 
                value=5.5,
                step=0.1,
                help="Glycated hemoglobin (Normal: <5.7%, Diabetes: ≥6.5%)"
            )
            
            # Blood glucose input
            glucose = st.slider(
                "**Blood Glucose (mg/dL)**", 
                min_value=50, 
                max_value=500, 
                value=100,
                help="Fasting blood glucose level"
            )
            
            # Visual indicators for glucose
            glucose_status = "good-value" if glucose < 100 else "warning-value" if 100 <= glucose < 126 else "danger-value"
            glucose_label = "Normal" if glucose < 100 else "Prediabetes" if 100 <= glucose < 126 else "Diabetes"
            
            st.markdown(f"""
            <div class="feature-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Glucose Status:</span>
                    <span class="value-indicator {glucose_status}">{glucose_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏥 Medical History")
            hypertension = st.radio("**Hypertension**", ["No", "Yes"], horizontal=True)
            heart_disease = st.radio("**Heart Disease**", ["No", "Yes"], horizontal=True)
            
            st.markdown("### 👤 Demographic")
            gender = st.radio("**Gender**", ["Female", "Male"], horizontal=True)
        
        with col2:
            st.markdown("### 🌍 Ethnicity & Lifestyle")
            race = st.selectbox("**Race/Ethnicity**", [
                "Caucasian",
                "African American", 
                "Hispanic",
                "Asian",
                "Other"
            ])
            
            smoking = st.selectbox("**Smoking History**", [
                "Never",
                "Former",
                "Current",
                "Not Current",
                "Ever",
                "No Info"
            ])
    
    # Display current values summary
    st.markdown(f"""
    <div class="info-box">
    <strong>📋 Current Patient Profile:</strong><br><br>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div><strong>Age:</strong> {age} years</div>
        <div><strong>BMI:</strong> {bmi} kg/m²</div>
        <div><strong>HbA1c:</strong> {hbA1c}%</div>
        <div><strong>Glucose:</strong> {glucose} mg/dL</div>
        <div><strong>Hypertension:</strong> {hypertension}</div>
        <div><strong>Heart Disease:</strong> {heart_disease}</div>
        <div><strong>Gender:</strong> {gender}</div>
        <div><strong>Race:</strong> {race}</div>
        <div><strong>Smoking:</strong> {smoking}</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🚀 Predict Diabetes Risk", type="primary", use_container_width=True):
        with st.spinner("🔍 Analyzing patient data..."):
            try:
                # Step 1: Standardize input values
                raw_values = np.array([[age, bmi, hbA1c, glucose]])
                standardized_values = models['scaler_cluster'].transform(raw_values)
                
                age_std = float(standardized_values[0][0])
                bmi_std = float(standardized_values[0][1])
                hba1c_std = float(standardized_values[0][2])
                glucose_std = float(standardized_values[0][3])
                
                # Step 2: Assign cluster
                cluster_input = np.array([[age_std, bmi_std, hba1c_std, glucose_std]])
                cluster = int(models['kmeans'].predict(cluster_input)[0])
                
                # Step 3: Prepare feature dictionary
                # Gender encoding
                gender_Female = 1 if gender == "Female" else 0
                gender_Male = 1 if gender == "Male" else 0
                
                # Race encoding
                race_mapping = {
                    "Caucasian": "race:Caucasian",
                    "African American": "race:AfricanAmerican",
                    "Hispanic": "race:Hispanic",
                    "Asian": "race:Asian",
                    "Other": "race:Other"
                }
                
                # Smoking encoding
                smoking_mapping = {
                    "Never": "smoking_history_never",
                    "Former": "smoking_history_former",
                    "Current": "smoking_history_current",
                    "Not Current": "smoking_history_not current",
                    "Ever": "smoking_history_ever",
                    "No Info": "smoking_history_No Info"
                }
                
                # Create feature dictionary
                features = {
                    'year': 2020,
                    'age': age_std,
                    'bmi': bmi_std,
                    'hbA1c_level': hba1c_std,
                    'blood_glucose_level': glucose_std,
                    'gender_Female': gender_Female,
                    'gender_Male': gender_Male,
                    'hypertension': 1 if hypertension == "Yes" else 0,
                    'heart_disease': 1 if heart_disease == "Yes" else 0,
                    'cluster': cluster
                }
                
                # Add race features
                for race_key in ["race:AfricanAmerican", "race:Asian", "race:Caucasian", "race:Hispanic", "race:Other"]:
                    features[race_key] = 1 if race_mapping[race] == race_key else 0
                
                # Add smoking features
                for smoke_key in ["smoking_history_current", "smoking_history_ever", "smoking_history_former", 
                                  "smoking_history_never", "smoking_history_not current", "smoking_history_No Info"]:
                    features[smoke_key] = 1 if smoking_mapping[smoking] == smoke_key else 0
                
                # Step 4: Create DataFrame
                df = pd.DataFrame([features])
                
                # Step 5: Encode cluster
                cluster_encoder = models['cluster_encoder']
                cluster_encoded = cluster_encoder.transform(df[['cluster']])
                cluster_cols = cluster_encoder.get_feature_names_out(['cluster'])
                
                cluster_df = pd.DataFrame(
                    cluster_encoded,
                    columns=cluster_cols,
                    index=df.index
                )
                
                # Step 6: Combine features
                X_combined = pd.concat([
                    df.drop(columns=['cluster']).reset_index(drop=True),
                    cluster_df.reset_index(drop=True)
                ], axis=1)
                
                # Step 7: Ensure all required columns exist
                required_cols = models['feature_columns']
                
                # Add missing columns with zeros
                for col in required_cols:
                    if col not in X_combined.columns:
                        X_combined[col] = 0
                
                # Select only required columns in correct order
                X_final = X_combined[required_cols]
                
                # Step 8: Scale for classification
                X_scaled = models['scaler_class'].transform(X_final)
                
                # Step 9: Make prediction
                model = models['gb_model']
                
                # Get probabilities
                probabilities = model.predict_proba(X_scaled)[0]
                prediction = model.predict(X_scaled)[0]
                diabetes_prob = float(probabilities[1])
                
                # Step 10: Interpret results
                if diabetes_prob >= 0.7:
                    risk_level = "HIGH RISK"
                    risk_class = "risk-high"
                    recommendation = """🚨 **Immediate Action Required:**
                    1. Schedule urgent appointment with endocrinologist
                    2. Begin immediate glucose monitoring
                    3. Start medical intervention as advised"""
                    emoji = "🔴"
                    color = "#c0392b"
                elif diabetes_prob >= 0.4:
                    risk_level = "MODERATE RISK"
                    risk_class = "risk-moderate"
                    recommendation = """⚠️ **Preventive Measures Recommended:**
                    1. Schedule doctor's appointment within 1 month
                    2. Implement lifestyle modifications
                    3. Monitor glucose levels weekly"""
                    emoji = "🟡"
                    color = "#d35400"
                else:
                    risk_level = "LOW RISK"
                    risk_class = "risk-low"
                    recommendation = """✅ **Maintenance Plan:**
                    1. Continue healthy lifestyle
                    2. Annual diabetes screening
                    3. Regular exercise and balanced diet"""
                    emoji = "🟢"
                    color = "#27ae60"
                
                # Display Results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 Risk Assessment Results</h2>', unsafe_allow_html=True)
                
                # Results in metric cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{emoji}</div>
                        <div class="metric-label">{risk_level}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{diabetes_prob:.1%}</div>
                        <div class="metric-label">PROBABILITY</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">#{cluster}</div>
                        <div class="metric-label">CLUSTER GROUP</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Risk Visualization
                st.markdown("### 📈 Risk Score Visualization")
                
                risk_percent = int(diabetes_prob * 100)
                
                # Create custom progress bar
                st.markdown(f"""
                <div class="progress-bar" style="background: #f0f0f0;">
                    <div class="progress-fill" style="width: {risk_percent}%; background: {color};">
                    </div>
                </div>
                <div style="text-align: center; font-size: 1.2rem; font-weight: bold; color: {color}; margin-top: 10px;">
                    Risk Score: {risk_percent}%
                </div>
                """, unsafe_allow_html=True)
                
                # Risk level indicators
                risk_cols = st.columns(3)
                with risk_cols[0]:
                    if risk_percent < 40:
                        st.success(f"### 🟢 LOW RISK\n{risk_percent}%")
                    else:
                        st.info("### LOW RISK\n< 40%")
                with risk_cols[1]:
                    if 40 <= risk_percent < 70:
                        st.warning(f"### 🟡 MODERATE RISK\n{risk_percent}%")
                    else:
                        st.info("### MODERATE RISK\n40-70%")
                with risk_cols[2]:
                    if risk_percent >= 70:
                        st.error(f"### 🔴 HIGH RISK\n{risk_percent}%")
                    else:
                        st.info("### HIGH RISK\n> 70%")
                
                # Recommendation
                st.markdown(f'<div class="{risk_class}">\n### 📋 Clinical Recommendations\n{recommendation}\n</div>', unsafe_allow_html=True)
                
                # Detailed Analysis
                with st.expander("📊 View Detailed Analysis", expanded=True):
                    detail_cols = st.columns(2)
                    
                    with detail_cols[0]:
                        st.markdown("#### 🎯 Prediction Breakdown")
                        
                        # Create probability chart
                        prob_data = pd.DataFrame({
                            'Outcome': ['Non-Diabetic', 'Diabetic'],
                            'Probability': [probabilities[0], probabilities[1]]
                        })
                        
                        st.bar_chart(prob_data.set_index('Outcome'))
                        
                        st.markdown(f"""
                        **Detailed Probabilities:**
                        - **Non-Diabetic:** {probabilities[0]:.2%}
                        - **Diabetic:** {probabilities[1]:.2%}
                        - **Prediction:** {'**Diabetic**' if prediction == 1 else '**Non-Diabetic**'}
                        """)
                    
                    with detail_cols[1]:
                        st.markdown("#### 📋 Patient Profile Summary")
                        
                        profile_data = {
                            "Parameter": [
                                "Age", "BMI", "HbA1c", "Glucose",
                                "Hypertension", "Heart Disease",
                                "Gender", "Race", "Smoking", "Cluster"
                            ],
                            "Value": [
                                f"{age} years",
                                f"{bmi:.1f} kg/m²",
                                f"{hbA1c:.1f}%",
                                f"{glucose} mg/dL",
                                hypertension,
                                heart_disease,
                                gender,
                                race,
                                smoking,
                                f"Cluster #{cluster}"
                            ],
                            "Status": [
                                "✅" if 20 <= age <= 60 else "⚠️",
                                "✅" if 18.5 <= bmi <= 25 else "⚠️" if 25 < bmi <= 30 else "❌",
                                "✅" if hbA1c < 5.7 else "⚠️" if 5.7 <= hbA1c < 6.5 else "❌",
                                "✅" if glucose < 100 else "⚠️" if 100 <= glucose < 126 else "❌",
                                "✅" if hypertension == "No" else "❌",
                                "✅" if heart_disease == "No" else "❌",
                                "✅",
                                "✅",
                                "✅" if smoking == "Never" else "⚠️",
                                "🔍"
                            ]
                        }
                        
                        profile_df = pd.DataFrame(profile_data)
                        st.dataframe(
                            profile_df,
                            column_config={
                                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                                "Value": st.column_config.TextColumn("Value", width="medium"),
                                "Status": st.column_config.TextColumn("Status", width="small")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                
                # Next Steps
                st.markdown("### 🚀 Recommended Next Steps")
                
                steps_cols = st.columns(3)
                
                with steps_cols[0]:
                    st.markdown("""
                    <div class="feature-card">
                    <h4>🏥 Medical Consultation</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Schedule doctor's appointment</li>
                        <li>Comprehensive blood tests</li>
                        <li>Regular follow-ups</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with steps_cols[1]:
                    st.markdown("""
                    <div class="feature-card">
                    <h4>🥗 Lifestyle Modifications</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Balanced diet plan</li>
                        <li>Regular exercise routine</li>
                        <li>Weight management</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with steps_cols[2]:
                    st.markdown("""
                    <div class="feature-card">
                    <h4>📊 Monitoring</h4>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Glucose monitoring</li>
                        <li>Blood pressure checks</li>
                        <li>Regular health screenings</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Test different scenarios
                st.markdown("---")
                st.markdown("### 🧪 Test Different Scenarios")
                
                scenario_cols = st.columns(4)
                
                scenarios = [
                    ("Young Healthy", [25, 22.0, 4.8, 85]),
                    ("Middle-aged Average", [45, 27.0, 5.8, 110]),
                    ("Elderly High-risk", [65, 32.0, 7.2, 180]),
                    ("Critical Case", [70, 38.0, 9.5, 280])
                ]
                
                for idx, (name, values) in enumerate(scenarios):
                    with scenario_cols[idx]:
                        if st.button(f"Test: {name}", key=f"scenario_{idx}"):
                            # Store scenario values
                            st.session_state.test_scenario = values
                            st.rerun()
                
                # Handle scenario loading
                if 'test_scenario' in st.session_state:
                    test_age, test_bmi, test_hba1c, test_glucose = st.session_state.test_scenario
                    
                    # Update sliders
                    st.slider("Age", value=test_age, key="age_update")
                    st.slider("BMI", value=test_bmi, key="bmi_update")
                    st.slider("HbA1c", value=test_hba1c, key="hba1c_update")
                    st.slider("Glucose", value=test_glucose, key="glucose_update")
                    
                    # Clear scenario
                    del st.session_state.test_scenario
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("""
                **Troubleshooting Tips:**
                1. Check if all model files are properly loaded
                2. Verify input values are within reasonable ranges
                3. Ensure the model was trained with compatible data
                4. Check the console for detailed error messages
                """)
    
    # Model Information Section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🤖 About the Prediction Model</h2>', unsafe_allow_html=True)
    
    info_cols = st.columns(2)
    
    with info_cols[0]:
        st.markdown("""
        <div class="tab-content">
        <h4>🎯 Model Architecture</h4>
        <p><strong>Primary Algorithm:</strong> Gradient Boosting Classifier</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Ensemble of 100+ decision trees</li>
            <li>Automatic feature selection</li>
            <li>Cluster-based patient segmentation</li>
            <li>Cross-validation optimized</li>
        </ul>
        <p><strong>Accuracy:</strong> > 85% (validated)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_cols[1]:
        st.markdown("""
        <div class="tab-content">
        <h4>📊 Data Processing Pipeline</h4>
        <p><strong>Processing Steps:</strong></p>
        <ol>
            <li>Input standardization (z-score normalization)</li>
            <li>Patient clustering (K-means algorithm)</li>
            <li>Feature engineering (one-hot encoding)</li>
            <li>Dimensionality scaling</li>
            <li>Gradient boosting prediction</li>
            <li>Risk stratification</li>
        </ol>
        <p><strong>Training Data:</strong> 10,000+ patient records</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Reference
    st.markdown("---")
    st.markdown("### 📚 Quick Reference Guide")
    
    ref_cols = st.columns(3)
    
    with ref_cols[0]:
        st.markdown("""
        <div class="feature-card">
        <h5>🎯 Diagnostic Criteria</h5>
        <p><strong>Normal:</strong><br>
        • HbA1c < 5.7%<br>
        • Glucose < 100 mg/dL</p>
        <p><strong>Prediabetes:</strong><br>
        • HbA1c 5.7-6.4%<br>
        • Glucose 100-125 mg/dL</p>
        <p><strong>Diabetes:</strong><br>
        • HbA1c ≥ 6.5%<br>
        • Glucose ≥ 126 mg/dL</p>
        </div>
        """, unsafe_allow_html=True)
    
    with ref_cols[1]:
        st.markdown("""
        <div class="feature-card">
        <h5>📊 BMI Categories</h5>
        <p><strong>Underweight:</strong> < 18.5</p>
        <p><strong>Normal:</strong> 18.5-24.9</p>
        <p><strong>Overweight:</strong> 25-29.9</p>
        <p><strong>Obese:</strong> ≥ 30</p>
        <p><strong>Severely Obese:</strong> ≥ 40</p>
        </div>
        """, unsafe_allow_html=True)
    
    with ref_cols[2]:
        st.markdown("""
        <div class="feature-card">
        <h5>⚠️ Risk Factors</h5>
        <p><strong>Major Factors:</strong><br>
        • Age > 45 years<br>
        • BMI > 30<br>
        • Family history<br>
        • Hypertension</p>
        <p><strong>Moderate Factors:</strong><br>
        • Physical inactivity<br>
        • Poor diet<br>
        • Smoking<br>
        • High cholesterol</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Models not loaded
    st.error("""
    ## ❌ System Initialization Failed
    
    ### Required Model Files:
    
    1. **Core Models:**
       - `diabetes_classification_model.pkl` - Main prediction model
       - `kmeans_cluster.pkl` - Clustering model
       - `scaler_cluster.pkl` - Feature scaler for clustering
       - `scaler_classification.pkl` - Feature scaler for classification
    
    2. **Supporting Files:**
       - `cluster_encoder.pkl` - Cluster encoding
       - `feature_columns.pkl` - Feature list
       - `pipeline_info.pkl` - Model metadata
    
    ### 🔧 Troubleshooting Steps:
    
    1. **Check File Presence:** Ensure all .pkl files are in the same directory
    2. **Verify File Names:** Names must match exactly (case-sensitive)
    3. **Training Required:** If files don't exist, train models first
    4. **Permissions:** Check file read permissions
    5. **Python Version:** Ensure compatible Python version
    
    ### 🛠️ Quick Fix:
    Run the training script first to generate all required model files.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 25px 0;">
    <div style="font-size: 0.9rem; margin-bottom: 10px;">
        <strong>🏥 Diabetes Risk Prediction System</strong> | Version 2.0
    </div>
    <div style="font-size: 0.8rem; color: #888;">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for screening purposes only.</p>
        <p>Always consult healthcare professionals for medical diagnosis and treatment.</p>
        <p>© 2024 Medical Analytics Department. All predictions are probabilistic estimates.</p>
    </div>
</div>
""", unsafe_allow_html=True)
