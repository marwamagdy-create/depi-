
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
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #FF6B6B;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 15px;
        background-color: #FFE6E6;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 10px 0;
    }
    .risk-moderate {
        color: #FFD93D;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 15px;
        background-color: #FFF9E6;
        border-radius: 10px;
        border-left: 5px solid #FFD93D;
        margin: 10px 0;
    }
    .risk-low {
        color: #6BCF7F;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 15px;
        background-color: #E6FFEB;
        border-radius: 10px;
        border-left: 5px solid #6BCF7F;
        margin: 10px 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .feature-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .cluster-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2E86AB;
    }
    .range-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
    }
    .debug-box {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
<div class="info-box">
This system uses advanced machine learning to predict diabetes risk based on patient characteristics. 
Enter the patient information below to get an instant risk assessment.
</div>
""", unsafe_allow_html=True)

# Function to load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        models = {
            'gb_model': joblib.load('diabetes_classification_model.pkl'),
            'scaler_cluster': joblib.load('scaler_cluster.pkl'),
            'scaler_class': joblib.load('scaler_classification.pkl'),
            'cluster_encoder': joblib.load('cluster_encoder.pkl'),
            'kmeans': joblib.load('kmeans_cluster.pkl'),
            'feature_columns': joblib.load('feature_columns.pkl'),
            'clustering_features': joblib.load('clustering_features.pkl'),
            'pipeline_info': joblib.load('pipeline_info.pkl')
        }
        st.success("✅ All models loaded successfully!")
        
        # Debug: Show loaded model info
        st.sidebar.markdown("### 🐛 Debug Info")
        st.sidebar.write(f"Model type: {type(models['gb_model'])}")
        st.sidebar.write(f"Features expected: {len(models['feature_columns'])}")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load models
models = load_models()

if models:
    # Display model info in sidebar
    with st.sidebar:
        st.markdown("### 📋 Model Information")
        
        # Use safe get methods with defaults
        n_clusters = models['pipeline_info'].get('n_clusters', 5)
        diabetes_rate = models['pipeline_info'].get('diabetes_rate', 0.086)
        
        st.write(f"**Number of Clusters:** {n_clusters}")
        st.write(f"**Diabetes Rate:** {diabetes_rate:.1%}")
        
        # Show clustering features if available
        clustering_features = models['pipeline_info'].get('clustering_features', ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level'])
        st.markdown("### 🎯 Clustering Features")
        for feature in clustering_features:
            st.write(f"• {feature}")
        
        # Display normal ranges for reference
        st.markdown("### 📊 Typical Ranges")
        st.markdown("""
        **Typical Values:**
        - Age: 25-65 years
        - BMI: 18-35 kg/m²
        - HbA1c: 4-7% (normal: <5.7%)
        - Blood Glucose: 70-140 mg/dL (fasting)
        """)
        
        # Debug mode toggle
        debug_mode = st.checkbox("Debug Mode", value=False)
    
    # Main content - Prediction Form
    st.markdown('<h2 class="sub-header">📝 Patient Information Form</h2>', unsafe_allow_html=True)
    
    # Create input form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Biometric Information")
        
        age = st.slider("Age (years)", min_value=0, max_value=120, value=45, step=1, 
                       help="Patient's age in years")
        
        bmi = st.slider("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1,
                       help="Body Mass Index in kg/m²")
        
        hbA1c = st.slider("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1,
                         help="Glycated hemoglobin percentage (Normal: <5.7%, Prediabetes: 5.7-6.4%, Diabetes: ≥6.5%)")
        
        blood_glucose = st.slider("Blood Glucose (mg/dL)", min_value=50, max_value=500, value=100, step=1,
                                 help="Fasting blood glucose level (Normal: <100 mg/dL, Prediabetes: 100-125 mg/dL, Diabetes: ≥126 mg/dL)")
        
        # Display current values
        st.markdown(f"""
        <div class="range-info">
        **Current Values:**  
        • Age: {age} years  
        • BMI: {bmi:.1f} kg/m²  
        • HbA1c: {hbA1c:.1f}%  
        • Glucose: {blood_glucose} mg/dL
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Medical History")
        
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
        
        st.markdown("### 👤 Demographic Information")
        
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        
        race = st.selectbox("Race/Ethnicity", [
            "Caucasian",
            "African American", 
            "Hispanic",
            "Asian",
            "Other"
        ], index=0)
        
        smoking = st.selectbox("Smoking History", [
            "Never",
            "Former",
            "Current",
            "Not Current",
            "Ever",
            "No Info"
        ], index=0)
    
    # Prediction button
    if st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            # Display entered values for confirmation
            st.markdown(f"""
            <div class="info-box">
            <strong>📋 Patient Data Summary:</strong><br>
            • Age: {age} years | BMI: {bmi:.1f} kg/m²<br>
            • HbA1c: {hbA1c:.1f}% | Glucose: {blood_glucose} mg/dL<br>
            • Hypertension: {hypertension} | Heart Disease: {heart_disease}<br>
            • Gender: {gender} | Race: {race} | Smoking: {smoking}
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Step 1: Standardize the input values using the cluster scaler
                scaler_cluster = models['scaler_cluster']
                
                # Create numpy array of raw values
                raw_values = np.array([[age, bmi, hbA1c, blood_glucose]])
                
                # Standardize using the cluster scaler
                if debug_mode:
                    st.markdown('<div class="debug-box">🔍 <strong>Debug - Raw Values:</strong><br>'
                               f'Age: {age}, BMI: {bmi}, HbA1c: {hbA1c}, Glucose: {blood_glucose}</div>', 
                               unsafe_allow_html=True)
                
                standardized_values = scaler_cluster.transform(raw_values)
                
                age_std = float(standardized_values[0][0])
                bmi_std = float(standardized_values[0][1])
                hbA1c_std = float(standardized_values[0][2])
                blood_glucose_std = float(standardized_values[0][3])
                
                if debug_mode:
                    st.markdown('<div class="debug-box">🔍 <strong>Debug - Standardized Values:</strong><br>'
                               f'Age (z): {age_std:.3f}, BMI (z): {bmi_std:.3f}<br>'
                               f'HbA1c (z): {hbA1c_std:.3f}, Glucose (z): {blood_glucose_std:.3f}</div>', 
                               unsafe_allow_html=True)
                
                # Step 2: Clustering using standardized values
                clustering_features_array = np.array([[age_std, bmi_std, hbA1c_std, blood_glucose_std]])
                cluster = int(models['kmeans'].predict(clustering_features_array)[0])
                
                if debug_mode:
                    st.markdown(f'<div class="debug-box">🔍 <strong>Debug - Cluster Assignment:</strong> Cluster {cluster}</div>', 
                               unsafe_allow_html=True)
                
                # Step 3: Prepare input data dictionary
                gender_Female = 1 if gender == "Female" else 0
                gender_Male = 1 if gender == "Male" else 0
                
                # Race encoding
                race_encoding = {
                    "African American": "race:AfricanAmerican",
                    "Asian": "race:Asian",
                    "Caucasian": "race:Caucasian",
                    "Hispanic": "race:Hispanic",
                    "Other": "race:Other"
                }
                
                # Smoking encoding
                smoking_encoding = {
                    "Current": "smoking_history_current",
                    "Ever": "smoking_history_ever",
                    "Former": "smoking_history_former",
                    "Never": "smoking_history_never",
                    "Not Current": "smoking_history_not current",
                    "No Info": "smoking_history_No Info"
                }
                
                # Create input dictionary
                input_data = {
                    'year': 2020,
                    'age': age_std,
                    'bmi': bmi_std,
                    'hbA1c_level': hbA1c_std,
                    'blood_glucose_level': blood_glucose_std,
                    'gender_Female': gender_Female,
                    'gender_Male': gender_Male,
                    'race:AfricanAmerican': 0,
                    'race:Asian': 0,
                    'race:Caucasian': 0,
                    'race:Hispanic': 0,
                    'race:Other': 0,
                    'hypertension': 1 if hypertension == "Yes" else 0,
                    'heart_disease': 1 if heart_disease == "Yes" else 0,
                    'smoking_history_current': 0,
                    'smoking_history_ever': 0,
                    'smoking_history_former': 0,
                    'smoking_history_never': 0,
                    'smoking_history_not current': 0,
                    'smoking_history_No Info': 0,
                    'cluster': cluster
                }
                
                # Set selected race
                selected_race_key = race_encoding[race]
                input_data[selected_race_key] = 1
                
                # Set selected smoking history
                selected_smoking_key = smoking_encoding[smoking]
                input_data[selected_smoking_key] = 1
                
                if debug_mode:
                    st.markdown('<div class="debug-box">🔍 <strong>Debug - Input Dictionary Keys:</strong><br>'
                               f'{list(input_data.keys())}</div>', unsafe_allow_html=True)
                
                # Step 4: Prepare DataFrame for prediction
                new_df = pd.DataFrame([input_data])
                
                # Encode cluster
                cluster_encoded = models['cluster_encoder'].transform(new_df[['cluster']])
                cluster_encoded_df = pd.DataFrame(
                    cluster_encoded,
                    columns=models['cluster_encoder'].get_feature_names_out(['cluster']),
                    index=new_df.index
                )
                
                # Combine features
                X_new = pd.concat([
                    new_df.drop(columns=['cluster']).reset_index(drop=True),
                    cluster_encoded_df.reset_index(drop=True)
                ], axis=1)
                
                if debug_mode:
                    st.markdown(f'<div class="debug-box">🔍 <strong>Debug - DataFrame Shape:</strong> {X_new.shape}</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div class="debug-box">🔍 <strong>Debug - DataFrame Columns ({len(X_new.columns)}):</strong><br>'
                               f'{list(X_new.columns)}</div>', unsafe_allow_html=True)
                
                # Ensure all columns match expected features
                expected_features = models['feature_columns']
                missing_features = [col for col in expected_features if col not in X_new.columns]
                extra_features = [col for col in X_new.columns if col not in expected_features]
                
                if debug_mode and missing_features:
                    st.markdown(f'<div class="debug-box">⚠️ <strong>Debug - Missing Features:</strong><br>{missing_features}</div>', 
                               unsafe_allow_html=True)
                
                if debug_mode and extra_features:
                    st.markdown(f'<div class="debug-box">⚠️ <strong>Debug - Extra Features:</strong><br>{extra_features}</div>', 
                               unsafe_allow_html=True)
                
                # Add missing columns with zeros
                for col in expected_features:
                    if col not in X_new.columns:
                        X_new[col] = 0
                
                # Remove extra columns
                X_new = X_new[expected_features]
                
                # Step 5: Scale features for classification
                X_new_scaled = models['scaler_class'].transform(X_new)
                
                # Step 6: Make prediction
                proba = models['gb_model'].predict_proba(X_new_scaled)[0]
                prediction = models['gb_model'].predict(X_new_scaled)[0]
                diabetes_prob = float(proba[1])
                
                if debug_mode:
                    st.markdown(f'<div class="debug-box">🔍 <strong>Debug - Prediction Probabilities:</strong><br>'
                               f'Non-Diabetic: {proba[0]:.4f} | Diabetic: {proba[1]:.4f}</div>', 
                               unsafe_allow_html=True)
                
                # Interpret results
                if diabetes_prob >= 0.7:
                    risk_level = "HIGH RISK"
                    risk_class = "risk-high"
                    recommendation = "Immediate medical consultation recommended. Schedule appointment with endocrinologist."
                    emoji = "🔴"
                elif diabetes_prob >= 0.4:
                    risk_level = "MODERATE RISK"
                    risk_class = "risk-moderate"
                    recommendation = "Regular monitoring advised. Consider lifestyle modifications and quarterly checkups."
                    emoji = "🟡"
                else:
                    risk_level = "LOW RISK"
                    risk_class = "risk-low"
                    recommendation = "Maintain healthy lifestyle. Annual checkups recommended."
                    emoji = "🟢"
                
                # Display results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 Assessment Results</h2>', unsafe_allow_html=True)
                
                # Results in metric cards
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{emoji}</div>
                        <div class="metric-label">{risk_level}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with res_col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">{diabetes_prob:.1%}</div>
                        <div class="metric-label">DIABETES PROBABILITY</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with res_col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-value">#{cluster}</div>
                        <div class="metric-label">PATIENT CLUSTER</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Risk visualization
                st.markdown("### 📈 Risk Visualization")
                
                risk_percent = int(diabetes_prob * 100)
                st.progress(risk_percent / 100, text=f"Risk Score: {risk_percent}%")
                
                # Risk level indicators
                col_low, col_mod, col_high = st.columns(3)
                with col_low:
                    if risk_percent < 40:
                        st.success(f"### 🟢 LOW RISK\n{risk_percent}%")
                    else:
                        st.info("### LOW RISK\n< 40%")
                with col_mod:
                    if 40 <= risk_percent < 70:
                        st.warning(f"### 🟡 MODERATE RISK\n{risk_percent}%")
                    else:
                        st.info("### MODERATE RISK\n40-70%")
                with col_high:
                    if risk_percent >= 70:
                        st.error(f"### 🔴 HIGH RISK\n{risk_percent}%")
                    else:
                        st.info("### HIGH RISK\n> 70%")
                
                # Recommendation
                st.markdown(f'<div class="{risk_class}">\n### 📋 Recommendation\n{recommendation}\n</div>', unsafe_allow_html=True)
                
                # Cluster information
                st.markdown("### 🎯 Patient Cluster Information")
                
                cluster_descriptions = {
                    0: "**Average Profile**: Normal biometric ranges with minimal risk factors",
                    1: "**Elevated Risk Profile**: Above-average biometric indicators",
                    2: "**Low-Risk Profile**: Healthy biometric indicators",
                    3: "**High-Risk Profile**: Critical biometric indicators requiring attention",
                    4: "**Mixed Profile**: Variable indicators requiring monitoring"
                }
                
                if cluster in cluster_descriptions:
                    st.markdown(f'''
                    <div class="cluster-card">
                        <h4>Cluster {cluster}</h4>
                        <p>{cluster_descriptions[cluster]}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Detailed analysis
                with st.expander("📋 View Detailed Analysis"):
                    col_det1, col_det2 = st.columns(2)
                    
                    with col_det1:
                        st.markdown("#### 📊 Prediction Probabilities")
                        
                        # Create comparison
                        comparison_data = pd.DataFrame({
                            'Category': ['Non-Diabetic', 'Diabetic'],
                            'Probability': [proba[0], proba[1]]
                        })
                        
                        st.bar_chart(comparison_data.set_index('Category'))
                        
                        st.markdown(f"""
                        **Detailed Probabilities:**
                        - **Non-Diabetic Probability:** {proba[0]:.2%}
                        - **Diabetic Probability:** {proba[1]:.2%}
                        - **Prediction:** {'Diabetic' if prediction == 1 else 'Non-Diabetic'}
                        """)
                    
                    with col_det2:
                        st.markdown("#### 📋 Feature Values")
                        
                        features_data = {
                            "Feature": [
                                "Age", "BMI", "HbA1c", "Glucose",
                                "Hypertension", "Heart Disease",
                                "Gender", "Race", "Smoking", "Cluster"
                            ],
                            "Value": [
                                f"{age} years ({age_std:.2f} z)",
                                f"{bmi:.1f} kg/m² ({bmi_std:.2f} z)",
                                f"{hbA1c:.1f}% ({hbA1c_std:.2f} z)",
                                f"{blood_glucose} mg/dL ({blood_glucose_std:.2f} z)",
                                hypertension,
                                heart_disease,
                                gender,
                                race,
                                smoking,
                                f"#{cluster}"
                            ]
                        }
                        
                        features_df = pd.DataFrame(features_data)
                        st.dataframe(features_df, hide_index=True, use_container_width=True)
                
                # Test different scenarios
                st.markdown("---")
                st.markdown("### 🧪 Test Different Scenarios")
                
                test_col1, test_col2 = st.columns(2)
                
                with test_col1:
                    if st.button("Test High Risk Scenario", type="secondary"):
                        st.session_state['test_high'] = True
                        st.rerun()
                
                with test_col2:
                    if st.button("Test Low Risk Scenario", type="secondary"):
                        st.session_state['test_low'] = True
                        st.rerun()
                
                # Update session state for test scenarios
                if 'test_high' in st.session_state and st.session_state['test_high']:
                    st.session_state['test_high'] = False
                    # Set high risk values
                    st.slider("Age", value=65, key="age_high")
                    st.slider("BMI", value=35.0, key="bmi_high")
                    st.slider("HbA1c", value=8.5, key="hbA1c_high")
                    st.slider("Blood Glucose", value=250, key="glucose_high")
                
                if 'test_low' in st.session_state and st.session_state['test_low']:
                    st.session_state['test_low'] = False
                    # Set low risk values
                    st.slider("Age", value=30, key="age_low")
                    st.slider("BMI", value=22.0, key="bmi_low")
                    st.slider("HbA1c", value=5.0, key="hbA1c_low")
                    st.slider("Blood Glucose", value=90, key="glucose_low")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                if debug_mode:
                    import traceback
                    st.markdown(f'<div class="debug-box">❌ <strong>Error Details:</strong><br>{traceback.format_exc()}</div>', 
                               unsafe_allow_html=True)
    
    # Model information section
    st.markdown("---")
    
    # Quick test interface
    st.markdown("### ⚡ Quick Test")
    
    test_cols = st.columns(4)
    with test_cols[0]:
        test_age = st.number_input("Test Age", min_value=0, max_value=120, value=45, key="test_age")
    with test_cols[1]:
        test_bmi = st.number_input("Test BMI", min_value=10.0, max_value=50.0, value=25.0, key="test_bmi")
    with test_cols[2]:
        test_hba1c = st.number_input("Test HbA1c", min_value=3.0, max_value=15.0, value=5.5, key="test_hba1c")
    with test_cols[3]:
        test_glucose = st.number_input("Test Glucose", min_value=50, max_value=500, value=100, key="test_glucose")
    
    if st.button("Run Quick Test", type="secondary"):
        # Run prediction with test values
        raw_test = np.array([[test_age, test_bmi, test_hba1c, test_glucose]])
        standardized_test = models['scaler_cluster'].transform(raw_test)
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Quick Test Results:</strong><br>
        • Input: Age={test_age}, BMI={test_bmi}, HbA1c={test_hba1c}, Glucose={test_glucose}<br>
        • Standardized: Age(z)={standardized_test[0][0]:.3f}, BMI(z)={standardized_test[0][1]:.3f}, 
          HbA1c(z)={standardized_test[0][2]:.3f}, Glucose(z)={standardized_test[0][3]:.3f}<br>
        • Cluster: {models['kmeans'].predict(standardized_test)[0]}
        </div>
        """, unsafe_allow_html=True)
    
else:
    # Models not loaded - show error
    st.error("""
    ⚠️ **Models could not be loaded!**
    
    **Troubleshooting Steps:**
    1. Check that all model files are in the same directory
    2. Verify file names match exactly
    3. Try training the models again if files are missing
    4. Check file permissions
    
    **Required files:**
    - diabetes_classification_model.pkl
    - scaler_cluster.pkl  
    - scaler_classification.pkl
    - cluster_encoder.pkl
    - kmeans_cluster.pkl
    - feature_columns.pkl
    - clustering_features.pkl
    - pipeline_info.pkl
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p>🏥 <strong>Diabetes Risk Prediction System</strong></p>
    <p><small>For screening purposes only | Results may vary based on input data</small></p>
    <p><small>⚠️ This is not a medical diagnosis tool. Consult healthcare professionals.</small></p>
</div>
""", unsafe_allow_html=True)
