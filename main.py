# app.py - Diabetes Prediction System (Prediction Only)
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
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Load models
models = load_models()

if models:
    # Display model info in sidebar
    with st.sidebar:
        st.success("✅ Models loaded successfully!")
        
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
        st.markdown("### 📊 Typical Ranges (Original Data)")
        st.markdown("""
        **Age:** 20-80 years  
        **BMI:** 15-50 kg/m²  
        **HbA1c:** 3.5-9%  
        **Blood Glucose:** 70-300 mg/dL
        """)
    
    # Main content - Prediction Form
    st.markdown('<h2 class="sub-header">📝 Patient Information Form</h2>', unsafe_allow_html=True)
    
    # Create input form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Biometric Information")
        st.info("Enter actual patient values (not standardized)")
        
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0, 
                             help="Patient's age in years")
        
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                             help="Body Mass Index in kg/m²")
        
        hbA1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1,
                               help="Glycated hemoglobin percentage")
        
        blood_glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50.0, max_value=500.0, value=100.0, step=1.0,
                                       help="Fasting blood glucose level in mg/dL")
        
        # Display input values
        st.markdown(f"""
        <div class="range-info">
        **Entered Values:**  
        Age: {age} years | BMI: {bmi} kg/m² | HbA1c: {hbA1c}% | Glucose: {blood_glucose} mg/dL
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🏥 Medical History")
        
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        
        st.markdown("### 👤 Demographic Information")
        
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race/Ethnicity", [
            "African American",
            "Asian",
            "Caucasian",
            "Hispanic",
            "Other"
        ])
        
        smoking = st.selectbox("Smoking History", [
            "Never",
            "Former",
            "Current",
            "Not Current",
            "Ever",
            "No Info"
        ])
    
    # Prediction button
    if st.button("🔍 Assess Diabetes Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing patient data..."):
            # Display entered values for confirmation
            st.markdown(f"""
            <div class="info-box">
            <strong>Patient Data Summary:</strong><br>
            Age: {age} years | BMI: {bmi} kg/m² | HbA1c: {hbA1c}% | Glucose: {blood_glucose} mg/dL<br>
            Hypertension: {hypertension} | Heart Disease: {heart_disease} | Gender: {gender}<br>
            Race: {race} | Smoking: {smoking}
            </div>
            """, unsafe_allow_html=True)
            
            # Step 1: Standardize the input values
            # Get the training data statistics if available
            try:
                # If you have saved the training data mean and std, use them
                # Otherwise, use typical ranges (you should replace these with actual values from your training)
                
                # Get scaler parameters
                scaler_cluster = models['scaler_cluster']
                
                # Create numpy array of raw values
                raw_values = np.array([[age, bmi, hbA1c, blood_glucose]])
                
                # Standardize using the cluster scaler
                standardized_values = scaler_cluster.transform(raw_values)
                
                age_std = standardized_values[0][0]
                bmi_std = standardized_values[0][1]
                hbA1c_std = standardized_values[0][2]
                blood_glucose_std = standardized_values[0][3]
                
                # Display standardized values for debugging/transparency
                st.markdown(f"""
                <div class="info-box" style="background-color: #f0f0f0;">
                <strong>Standardized Values (Internal):</strong><br>
                Age (z-score): {age_std:.2f} | BMI (z-score): {bmi_std:.2f}<br>
                HbA1c (z-score): {hbA1c_std:.2f} | Glucose (z-score): {blood_glucose_std:.2f}<br>
                <small>These standardized values are used internally by the model.</small>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error standardizing values: {str(e)}")
                # Fallback: use manual standardization (replace with your actual training data stats)
                age_mean, age_std_dev = 42.0, 22.5  # Replace with actual values
                bmi_mean, bmi_std_dev = 27.3, 6.7    # Replace with actual values
                hbA1c_mean, hbA1c_std_dev = 5.5, 1.5  # Replace with actual values
                glucose_mean, glucose_std_dev = 138.0, 40.0  # Replace with actual values
                
                age_std = (age - age_mean) / age_std_dev
                bmi_std = (bmi - bmi_mean) / bmi_std_dev
                hbA1c_std = (hbA1c - hbA1c_mean) / hbA1c_std_dev
                blood_glucose_std = (blood_glucose - glucose_mean) / glucose_std_dev
            
            # Prepare input data for classification
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
            
            # Create input dictionary with STANDARDIZED values
            input_data = {
                'year': 2020,
                'age': float(age_std),  # Use standardized age
                'bmi': float(bmi_std),  # Use standardized BMI
                'hbA1c_level': float(hbA1c_std),  # Use standardized HbA1c
                'blood_glucose_level': float(blood_glucose_std),  # Use standardized glucose
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
            }
            
            # Set selected race
            input_data[race_encoding[race]] = 1
            
            # Set selected smoking history
            input_data[smoking_encoding[smoking]] = 1
            
            # Step 2: Clustering using standardized values
            clustering_features_array = np.array([[age_std, bmi_std, hbA1c_std, blood_glucose_std]])
            cluster = models['kmeans'].predict(clustering_features_array)[0]
            input_data['cluster'] = cluster
            
            # Step 3: Prepare for classification
            new_df = pd.DataFrame([input_data])
            cluster_encoded = models['cluster_encoder'].transform(new_df[['cluster']])
            cluster_encoded_df = pd.DataFrame(
                cluster_encoded,
                columns=models['cluster_encoder'].get_feature_names_out(['cluster']),
                index=new_df.index
            )
            
            X_new = pd.concat([
                new_df.drop(columns=['cluster']).reset_index(drop=True),
                cluster_encoded_df.reset_index(drop=True)
            ], axis=1)
            
            # Ensure all columns match
            for col in models['feature_columns']:
                if col not in X_new.columns:
                    X_new[col] = 0
            
            X_new = X_new[models['feature_columns']]
            
            # Scale and predict
            X_new_scaled = models['scaler_class'].transform(X_new)
            proba = models['gb_model'].predict_proba(X_new_scaled)[0]
            prediction = models['gb_model'].predict(X_new_scaled)[0]
            diabetes_prob = proba[1]
            
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
                0: "**High-Risk Profile**: Elevated biometric indicators, often with comorbidities",
                1: "**Moderate-Risk Profile**: Mixed biometric profile, requires monitoring",
                2: "**Low-Risk Profile**: Below-average indicators, healthy profile",
                3: "**Very High-Risk Profile**: Critical indicators, urgent attention needed",
                4: "**Average-Risk Profile**: Normal biometric ranges"
            }
            
            if cluster in cluster_descriptions:
                st.markdown(f'''
                <div class="cluster-card">
                    <h4>Cluster {cluster}</h4>
                    <p>{cluster_descriptions[cluster]}</p>
                    <small>Based on standardized biometric values</small>
                </div>
                ''', unsafe_allow_html=True)
            
            # Detailed analysis
            with st.expander("📋 View Detailed Analysis"):
                col_det1, col_det2 = st.columns(2)
                
                with col_det1:
                    st.markdown("#### 📊 Prediction Probabilities")
                    prob_non_diabetic = proba[0]
                    prob_diabetic = proba[1]
                    
                    # Create a simple bar chart using columns
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        height_non = int(prob_non_diabetic * 100)
                        st.markdown(f'''
                        <div style="text-align: center; margin: 10px 0;">
                            <div style="font-size: 2rem; color: #4CAF50; font-weight: bold;">{prob_non_diabetic:.1%}</div>
                            <div style="background-color: #4CAF50; height: {height_non}px; width: 100%; border-radius: 5px;"></div>
                            <div style="margin-top: 5px;">Non-Diabetic</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with prob_col2:
                        height_diab = int(prob_diabetic * 100)
                        st.markdown(f'''
                        <div style="text-align: center; margin: 10px 0;">
                            <div style="font-size: 2rem; color: #FF6B6B; font-weight: bold;">{prob_diabetic:.1%}</div>
                            <div style="background-color: #FF6B6B; height: {height_diab}px; width: 100%; border-radius: 5px;"></div>
                            <div style="margin-top: 5px;">Diabetic</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with col_det2:
                    st.markdown("#### 📋 Input Summary")
                    
                    # Display both raw and standardized values
                    features = [
                        ("Age (years)", f"{age}"),
                        ("Age (z-score)", f"{age_std:.2f}"),
                        ("BMI (kg/m²)", f"{bmi}"),
                        ("BMI (z-score)", f"{bmi_std:.2f}"),
                        ("HbA1c (%)", f"{hbA1c}"),
                        ("HbA1c (z-score)", f"{hbA1c_std:.2f}"),
                        ("Glucose (mg/dL)", f"{blood_glucose}"),
                        ("Glucose (z-score)", f"{blood_glucose_std:.2f}"),
                        ("Hypertension", hypertension),
                        ("Heart Disease", heart_disease),
                        ("Gender", gender),
                        ("Race", race),
                        ("Smoking", smoking),
                        ("Cluster", f"#{cluster}")
                    ]
                    
                    for feature_name, feature_value in features:
                        st.markdown(f'''
                        <div class="feature-card">
                            <div style="font-weight: bold; color: #2E86AB;">{feature_name}</div>
                            <div style="font-size: 1.1rem;">{feature_value}</div>
                        </div>
                        ''', unsafe_allow_html=True)
            
            # Action steps
            st.markdown("### 🚀 Next Steps")
            
            if risk_level == "HIGH RISK":
                st.markdown("""
                **Immediate Actions:**
                1. 🏥 Schedule appointment with healthcare provider
                2. 🩸 Get comprehensive blood tests
                3. 🍎 Consult with nutritionist
                4. 🏃‍♂️ Start supervised exercise program
                5. 📊 Regular glucose monitoring
                """)
            elif risk_level == "MODERATE RISK":
                st.markdown("""
                **Recommended Actions:**
                1. 🩺 Schedule doctor's appointment within 1 month
                2. 📈 Monitor indicators quarterly
                3. 🥗 Adopt healthy diet
                4. 🚶‍♂️ Increase physical activity
                5. ⚖️ Manage weight effectively
                """)
            else:
                st.markdown("""
                **Maintenance Actions:**
                1. 🩺 Annual health checkup
                2. 🥗 Maintain balanced diet
                3. 🏃‍♂️ Regular physical activity
                4. 😴 Adequate sleep and stress management
                5. 📊 Periodic self-assessment
                """)
    
    # Model information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        ### 🎯 Model Architecture
        
        **Algorithm:** Gradient Boosting Classifier
        
        **Key Components:**
        - 100 Decision Trees
        - Maximum Depth: 3 levels
        - Learning Rate: 0.1
        
        **Features Analyzed:**
        - Biometric indicators (Age, BMI, HbA1c, Glucose)
        - Medical history (Hypertension, Heart Disease)
        - Demographic factors (Gender, Race, Smoking)
        - Patient cluster membership
        """)
    
    with col_info2:
        st.markdown("""
        ### 📊 How It Works
        
        1. **Input Processing**
           - Accepts actual patient values (years, kg/m², %, mg/dL)
           - Automatically standardizes to z-scores
        
        2. **Clustering Phase**
           - Patients grouped into clusters based on standardized biometrics
        
        3. **Feature Engineering**
           - Cluster assignment as predictive feature
           - One-hot encoding for categorical variables
        
        4. **Prediction Phase**
           - Gradient Boosting analyzes all features
           - Generates probability scores
        
        5. **Risk Stratification**
           - Low Risk: < 40% probability
           - Moderate Risk: 40-70% probability
           - High Risk: > 70% probability
        """)
    
else:
    # Models not loaded - show error
    st.error("""
    ⚠️ **Models could not be loaded!**
    
    **Required model files (8 files total):**
    
    1. `diabetes_classification_model.pkl`
    2. `scaler_cluster.pkl`
    3. `scaler_classification.pkl`
    4. `cluster_encoder.pkl`
    5. `kmeans_cluster.pkl`
    6. `feature_columns.pkl`
    7. `clustering_features.pkl`
    8. `pipeline_info.pkl`
    
    **To fix this:**
    1. Make sure all model files are in the same directory as this app
    2. Check that the file names match exactly
    3. Verify file permissions
    4. If files don't exist, you need to train the models first using a separate training script
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p>🏥 <strong>Diabetes Risk Prediction System</strong></p>
    <p><small>Developed for Healthcare Analytics | For screening purposes only</small></p>
    <p><small>⚠️ Always consult healthcare professionals for medical diagnosis</small></p>
</div>
""", unsafe_allow_html=True)
