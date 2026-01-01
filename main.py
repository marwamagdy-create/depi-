# main.py - Diabetes Prediction System (Debug Version)
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
    layout="wide"
)

st.title("🏥 Diabetes Risk Prediction System")

# Custom CSS
st.markdown("""
<style>
.stButton>button {
    background-color: #2E86AB;
    color: white;
    font-weight: bold;
}
.info-box {
    background-color: #e8f4f8;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid #2E86AB;
}
.warning-box {
    background-color: #fff3cd;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid #ffc107;
}
.success-box {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_debug():
    """Load models with detailed debugging"""
    try:
        models = {}
        
        # Load each model with debug info
        model_files = {
            'gb_model': 'diabetes_classification_model.pkl',
            'scaler_cluster': 'scaler_cluster.pkl', 
            'scaler_class': 'scaler_classification.pkl',
            'cluster_encoder': 'cluster_encoder.pkl',
            'kmeans': 'kmeans_cluster.pkl',
            'feature_columns': 'feature_columns.pkl',
            'clustering_features': 'clustering_features.pkl',
            'pipeline_info': 'pipeline_info.pkl'
        }
        
        for key, filename in model_files.items():
            try:
                models[key] = joblib.load(filename)
                st.sidebar.success(f"✅ Loaded: {filename}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed: {filename} - {str(e)}")
                return None
        
        return models
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# Load models
models = load_models_debug()

if models:
    # Debug info in sidebar
    with st.sidebar:
        st.header("🔧 Debug Information")
        
        # Check model type
        gb_model = models['gb_model']
        st.write(f"**Model Type:** {type(gb_model).__name__}")
        
        # Check if model has predict_proba
        has_predict_proba = hasattr(gb_model, 'predict_proba')
        st.write(f"**Has predict_proba:** {has_predict_proba}")
        
        # Show scaler info
        scaler_cluster = models['scaler_cluster']
        if hasattr(scaler_cluster, 'mean_') and hasattr(scaler_cluster, 'scale_'):
            st.write("**Cluster Scaler Info:**")
            st.write(f"Means: {scaler_cluster.mean_}")
            st.write(f"Scales: {scaler_cluster.scale_}")
        
        # Feature info
        st.write(f"**Expected features:** {len(models['feature_columns'])}")
        st.write(f"**First 5 features:** {models['feature_columns'][:5]}")
    
    # Main input form
    st.header("📝 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Biometric Information")
        
        # Use wider ranges for testing
        age = st.slider("Age (years)", 0, 100, 45)
        bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 25.0, 0.1)
        hbA1c = st.slider("HbA1c (%)", 3.0, 15.0, 5.5, 0.1)
        glucose = st.slider("Blood Glucose (mg/dL)", 50, 300, 100)
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Current Values:</strong><br>
        Age: {age} years<br>
        BMI: {bmi} kg/m²<br>
        HbA1c: {hbA1c}%<br>
        Glucose: {glucose} mg/dL
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Medical & Demographic")
        
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        
        race = st.selectbox("Race/Ethnicity", [
            "Caucasian", "African American", "Hispanic", "Asian", "Other"
        ])
        
        smoking = st.selectbox("Smoking History", [
            "Never", "Former", "Current", "Not Current", "Ever", "No Info"
        ])
    
    # Manual testing section
    st.markdown("---")
    st.header("🧪 Manual Model Testing")
    
    test_cols = st.columns(3)
    
    with test_cols[0]:
        st.subheader("Test Case 1: Healthy")
        if st.button("Load Healthy Profile"):
            st.session_state.test_case = "healthy"
            st.rerun()
    
    with test_cols[1]:
        st.subheader("Test Case 2: Moderate")
        if st.button("Load Moderate Profile"):
            st.session_state.test_case = "moderate"
            st.rerun()
    
    with test_cols[2]:
        st.subheader("Test Case 3: High Risk")
        if st.button("Load High Risk Profile"):
            st.session_state.test_case = "high"
            st.rerun()
    
    # Handle test cases
    if 'test_case' in st.session_state:
        if st.session_state.test_case == "healthy":
            age, bmi, hbA1c, glucose = 30, 22.0, 5.0, 85
            hypertension, heart_disease = "No", "No"
        elif st.session_state.test_case == "moderate":
            age, bmi, hbA1c, glucose = 55, 28.0, 6.2, 135
            hypertension, heart_disease = "Yes", "No"
        elif st.session_state.test_case == "high":
            age, bmi, hbA1c, glucose = 65, 35.0, 8.5, 250
            hypertension, heart_disease = "Yes", "Yes"
        st.session_state.test_case = None
    
    if st.button("🔍 Predict Diabetes Risk", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Step 1: Display raw inputs
                st.markdown("---")
                st.header("🔍 Processing Steps")
                
                with st.expander("Step 1: Raw Inputs", expanded=True):
                    st.write(f"**Age:** {age} years")
                    st.write(f"**BMI:** {bmi} kg/m²")
                    st.write(f"**HbA1c:** {hbA1c}%")
                    st.write(f"**Glucose:** {glucose} mg/dL")
                    st.write(f"**Hypertension:** {hypertension}")
                    st.write(f"**Heart Disease:** {heart_disease}")
                    st.write(f"**Gender:** {gender}")
                    st.write(f"**Race:** {race}")
                    st.write(f"**Smoking:** {smoking}")
                
                # Step 2: Standardization
                with st.expander("Step 2: Standardization"):
                    scaler = models['scaler_cluster']
                    raw_array = np.array([[age, bmi, hbA1c, glucose]])
                    
                    # Try to get mean and scale for display
                    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                        st.write(f"**Scaler Mean:** {scaler.mean_}")
                        st.write(f"**Scaler Scale:** {scaler.scale_}")
                    
                    standardized = scaler.transform(raw_array)
                    st.write(f"**Standardized Values:**")
                    st.write(f"- Age (z-score): {standardized[0][0]:.4f}")
                    st.write(f"- BMI (z-score): {standardized[0][1]:.4f}")
                    st.write(f"- HbA1c (z-score): {standardized[0][2]:.4f}")
                    st.write(f"- Glucose (z-score): {standardized[0][3]:.4f}")
                    
                    age_std, bmi_std, hba1c_std, glucose_std = standardized[0]
                
                # Step 3: Clustering
                with st.expander("Step 3: Clustering"):
                    kmeans = models['kmeans']
                    cluster_input = np.array([[age_std, bmi_std, hba1c_std, glucose_std]])
                    cluster = kmeans.predict(cluster_input)[0]
                    
                    # Get cluster centers for comparison
                    if hasattr(kmeans, 'cluster_centers_'):
                        st.write(f"**Cluster Centers Shape:** {kmeans.cluster_centers_.shape}")
                    
                    st.write(f"**Assigned Cluster:** {cluster}")
                
                # Step 4: Prepare features
                with st.expander("Step 4: Feature Preparation"):
                    # Prepare feature dictionary
                    features = {
                        'year': 2020,
                        'age': float(age_std),
                        'bmi': float(bmi_std),
                        'hbA1c_level': float(hba1c_std),
                        'blood_glucose_level': float(glucose_std),
                        'gender_Female': 1 if gender == "Female" else 0,
                        'gender_Male': 1 if gender == "Male" else 0,
                        'race:AfricanAmerican': 1 if race == "African American" else 0,
                        'race:Asian': 1 if race == "Asian" else 0,
                        'race:Caucasian': 1 if race == "Caucasian" else 0,
                        'race:Hispanic': 1 if race == "Hispanic" else 0,
                        'race:Other': 1 if race == "Other" else 0,
                        'hypertension': 1 if hypertension == "Yes" else 0,
                        'heart_disease': 1 if heart_disease == "Yes" else 0,
                        'smoking_history_current': 1 if smoking == "Current" else 0,
                        'smoking_history_ever': 1 if smoking == "Ever" else 0,
                        'smoking_history_former': 1 if smoking == "Former" else 0,
                        'smoking_history_never': 1 if smoking == "Never" else 0,
                        'smoking_history_not current': 1 if smoking == "Not Current" else 0,
                        'smoking_history_No Info': 1 if smoking == "No Info" else 0,
                        'cluster': int(cluster)
                    }
                    
                    st.write("**Created Features Dictionary**")
                    st.write(f"Number of features: {len(features)}")
                    
                    # Create DataFrame
                    df = pd.DataFrame([features])
                    
                    # Encode cluster
                    encoder = models['cluster_encoder']
                    cluster_encoded = encoder.transform(df[['cluster']])
                    
                    # Get cluster column names
                    cluster_cols = encoder.get_feature_names_out(['cluster'])
                    
                    # Create cluster DataFrame
                    cluster_df = pd.DataFrame(
                        cluster_encoded,
                        columns=cluster_cols,
                        index=df.index
                    )
                    
                    # Combine
                    X_combined = pd.concat([
                        df.drop(columns=['cluster']).reset_index(drop=True),
                        cluster_df.reset_index(drop=True)
                    ], axis=1)
                    
                    st.write(f"**Combined DataFrame shape:** {X_combined.shape}")
                    
                    # Ensure all expected columns exist
                    expected_cols = models['feature_columns']
                    
                    st.write(f"**Expected columns:** {len(expected_cols)}")
                    st.write(f"**Our columns:** {len(X_combined.columns)}")
                    
                    # Find missing/extra columns
                    missing = set(expected_cols) - set(X_combined.columns)
                    extra = set(X_combined.columns) - set(expected_cols)
                    
                    if missing:
                        st.warning(f"Missing columns: {len(missing)}")
                        for col in list(missing)[:5]:
                            st.write(f"  - {col}")
                    
                    if extra:
                        st.warning(f"Extra columns: {len(extra)}")
                        for col in list(extra)[:5]:
                            st.write(f"  - {col}")
                    
                    # Add missing columns with zeros
                    for col in expected_cols:
                        if col not in X_combined.columns:
                            X_combined[col] = 0
                    
                    # Select only expected columns
                    X_final = X_combined[expected_cols]
                    st.write(f"**Final DataFrame shape:** {X_final.shape}")
                
                # Step 5: Scale for classification
                with st.expander("Step 5: Classification Scaling"):
                    scaler_class = models['scaler_class']
                    X_scaled = scaler_class.transform(X_final)
                    
                    # Check first few values
                    st.write("**First 5 scaled values:**")
                    st.write(X_scaled[0, :5])
                
                # Step 6: Prediction
                with st.expander("Step 6: Model Prediction", expanded=True):
                    model = models['gb_model']
                    
                    # Try different prediction methods
                    try:
                        # Method 1: predict_proba
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(X_scaled)[0]
                            st.write(f"**Probabilities (predict_proba):**")
                            st.write(f"- Class 0 (Non-diabetic): {probabilities[0]:.6f}")
                            st.write(f"- Class 1 (Diabetic): {probabilities[1]:.6f}")
                            diabetes_prob = probabilities[1]
                        else:
                            st.warning("Model doesn't have predict_proba method")
                            diabetes_prob = 0.5
                        
                        # Method 2: predict
                        prediction = model.predict(X_scaled)[0]
                        st.write(f"**Binary Prediction:** {prediction} ({'Diabetic' if prediction == 1 else 'Non-diabetic'})")
                        
                        # Method 3: decision_function if available
                        if hasattr(model, 'decision_function'):
                            decision_scores = model.decision_function(X_scaled)
                            st.write(f"**Decision Scores:** {decision_scores[0]:.6f}")
                        
                    except Exception as pred_error:
                        st.error(f"Prediction error: {str(pred_error)}")
                        # Try direct prediction as fallback
                        try:
                            prediction = model.predict(X_scaled)[0]
                            st.write(f"**Direct Prediction:** {prediction}")
                            diabetes_prob = float(prediction)
                        except:
                            diabetes_prob = 0.5
                
                # Step 7: Results Display
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                # Create metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Diabetes Probability", f"{diabetes_prob:.1%}")
                
                with col2:
                    risk_level = "HIGH" if diabetes_prob > 0.7 else "MODERATE" if diabetes_prob > 0.4 else "LOW"
                    st.metric("Risk Level", risk_level)
                
                with col3:
                    st.metric("Patient Cluster", f"#{cluster}")
                
                # Risk interpretation
                st.markdown("### 📋 Risk Interpretation")
                
                if diabetes_prob > 0.7:
                    st.markdown("""
                    <div class="warning-box">
                    <h4>🔴 HIGH RISK DETECTED</h4>
                    <p><strong>Probability:</strong> {:.1%}</p>
                    <p><strong>Recommendation:</strong> Immediate medical consultation recommended. 
                    Please schedule an appointment with your healthcare provider.</p>
                    </div>
                    """.format(diabetes_prob), unsafe_allow_html=True)
                elif diabetes_prob > 0.4:
                    st.markdown("""
                    <div class="info-box">
                    <h4>🟡 MODERATE RISK</h4>
                    <p><strong>Probability:</strong> {:.1%}</p>
                    <p><strong>Recommendation:</strong> Regular monitoring advised. 
                    Consider lifestyle modifications and schedule a checkup within 3 months.</p>
                    </div>
                    """.format(diabetes_prob), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                    <h4>🟢 LOW RISK</h4>
                    <p><strong>Probability:</strong> {:.1%}</p>
                    <p><strong>Recommendation:</strong> Maintain healthy lifestyle. 
                    Annual checkups are recommended for ongoing monitoring.</p>
                    </div>
                    """.format(diabetes_prob), unsafe_allow_html=True)
                
                # Model verification
                st.markdown("---")
                st.header("🔬 Model Verification")
                
                # Test with extreme values
                test_cases = [
                    ("Very Healthy", [25, 20.0, 4.5, 80]),
                    ("Very High Risk", [70, 40.0, 9.0, 300]),
                    ("Average", [45, 25.0, 5.5, 100])
                ]
                
                results = []
                for name, values in test_cases:
                    test_array = np.array([values])
                    test_scaled = scaler.transform(test_array)
                    test_cluster = kmeans.predict(test_scaled)[0]
                    
                    # Quick probability estimate
                    results.append({
                        "Case": name,
                        "Age": values[0],
                        "BMI": values[1],
                        "HbA1c": values[2],
                        "Glucose": values[3],
                        "Cluster": test_cluster
                    })
                
                st.dataframe(pd.DataFrame(results))
                
                # Check if probabilities vary
                st.write("**Note:** If all probabilities are similar, the model may not be learning properly.")
                
            except Exception as e:
                st.error(f"Error in prediction pipeline: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Direct model test
    st.markdown("---")
    st.header("⚡ Direct Model Test")
    
    direct_test_cols = st.columns(2)
    
    with direct_test_cols[0]:
        test_values = st.text_input("Enter test values (age,bmi,hba1c,glucose):", "45,25,5.5,100")
    
    with direct_test_cols[1]:
        if st.button("Run Direct Test"):
            try:
                values = [float(x.strip()) for x in test_values.split(",")]
                if len(values) == 4:
                    # Standardize
                    test_array = np.array([values])
                    standardized = scaler.transform(test_array)
                    
                    # Cluster
                    cluster = kmeans.predict(standardized)[0]
                    
                    # Simple prediction (without full feature engineering)
                    # Just to see if standardization works
                    st.write(f"**Test Results:**")
                    st.write(f"Input: {values}")
                    st.write(f"Standardized: {standardized[0].tolist()}")
                    st.write(f"Cluster: {cluster}")
                    
                    # Try to predict with just these 4 features
                    simple_features = standardized
                    try:
                        simple_pred = model.predict(simple_features)[0]
                        st.write(f"Simple Prediction: {simple_pred}")
                    except:
                        st.write("Full prediction requires all features")
                else:
                    st.error("Please enter 4 values separated by commas")
            except Exception as e:
                st.error(f"Direct test error: {str(e)}")
    
else:
    st.error("""
    ## ❌ Models Not Loaded
    
    Please check:
    1. All .pkl files are in the same directory as this app
    2. File names are correct
    3. You have trained the models first
    
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
st.markdown("*For demonstration purposes only. Always consult healthcare professionals for medical advice.*")
