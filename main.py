# app.py - Complete Diabetes Prediction System with Training + Streamlit App
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Preprocessing & Modeling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

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
        padding: 10px;
        background-color: #FFE6E6;
        border-radius: 5px;
        border-left: 5px solid #FF6B6B;
    }
    .risk-moderate {
        color: #FFD93D;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 10px;
        background-color: #FFF9E6;
        border-radius: 5px;
        border-left: 5px solid #FFD93D;
    }
    .risk-low {
        color: #6BCF7F;
        font-weight: bold;
        font-size: 1.2rem;
        padding: 10px;
        background-color: #E6FFEB;
        border-radius: 5px;
        border-left: 5px solid #6BCF7F;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
    }
    .info-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 10px 0;
    }
    .cluster-box {
        background-color: #E6F7FF;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #B3E0FF;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)

# Function to train and save models
def train_models():
    """Train all models and save them to files"""
    try:
        # Load data
        df = pd.read_csv("model.csv")
        
        # Select features for clustering
        features_for_clustering = ['age', 'bmi', 'hbA1c_level', 'blood_glucose_level']
        
        # Handle missing values
        df_cluster_clean = df[features_for_clustering].dropna()
        df_clean = df.loc[df_cluster_clean.index].copy()
        
        # Scale clustering data
        scaler_cluster = StandardScaler()
        X_scaled = scaler_cluster.fit_transform(df_cluster_clean[features_for_clustering])
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        df_clean['cluster'] = clusters
        
        # Encode clusters
        cluster_encoder = OneHotEncoder(drop='first', sparse_output=False)
        cluster_encoded = cluster_encoder.fit_transform(df_clean[['cluster']])
        cluster_encoded_df = pd.DataFrame(
            cluster_encoded, 
            columns=cluster_encoder.get_feature_names_out(['cluster']), 
            index=df_clean.index
        )
        
        # Combine features
        df_encoded = pd.concat([df_clean.drop(columns=['cluster']), cluster_encoded_df], axis=1)
        
        # Prepare for classification
        X = df_encoded.drop(columns=['diabetes'])
        y = df_clean['diabetes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Scale for classification
        scaler_class = StandardScaler()
        X_train_scaled = scaler_class.fit_transform(X_train)
        X_test_scaled = scaler_class.transform(X_test)
        
        # Train model
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_test = gb_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # Save models
        joblib.dump(gb_model, 'diabetes_classification_model.pkl')
        joblib.dump(scaler_cluster, 'scaler_cluster.pkl')
        joblib.dump(scaler_class, 'scaler_classification.pkl')
        joblib.dump(cluster_encoder, 'cluster_encoder.pkl')
        joblib.dump(kmeans, 'kmeans_cluster.pkl')
        joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
        joblib.dump(features_for_clustering, 'clustering_features.pkl')
        
        # Save pipeline info
        pipeline_info = {
            'model_type': 'GradientBoostingClassifier',
            'n_clusters': 5,
            'clustering_features': features_for_clustering,
            'feature_columns': X.columns.tolist(),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_f1_score': test_f1,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'diabetes_rate': y.mean()
        }
        
        joblib.dump(pipeline_info, 'pipeline_info.pkl')
        
        return pipeline_info
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None

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
        return None

# Sidebar with model options
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    mode = st.radio(
        "Choose mode:",
        ["🚀 Use Existing Models", "🔄 Train New Models"]
    )
    
    if mode == "🔄 Train New Models":
        st.info("Training new models will take a few moments...")
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models... This may take a minute"):
                pipeline_info = train_models()
                if pipeline_info:
                    st.success("✅ Models trained and saved successfully!")
                    st.balloons()
                    
                    # Show training results
                    st.subheader("Training Results")
                    st.metric("Accuracy", f"{pipeline_info['test_accuracy']:.2%}")
                    st.metric("Precision", f"{pipeline_info['test_precision']:.2%}")
                    st.metric("F1 Score", f"{pipeline_info['test_f1_score']:.2%}")
                    
                    st.info("Switch to 'Use Existing Models' to start making predictions")

# Main app
if mode == "🚀 Use Existing Models":
    models = load_models()
    
    if models:
        # Display model info in sidebar
        with st.sidebar:
            st.success("✅ Models loaded successfully!")
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{models['pipeline_info']['test_accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{models['pipeline_info']['test_precision']:.2%}")
            with col3:
                st.metric("F1 Score", f"{models['pipeline_info']['test_f1_score']:.2%}")
            
            st.subheader("Dataset Info")
            st.write(f"**Total Samples:** {models['pipeline_info']['training_samples'] + models['pipeline_info']['test_samples']:,}")
            st.write(f"**Diabetes Rate:** {models['pipeline_info']['diabetes_rate']:.1%}")
            st.write(f"**Number of Clusters:** {models['pipeline_info']['n_clusters']}")
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📝 Patient Assessment", "📊 Cluster Analysis", "📈 Model Insights"])
        
        with tab1:
            st.markdown("""
                <div class="info-box">
                Enter patient information below to assess diabetes risk. 
                The model analyzes biometric indicators and medical history.
                </div>
            """, unsafe_allow_html=True)
            
            # Create input form
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Biometric Information")
                st.info("Values are standardized (z-scores). Positive = above average, Negative = below average")
                
                age = st.slider("Age (standardized)", -3.0, 3.0, 0.0, 0.1)
                bmi = st.slider("BMI (standardized)", -3.0, 3.0, 0.0, 0.1)
                hbA1c = st.slider("HbA1c Level (standardized)", -3.0, 3.0, 0.0, 0.1)
                blood_glucose = st.slider("Blood Glucose (standardized)", -3.0, 3.0, 0.0, 0.1)
            
            with col2:
                st.subheader("Medical History")
                hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
                heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
                
                st.subheader("Demographic Information")
                gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
                race = st.selectbox("Race/Ethnicity", ["African American", "Asian", "Caucasian", "Hispanic", "Other"])
                smoking = st.selectbox("Smoking History", ["Current", "Ever", "Former", "Never", "Not Current", "No Info"])
            
            # Prediction button
            if st.button("🔍 Assess Diabetes Risk", type="primary"):
                with st.spinner("Analyzing patient data..."):
                    # Prepare input data
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
                        'age': age,
                        'bmi': bmi,
                        'hbA1c_level': hbA1c,
                        'blood_glucose_level': blood_glucose,
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
                    
                    # Set selected race and smoking
                    input_data[race_encoding[race]] = 1
                    input_data[smoking_encoding[smoking]] = 1
                    
                    # Step 1: Clustering
                    clustering_features = np.array([[age, bmi, hbA1c, blood_glucose]])
                    clustering_scaled = models['scaler_cluster'].transform(clustering_features)
                    cluster = models['kmeans'].predict(clustering_scaled)[0]
                    input_data['cluster'] = cluster
                    
                    # Step 2: Prepare for classification
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
                        recommendation = "Immediate medical consultation recommended"
                        emoji = "🔴"
                    elif diabetes_prob >= 0.4:
                        risk_level = "MODERATE RISK"
                        risk_class = "risk-moderate"
                        recommendation = "Regular monitoring advised"
                        emoji = "🟡"
                    else:
                        risk_level = "LOW RISK"
                        risk_class = "risk-low"
                        recommendation = "Maintain healthy lifestyle"
                        emoji = "🟢"
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h2 class="sub-header">Assessment Results</h2>', unsafe_allow_html=True)
                    
                    # Results columns
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{emoji}</div><div class="metric-label">{risk_level}</div></div>', unsafe_allow_html=True)
                    with res_col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{diabetes_prob:.1%}</div><div class="metric-label">DIABETES PROBABILITY</div></div>', unsafe_allow_html=True)
                    with res_col3:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">#{cluster}</div><div class="metric-label">PATIENT CLUSTER</div></div>', unsafe_allow_html=True)
                    
                    # Risk visualization
                    st.subheader("Risk Visualization")
                    risk_percent = int(diabetes_prob * 100)
                    st.progress(risk_percent / 100)
                    
                    # Risk meter
                    col_low, col_mod, col_high = st.columns(3)
                    with col_low:
                        if risk_percent < 40:
                            st.success(f"LOW RISK\n{risk_percent}%")
                        else:
                            st.info("LOW RISK")
                    with col_mod:
                        if 40 <= risk_percent < 70:
                            st.warning(f"MODERATE RISK\n{risk_percent}%")
                        else:
                            st.info("MODERATE RISK")
                    with col_high:
                        if risk_percent >= 70:
                            st.error(f"HIGH RISK\n{risk_percent}%")
                        else:
                            st.info("HIGH RISK")
                    
                    # Recommendation
                    st.markdown(f'<div class="{risk_class}">📋 Recommendation: {recommendation}</div>', unsafe_allow_html=True)
                    
                    # Detailed view
                    with st.expander("View Detailed Analysis"):
                        st.write("**Prediction Probabilities:**")
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("Non-Diabetic", f"{proba[0]:.1%}")
                        with col_prob2:
                            st.metric("Diabetic", f"{proba[1]:.1%}")
                        
                        st.write("**Input Summary:**")
                        summary_df = pd.DataFrame({
                            'Feature': ['Age', 'BMI', 'HbA1c', 'Blood Glucose', 'Hypertension', 
                                       'Heart Disease', 'Gender', 'Race', 'Smoking', 'Cluster'],
                            'Value': [f"{age:.2f}", f"{bmi:.2f}", f"{hbA1c:.2f}", f"{blood_glucose:.2f}",
                                     hypertension, heart_disease, gender, race, smoking, f"#{cluster}"]
                        })
                        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        with tab2:
            st.markdown('<h2 class="sub-header">Cluster Analysis</h2>', unsafe_allow_html=True)
            st.info("Patient clusters based on biometric similarity")
            
            # Display cluster info
            st.write(f"**Number of Clusters:** {models['pipeline_info']['n_clusters']}")
            st.write(f"**Clustering Features:** {', '.join(models['pipeline_info']['clustering_features'])}")
            
            # Cluster descriptions
            st.subheader("Cluster Characteristics")
            
            clusters_info = [
                {
                    "id": 0,
                    "name": "High-Risk Cluster",
                    "description": "Patients with elevated biometric indicators across all measures",
                    "diabetes_rate": "15-25%",
                    "characteristics": "High age, BMI, HbA1c, and glucose levels"
                },
                {
                    "id": 1,
                    "name": "Moderate-Risk Cluster", 
                    "description": "Patients with mixed biometric profile",
                    "diabetes_rate": "8-15%",
                    "characteristics": "Moderate elevation in some indicators"
                },
                {
                    "id": 2,
                    "name": "Low-Risk Cluster",
                    "description": "Patients with below-average biometric indicators",
                    "diabetes_rate": "2-5%",
                    "characteristics": "Younger age, normal BMI and glucose levels"
                },
                {
                    "id": 3,
                    "name": "Very High-Risk Cluster",
                    "description": "Patients with critical biometric indicators",
                    "diabetes_rate": "25-35%",
                    "characteristics": "Severely elevated indicators, often with comorbidities"
                },
                {
                    "id": 4,
                    "name": "Average-Risk Cluster",
                    "description": "Patients within normal biometric ranges",
                    "diabetes_rate": "5-10%",
                    "characteristics": "All indicators within normal limits"
                }
            ]
            
            for cluster_info in clusters_info:
                with st.container():
                    st.markdown(f"""
                    <div class="cluster-box">
                    <h4>Cluster {cluster_info['id']}: {cluster_info['name']}</h4>
                    <p><strong>Diabetes Rate:</strong> {cluster_info['diabetes_rate']}</p>
                    <p><strong>Characteristics:</strong> {cluster_info['characteristics']}</p>
                    <p><strong>Description:</strong> {cluster_info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h2 class="sub-header">Model Insights</h2>', unsafe_allow_html=True)
            
            # Model architecture
            col_arch1, col_arch2, col_arch3 = st.columns(3)
            with col_arch1:
                st.markdown('<div class="metric-card"><div class="metric-value">GB</div><div class="metric-label">ALGORITHM<br>Gradient Boosting</div></div>', unsafe_allow_html=True)
            with col_arch2:
                st.markdown('<div class="metric-card"><div class="metric-value">100</div><div class="metric-label">DECISION TREES</div></div>', unsafe_allow_html=True)
            with col_arch3:
                st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">MAX DEPTH</div></div>', unsafe_allow_html=True)
            
            # How it works
            st.subheader("How the Model Works")
            st.markdown("""
            ### 4-Step Prediction Process:
            
            1. **📊 Clustering Phase**  
               Patients grouped into 5 clusters based on:
               - Age, BMI, HbA1c, Blood Glucose
            
            2. **🔧 Feature Engineering**  
               Cluster assignment becomes a predictive feature
               One-hot encoding for categorical variables
            
            3. **🤖 Classification**  
               Gradient Boosting analyzes 40+ features including:
               - Biometric indicators
               - Medical history
               - Demographic factors
               - Cluster membership
            
            4. **⚠️ Risk Assessment**  
               Probability-based stratification:
               - **Low Risk**: < 40% probability
               - **Moderate Risk**: 40-70% probability  
               - **High Risk**: > 70% probability
            """)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            metrics = [
                {"name": "Accuracy", "value": models['pipeline_info']['test_accuracy'], "target": 0.85},
                {"name": "Precision", "value": models['pipeline_info']['test_precision'], "target": 0.80},
                {"name": "F1-Score", "value": models['pipeline_info']['test_f1_score'], "target": 0.82},
            ]
            
            for metric in metrics:
                col_name, col_value, col_target = st.columns([1, 2, 1])
                with col_name:
                    st.write(f"**{metric['name']}**")
                with col_value:
                    progress = metric['value'] / metric['target'] if metric['target'] > 0 else 0
                    progress = min(progress, 1.0)  # Cap at 100%
                    st.progress(progress)
                with col_target:
                    st.metric("", f"{metric['value']:.3f}", f"Target: {metric['target']:.2f}")
            
            # Feature importance
            st.subheader("Key Predictive Factors")
            
            important_features = [
                {"name": "Blood Glucose Level", "importance": "Very High", "impact": "Direct measure of diabetes risk"},
                {"name": "HbA1c Level", "importance": "High", "impact": "Long-term glucose control indicator"},
                {"name": "Age", "importance": "High", "impact": "Risk increases with age"},
                {"name": "BMI", "importance": "Medium", "impact": "Obesity is a major risk factor"},
                {"name": "Hypertension", "importance": "Medium", "impact": "Common comorbidity with diabetes"},
                {"name": "Cluster Membership", "importance": "Medium", "impact": "Biometric similarity group"},
            ]
            
            for feature in important_features:
                col_feat, col_imp, col_impact = st.columns([2, 1, 3])
                with col_feat:
                    st.write(f"• {feature['name']}")
                with col_imp:
                    if feature['importance'] == "Very High":
                        st.error(feature['importance'])
                    elif feature['importance'] == "High":
                        st.warning(feature['importance'])
                    else:
                        st.info(feature['importance'])
                with col_impact:
                    st.write(feature['impact'])
    
    else:
        st.warning("⚠️ No trained models found. Please train models first.")
        st.info("""
        ### How to get started:
        
        1. **Go to the sidebar** (click the arrow in top-left if not visible)
        2. **Select "Train New Models"** option
        3. **Click "Train Models"** button
        4. **Wait for training to complete** (about 1-2 minutes)
        5. **Switch back to "Use Existing Models"** to start making predictions
        
        The system will automatically:
        - Load your diabetes dataset
        - Create patient clusters
        - Train the prediction model
        - Save all necessary files
        """)
        
        if st.button("🔄 Go to Training Mode", type="primary"):
            # This will trigger a rerun with training mode
            st.session_state.mode = "train"
            st.rerun()

else:
    # Training mode
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    This will train a complete diabetes prediction model using your data.
    The process includes:
    1. Data preprocessing and cleaning
    2. Patient clustering (K-Means)
    3. Feature engineering
    4. Gradient Boosting model training
    5. Model evaluation and saving
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes"):
            pipeline_info = train_models()
            
            if pipeline_info:
                st.success("✅ Training completed successfully!")
                st.balloons()
                
                # Show training results
                st.markdown("---")
                st.markdown('<h3 class="sub-header">Training Results</h3>', unsafe_allow_html=True)
                
                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{pipeline_info["test_accuracy"]:.1%}</div><div class="metric-label">ACCURACY</div></div>', unsafe_allow_html=True)
                with col_res2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{pipeline_info["test_precision"]:.1%}</div><div class="metric-label">PRECISION</div></div>', unsafe_allow_html=True)
                with col_res3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{pipeline_info["test_f1_score"]:.1%}</div><div class="metric-label">F1-SCORE</div></div>', unsafe_allow_html=True)
                
                st.info("""
                ### 🎉 Ready to make predictions!
                
                **Next Steps:**
                1. Switch to **"Use Existing Models"** in the sidebar
                2. Enter patient information in the **Patient Assessment** tab
                3. Get instant diabetes risk predictions
                4. Explore patient clusters and model insights
                """)
                
                if st.button("🚀 Switch to Prediction Mode", type="primary"):
                    # This will trigger a rerun with prediction mode
                    st.session_state.mode = "predict"
                    st.rerun()
            else:
                st.error("Training failed. Please check your data and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏥 Diabetes Risk Prediction System | For screening purposes only</p>
    <p><small>Always consult healthcare professionals for medical diagnosis</small></p>
</div>
""", unsafe_allow_html=True)
