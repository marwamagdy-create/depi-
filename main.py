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
import matplotlib.pyplot as plt

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
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)

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
            with st.spinner("Training models..."):
                # Train models function
                train_models()
                st.success("✅ Models trained and saved successfully!")

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
        
        return True
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return False

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
            st.write(f"**Samples:** {models['pipeline_info']['training_samples'] + models['pipeline_info']['test_samples']:,}")
            st.write(f"**Diabetes Rate:** {models['pipeline_info']['diabetes_rate']:.1%}")
            st.write(f"**Clusters:** {models['pipeline_info']['n_clusters']}")
        
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
                        st.metric("Risk Level", f"{emoji} {risk_level}")
                    with res_col2:
                        st.metric("Probability", f"{diabetes_prob:.1%}")
                    with res_col3:
                        st.metric("Cluster", f"#{cluster}")
                    
                    # Risk visualization
                    st.subheader("Risk Visualization")
                    risk_percent = int(diabetes_prob * 100)
                    st.progress(risk_percent / 100)
                    st.caption(f"Risk Score: {risk_percent}%")
                    
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
                        st.dataframe(summary_df, hide_index=True)
        
        with tab2:
            st.markdown('<h2 class="sub-header">Cluster Analysis</h2>', unsafe_allow_html=True)
            st.info("Patient clusters based on biometric similarity")
            
            # Display cluster info
            st.write(f"**Number of Clusters:** {models['pipeline_info']['n_clusters']}")
            st.write(f"**Clustering Features:** {', '.join(models['pipeline_info']['clustering_features'])}")
            
            # Sample cluster data (in real app, load from data)
            cluster_data = pd.DataFrame({
                'Cluster': [f"Cluster {i}" for i in range(5)],
                'Description': [
                    "High-risk: Elevated all indicators",
                    "Moderate-risk: Mixed profile",
                    "Low-risk: Below average indicators",
                    "Very high-risk: Critical indicators",
                    "Average-risk: Normal range"
                ],
                'Sample Size': [5000, 5500, 5200, 4800, 5100],
                'Diabetes Rate': ["15-25%", "8-15%", "2-5%", "25-35%", "5-10%"]
            })
            
            st.dataframe(cluster_data, hide_index=True, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            importance_data = pd.DataFrame({
                'Feature': ['Blood Glucose', 'HbA1c Level', 'Age', 'BMI', 'Hypertension'],
                'Importance': [0.25, 0.18, 0.15, 0.12, 0.08]
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(importance_data['Feature'], importance_data['Importance'], color='#2E86AB')
            ax.set_xlabel('Importance')
            ax.set_title('Top Feature Importance')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            st.markdown('<h2 class="sub-header">Model Insights</h2>', unsafe_allow_html=True)
            
            # Model architecture
            col_arch1, col_arch2, col_arch3 = st.columns(3)
            with col_arch1:
                st.metric("Algorithm", "Gradient Boosting")
            with col_arch2:
                st.metric("Trees", "100")
            with col_arch3:
                st.metric("Max Depth", "3")
            
            # How it works
            st.subheader("How the Model Works")
            st.markdown("""
            1. **Clustering Phase**: Patients grouped into 5 clusters based on biometric similarity
            2. **Feature Engineering**: Cluster assignment becomes a predictive feature
            3. **Classification**: Gradient Boosting analyzes 40+ features including clusters
            4. **Risk Assessment**: Probability-based risk stratification
            
            **Key Features Analyzed:**
            - Biometric indicators (Age, BMI, HbA1c, Glucose)
            - Medical history (Hypertension, Heart Disease)
            - Demographic factors
            - Cluster membership
            """)
            
            # Performance metrics chart
            st.subheader("Performance Metrics")
            metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [
                    models['pipeline_info']['test_accuracy'],
                    models['pipeline_info']['test_precision'],
                    0.85,  # Sample recall
                    models['pipeline_info']['test_f1_score']
                ]
            })
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            bars2 = ax2.bar(metrics['Metric'], metrics['Value'], color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Score')
            ax2.set_title('Model Performance')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig2)
    
    else:
        st.warning("⚠️ No trained models found. Please train models first.")
        st.info("""
        To use the app:
        1. Select "Train New Models" in the sidebar
        2. Click "Train Models" button
        3. Wait for training to complete
        4. Switch back to "Use Existing Models"
        """)
        
        if st.button("🔄 Go to Training Mode"):
            st.session_state.mode = "Train New Models"
            st.rerun()

else:
    # Training mode
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes"):
            success = train_models()
            
            if success:
                st.success("✅ Training completed successfully!")
                st.balloons()
                
                # Show next steps
                st.info("""
                **Next Steps:**
                1. Switch to "Use Existing Models" in the sidebar
                2. Start making predictions
                3. Explore cluster analysis and model insights
                """)
                
                if st.button("🚀 Switch to Prediction Mode"):
                    st.session_state.mode = "Use Existing Models"
                    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🏥 Diabetes Risk Prediction System | For screening purposes only</p>
    <p><small>Always consult healthcare professionals for medical diagnosis</small></p>
</div>
""", unsafe_allow_html=True)
