import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Title and description
st.markdown('<h1 class="main-header">🏥 Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="info-box">
    This system uses machine learning to predict diabetes risk based on patient characteristics. 
    The model analyzes key health indicators to provide risk assessment and recommendations.
    </div>
""", unsafe_allow_html=True)

# Sidebar for model loading and information
with st.sidebar:
    st.header("📊 Model Information")
    
    # Load models
    @st.cache_resource
    def load_models():
        try:
            gb_model = joblib.load('diabetes_classification_model.pkl')
            scaler_cluster = joblib.load('scaler_cluster.pkl')
            scaler_class = joblib.load('scaler_classification.pkl')
            cluster_encoder = joblib.load('cluster_encoder.pkl')
            kmeans = joblib.load('kmeans_cluster.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            clustering_features = joblib.load('clustering_features.pkl')
            pipeline_info = joblib.load('pipeline_info.pkl')
            
            return {
                'gb_model': gb_model,
                'scaler_cluster': scaler_cluster,
                'scaler_class': scaler_class,
                'cluster_encoder': cluster_encoder,
                'kmeans': kmeans,
                'feature_columns': feature_columns,
                'clustering_features': clustering_features,
                'pipeline_info': pipeline_info
            }
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None
    
    models = load_models()
    
    if models:
        st.success("✅ Models loaded successfully!")
        
        # Display model info
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{models['pipeline_info']['test_accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{models['pipeline_info']['test_precision']:.2%}")
        with col3:
            st.metric("F1 Score", f"{models['pipeline_info']['test_f1_score']:.2%}")
        
        st.subheader("Model Details")
        st.write(f"**Algorithm:** {models['pipeline_info']['model_type']}")
        st.write(f"**Clusters:** {models['pipeline_info']['n_clusters']}")
        st.write(f"**Features:** {len(models['feature_columns'])}")
        
        # Show cluster info
        if st.checkbox("Show Cluster Information"):
            st.write("**Clustering Features:**")
            for feature in models['pipeline_info']['clustering_features']:
                st.write(f"- {feature}")

# Main content area
if models:
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📝 Patient Assessment", "📊 Cluster Analysis", "📈 Model Insights"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
        
        # Create columns for input layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Biometric Information")
            
            # Note about standardized values
            st.info("""
            **Note:** These values should be standardized (z-scores). 
            Positive values indicate above average, negative values indicate below average.
            """)
            
            age = st.slider("Age (standardized)", -3.0, 3.0, 0.0, 0.1, 
                           help="Standardized age score")
            bmi = st.slider("BMI (standardized)", -3.0, 3.0, 0.0, 0.1,
                           help="Standardized BMI score")
            hbA1c = st.slider("HbA1c Level (standardized)", -3.0, 3.0, 0.0, 0.1,
                             help="Standardized HbA1c level")
            blood_glucose = st.slider("Blood Glucose (standardized)", -3.0, 3.0, 0.0, 0.1,
                                     help="Standardized blood glucose level")
        
        with col2:
            st.subheader("Medical History")
            
            hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
            heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
            
            st.subheader("Demographic Information")
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            
            race = st.selectbox("Race/Ethnicity", [
                "African American",
                "Asian", 
                "Caucasian",
                "Hispanic",
                "Other"
            ])
            
            smoking = st.selectbox("Smoking History", [
                "Current",
                "Ever",
                "Former",
                "Never",
                "Not Current",
                "No Info"
            ])
        
        # Prediction button
        if st.button("🔍 Assess Diabetes Risk", type="primary"):
            with st.spinner("Analyzing patient data..."):
                # Prepare input data
                gender_Female = 1 if gender == "Female" else 0
                gender_Male = 1 if gender == "Male" else 0
                
                # Race encoding
                race_dict = {
                    "African American": ("race:AfricanAmerican", 1),
                    "Asian": ("race:Asian", 1),
                    "Caucasian": ("race:Caucasian", 1),
                    "Hispanic": ("race:Hispanic", 1),
                    "Other": ("race:Other", 1)
                }
                
                # Smoking history encoding
                smoking_dict = {
                    "Current": ("smoking_history_current", 1),
                    "Ever": ("smoking_history_ever", 1),
                    "Former": ("smoking_history_former", 1),
                    "Never": ("smoking_history_never", 1),
                    "Not Current": ("smoking_history_not current", 1),
                    "No Info": ("smoking_history_No Info", 1)
                }
                
                # Prepare input dictionary
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
                
                # Set the selected race
                race_col, race_val = race_dict[race]
                input_data[race_col] = race_val
                
                # Set the selected smoking history
                smoking_col, smoking_val = smoking_dict[smoking]
                input_data[smoking_col] = smoking_val
                
                # Step 1: Clustering
                clustering_features_array = np.array([[age, bmi, hbA1c, blood_glucose]])
                clustering_features_scaled = models['scaler_cluster'].transform(clustering_features_array)
                cluster = models['kmeans'].predict(clustering_features_scaled)[0]
                
                # Step 2: Prepare for classification
                input_data['cluster'] = cluster
                
                # Create dataframe
                new_df = pd.DataFrame([input_data])
                
                # One-hot encode cluster
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
                
                # Ensure all columns match
                for col in models['feature_columns']:
                    if col not in X_new.columns:
                        X_new[col] = 0
                
                X_new = X_new[models['feature_columns']]
                
                # Scale for classification
                X_new_scaled = models['scaler_class'].transform(X_new)
                
                # Make prediction
                proba = models['gb_model'].predict_proba(X_new_scaled)[0]
                prediction = models['gb_model'].predict(X_new_scaled)[0]
                
                # Interpret results
                diabetes_prob = proba[1]
                
                if diabetes_prob >= 0.7:
                    risk_level = "HIGH RISK"
                    risk_class = "risk-high"
                    recommendation = "Immediate medical consultation and lifestyle intervention recommended"
                    emoji = "🔴"
                    color = "#FF6B6B"
                elif diabetes_prob >= 0.4:
                    risk_level = "MODERATE RISK"
                    risk_class = "risk-moderate"
                    recommendation = "Regular monitoring and lifestyle changes advised"
                    emoji = "🟡"
                    color = "#FFD93D"
                else:
                    risk_level = "LOW RISK"
                    risk_class = "risk-low"
                    recommendation = "Maintain healthy lifestyle with annual checkups"
                    emoji = "🟢"
                    color = "#6BCF7F"
                
                # Display results
                st.markdown("---")
                st.markdown('<h2 class="sub-header">Assessment Results</h2>', unsafe_allow_html=True)
                
                # Results in columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Risk Level", f"{emoji} {risk_level}")
                
                with res_col2:
                    st.metric("Diabetes Probability", f"{diabetes_prob:.1%}")
                
                with res_col3:
                    st.metric("Assigned Cluster", f"Cluster {cluster}")
                
                # Risk visualization using native Streamlit
                st.subheader("Risk Visualization")
                
                # Create a simple progress bar for risk
                risk_percentage = diabetes_prob * 100
                st.progress(int(risk_percentage))
                st.caption(f"Risk Score: {risk_percentage:.1f}%")
                
                # Create a simple color-coded risk indicator
                col_ind1, col_ind2, col_ind3 = st.columns(3)
                with col_ind1:
                    if risk_percentage < 40:
                        st.markdown(f'<div style="text-align: center; padding: 10px; background-color: #6BCF7F; color: white; border-radius: 5px;">LOW<br>{risk_percentage:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align: center; padding: 10px; background-color: #E0E0E0; color: #666; border-radius: 5px;">LOW</div>', unsafe_allow_html=True)
                
                with col_ind2:
                    if 40 <= risk_percentage < 70:
                        st.markdown(f'<div style="text-align: center; padding: 10px; background-color: #FFD93D; color: white; border-radius: 5px;">MODERATE<br>{risk_percentage:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align: center; padding: 10px; background-color: #E0E0E0; color: #666; border-radius: 5px;">MODERATE</div>', unsafe_allow_html=True)
                
                with col_ind3:
                    if risk_percentage >= 70:
                        st.markdown(f'<div style="text-align: center; padding: 10px; background-color: #FF6B6B; color: white; border-radius: 5px;">HIGH<br>{risk_percentage:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="text-align: center; padding: 10px; background-color: #E0E0E0; color: #666; border-radius: 5px;">HIGH</div>', unsafe_allow_html=True)
                
                # Recommendations
                st.markdown(f'<div class="{risk_class}">Recommendation: {recommendation}</div>', unsafe_allow_html=True)
                
                # Detailed information
                with st.expander("View Detailed Analysis"):
                    st.write("**Prediction Probabilities:**")
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Non-Diabetic", f"{proba[0]:.1%}")
                    with col_prob2:
                        st.metric("Diabetic", f"{proba[1]:.1%}")
                    
                    st.write("**Input Features:**")
                    
                    # Display input features in a nicer format
                    feature_table = pd.DataFrame({
                        'Feature': ['Age (standardized)', 'BMI (standardized)', 'HbA1c (standardized)', 
                                   'Blood Glucose (standardized)', 'Hypertension', 'Heart Disease',
                                   'Gender', 'Race', 'Smoking History', 'Assigned Cluster'],
                        'Value': [f"{age:.2f}", f"{bmi:.2f}", f"{hbA1c:.2f}", f"{blood_glucose:.2f}",
                                 hypertension, heart_disease, gender, race, smoking, f"Cluster {cluster}"]
                    })
                    
                    st.dataframe(feature_table, hide_index=True, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Cluster Analysis</h2>', unsafe_allow_html=True)
        
        # Load sample data for visualization
        st.info("This section shows characteristics of different patient clusters identified by the model.")
        
        # Create sample cluster data for visualization
        clusters = list(range(5))
        cluster_names = [f"Cluster {i}" for i in clusters]
        
        # Sample data (replace with your actual cluster statistics)
        cluster_data = pd.DataFrame({
            'Cluster': cluster_names,
            'Diabetes Rate (%)': [15.2, 8.5, 3.2, 21.8, 6.7],
            'Avg Age (std)': [1.2, 0.3, -0.8, 1.8, -0.2],
            'Avg BMI (std)': [1.5, 0.5, -0.5, 2.1, 0.2],
            'Avg HbA1c (std)': [1.8, 0.4, -0.9, 2.3, 0.1],
            'Avg Glucose (std)': [1.6, 0.3, -0.7, 2.0, 0.0]
        })
        
        # Display cluster statistics
        st.subheader("Cluster Characteristics")
        st.dataframe(cluster_data, use_container_width=True)
        
        # Bar chart for diabetes rate by cluster
        st.subheader("Diabetes Rate by Cluster")
        st.bar_chart(cluster_data.set_index('Cluster')['Diabetes Rate (%)'])
        
        # Display cluster descriptions
        st.subheader("Cluster Descriptions")
        
        cluster_descriptions = {
            0: "High-risk patients with elevated biometric indicators across all measures",
            1: "Average-risk patients with moderate biometric indicators",
            2: "Low-risk patients with below-average biometric indicators",
            3: "Very high-risk patients with significantly elevated indicators",
            4: "Moderate-risk patients with mixed indicator profile"
        }
        
        for cluster_num in clusters:
            with st.container():
                st.markdown(f'<div class="cluster-box">', unsafe_allow_html=True)
                st.markdown(f"**Cluster {cluster_num}**")
                st.write(f"Diabetes Rate: {cluster_data.loc[cluster_num, 'Diabetes Rate (%)']}%")
                st.write(cluster_descriptions[cluster_num])
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Insights</h2>', unsafe_allow_html=True)
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            # Feature importance (sample data - replace with actual)
            feature_importance_data = pd.DataFrame({
                'Feature': ['Blood Glucose', 'HbA1c Level', 'Age', 'BMI', 
                           'Hypertension', 'Heart Disease', 'Cluster Features', 'Smoking History'],
                'Importance': [0.25, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.03]
            })
            
            st.subheader("Top Feature Importance")
            # Create a horizontal bar chart using matplotlib
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(8, 6))
            feature_importance_sorted = feature_importance_data.sort_values('Importance', ascending=True)
            y_pos = range(len(feature_importance_sorted))
            
            ax.barh(y_pos, feature_importance_sorted['Importance'], color='#2E86AB')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_importance_sorted['Feature'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance Ranking')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
        
        with col_insight2:
            # Performance metrics
            metrics_data = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [0.92, 0.88, 0.85, 0.86]
            })
            
            st.subheader("Model Performance Metrics")
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            bars = ax2.bar(metrics_data['Metric'], metrics_data['Value'], color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Score')
            ax2.set_title('Model Performance')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig2)
        
        # Model information
        st.markdown("### Model Details")
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Algorithm", "Gradient Boosting")
        with info_col2:
            st.metric("Number of Trees", "100")
        with info_col3:
            st.metric("Max Depth", "3")
        
        # Model description
        st.markdown("""
        ### How the Model Works
        
        1. **Clustering Phase**: Patients are grouped into 5 clusters based on:
           - Age, BMI, HbA1c, and Blood Glucose levels
        
        2. **Feature Engineering**:
           - Cluster assignment becomes a new feature
           - One-hot encoding of clusters
        
        3. **Classification Phase**:
           - Gradient Boosting classifier analyzes all features
           - Predicts diabetes risk with probability scores
        
        4. **Risk Assessment**:
           - Low Risk: < 40% probability
           - Moderate Risk: 40-70% probability  
           - High Risk: > 70% probability
        """)

else:
    st.error("⚠️ Models could not be loaded. Please check if the model files exist in the current directory.")
    st.info("""
    Required model files:
    1. diabetes_classification_model.pkl
    2. scaler_cluster.pkl
    3. scaler_classification.pkl
    4. cluster_encoder.pkl
    5. kmeans_cluster.pkl
    6. feature_columns.pkl
    7. clustering_features.pkl
    8. pipeline_info.pkl
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏥 Diabetes Risk Prediction System | Developed for Healthcare Analytics</p>
    <p><small>Note: This tool is for screening purposes only. Always consult with healthcare professionals for medical diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)
