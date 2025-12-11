import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

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
    }
    .risk-moderate {
        color: #FFD93D;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #6BCF7F;
        font-weight: bold;
        font-size: 1.2rem;
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
                elif diabetes_prob >= 0.4:
                    risk_level = "MODERATE RISK"
                    risk_class = "risk-moderate"
                    recommendation = "Regular monitoring and lifestyle changes advised"
                    emoji = "🟡"
                else:
                    risk_level = "LOW RISK"
                    risk_class = "risk-low"
                    recommendation = "Maintain healthy lifestyle with annual checkups"
                    emoji = "🟢"
                
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
                
                # Risk visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = diabetes_prob * 100,
                    title = {'text': "Diabetes Risk Score"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': diabetes_prob * 100
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
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
                    st.json({
                        "Age (std)": age,
                        "BMI (std)": bmi,
                        "HbA1c (std)": hbA1c,
                        "Blood Glucose (std)": blood_glucose,
                        "Hypertension": hypertension,
                        "Heart Disease": heart_disease,
                        "Gender": gender,
                        "Race": race,
                        "Smoking History": smoking
                    })
    
    with tab2:
        st.markdown('<h2 class="sub-header">Cluster Analysis</h2>', unsafe_allow_html=True)
        
        # Load sample data for visualization (you should load your actual data)
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
        st.dataframe(cluster_data, use_container_width=True)
        
        # Cluster visualization
        fig = px.bar(cluster_data, x='Cluster', y='Diabetes Rate (%)',
                     title='Diabetes Rate by Cluster',
                     color='Diabetes Rate (%)',
                     color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, use_container_width=True)
        
        # Parallel coordinates plot
        features_to_show = ['Avg Age (std)', 'Avg BMI (std)', 'Avg HbA1c (std)', 'Avg Glucose (std)']
        
        fig2 = go.Figure(data=
            go.Parcoords(
                line=dict(color=cluster_data['Diabetes Rate (%)'],
                         colorscale='YlOrRd'),
                dimensions=list([
                    dict(label='Cluster', values=cluster_data.index),
                    dict(label='Diabetes Rate (%)', values=cluster_data['Diabetes Rate (%)']),
                    dict(label='Avg Age', values=cluster_data['Avg Age (std)']),
                    dict(label='Avg BMI', values=cluster_data['Avg BMI (std)']),
                    dict(label='Avg HbA1c', values=cluster_data['Avg HbA1c (std)']),
                    dict(label='Avg Glucose', values=cluster_data['Avg Glucose (std)'])
                ])
            )
        )
        
        fig2.update_layout(title='Cluster Characteristics Comparison')
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Insights</h2>', unsafe_allow_html=True)
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            # Feature importance (sample data - replace with actual)
            feature_importance_data = pd.DataFrame({
                'Feature': ['blood_glucose_level', 'hbA1c_level', 'age', 'bmi', 
                           'hypertension', 'heart_disease', 'cluster_1', 'cluster_2'],
                'Importance': [0.25, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.03]
            })
            
            fig3 = px.bar(feature_importance_data.sort_values('Importance', ascending=True),
                         x='Importance', y='Feature', orientation='h',
                         title='Top Feature Importance',
                         color='Importance',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col_insight2:
            # Performance metrics
            metrics_data = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Value': [0.92, 0.88, 0.85, 0.86]
            })
            
            fig4 = px.bar(metrics_data, x='Metric', y='Value',
                         title='Model Performance Metrics',
                         color='Value',
                         color_continuous_scale='Greens',
                         text='Value')
            fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig4, use_container_width=True)
        
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