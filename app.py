import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Salary Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 45px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🎓 Student Placement Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict your placement package now!</div>', unsafe_allow_html=True)
st.write("---")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('salary_prediction_model.pkl')
        return model, True
    except:
        return None, False

model, model_loaded = load_model()

if not model_loaded:
    st.error("❌ Model file not found! Please run the notebook first.")
    st.stop()

# Load dataset
df = pd.read_csv('updated_data.csv')

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Predict Salary", "📁 Batch Upload", "📊 Dataset Info"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.header("Enter Student Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Coding Skills Score")
        coding_skills = st.slider(
            "Rate your coding skills (0-10):",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )
        
        st.info("📌 **Tips:**\n• 2-4: Beginner\n• 5-6: Intermediate\n• 7-8: Advanced\n• 9-10: Expert")
    
    with col2:
        st.metric("Model Type", "Linear Regression")
        st.metric("Input Feature", "Coding Skills")
    
    st.write("---")
    
    # Prediction button
    if st.button("🔮 Predict Salary", use_container_width=True, key="predict"):
        try:
            predicted_salary = model.predict([[coding_skills]])[0]
            
            # Display prediction
            st.success("✅ Prediction Complete!")
            st.markdown(f'<div class="prediction-box">💰 ₹{predicted_salary:.2f} LPA</div>', unsafe_allow_html=True)
            
            # Salary category
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if predicted_salary >= 20:
                    st.markdown("### 🌟 High Package\nExcellent opportunity!")
                elif predicted_salary >= 15:
                    st.markdown("### ✅ Good Package\nSatisfactory offer!")
                else:
                    st.markdown("### 📊 Moderate Package\nBasic offer!")
            
            with col_res2:
                st.metric("Your Coding Score", f"{coding_skills:.1f}/10")
                st.metric("Predicted Salary", f"₹{predicted_salary:.2f} LPA")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==================== TAB 2: BATCH UPLOAD ====================
with tab2:
    st.header("Batch Predictions from CSV")
    
    st.write("Upload a CSV file with 'coding_skills' column for batch predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file:", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'coding_skills' not in df_upload.columns:
                st.error("❌ CSV must contain 'coding_skills' column")
            else:
                st.success(f"✅ File loaded! {len(df_upload)} records found")
                st.dataframe(df_upload.head(10), use_container_width=True)
                
                if st.button("🔮 Predict for All", use_container_width=True, key="batch_predict"):
                    predictions = model.predict(df_upload[['coding_skills']].values)
                    
                    results = df_upload.copy()
                    results['Predicted_Salary'] = predictions
                    results['Category'] = results['Predicted_Salary'].apply(
                        lambda x: '🌟 High (≥20)' if x >= 20 else ('✅ Good (15-20)' if x >= 15 else '📊 Moderate')
                    )
                    
                    st.write("---")
                    st.header("Results")
                    st.dataframe(results[['coding_skills', 'Predicted_Salary', 'Category']], use_container_width=True)
                    
                    # Stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(results))
                    col2.metric("Average", f"₹{predictions.mean():.2f}")
                    col3.metric("Max", f"₹{predictions.max():.2f}")
                    col4.metric("Min", f"₹{predictions.min():.2f}")
                    
                    # Download
                    csv = results.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv", use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==================== TAB 3: DATASET INFO ====================
with tab3:
    st.header("Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("File Size", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    st.write("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)
    
    st.write("---")
    st.subheader("Statistical Summary")
    st.dataframe(df[['coding_skills', 'salary_package_lpa']].describe(), use_container_width=True)

# Footer
st.write("---")
st.markdown("""
<center style='margin-top: 30px;'>
    <p style='font-size: 12px; color: gray;'>
        🎓 Student Placement Salary Predictor | ML Model
    </p>
</center>
""", unsafe_allow_html=True)
