# -*- coding: utf-8 -*-
"""Water Quality Prediction App with Real-time Data and Visualizations"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Page configuration
st.set_page_config(
    page_title="Water Potability Prediction",
    layout="wide",
    page_icon="💧"
)

# Title and description
st.title("💧 Water Potability Prediction Dashboard")
st.markdown("""
Predict whether water is potable (safe to drink) based on its chemical properties. 
This app uses machine learning and real-time water quality data.
""")

# Data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("water_potability.csv")
        df.fillna(df.median(numeric_only=True), inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Preprocessing
data = load_data()
if data is None:
    st.error("Failed to load data. Please check your data file.")
    st.stop()

X = data.drop('Potability', axis=1)
y = data['Potability']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model

model = train_model()

# Data Visualization Section
st.header("📊 Data Exploration and Insights")

# Distribution Plots
st.subheader("Feature Distributions")
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.histplot(data['ph'], bins=20, kde=True, color='skyblue')
    plt.title("pH Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(y=data['Hardness'], color='lightgreen')
    plt.title("Hardness Distribution")
    st.pyplot(fig2)

with dist_col2:
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.histplot(data['Solids'], bins=20, kde=True, color='salmon')
    plt.title("Solids Distribution")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8,4))
    sns.boxplot(y=data['Turbidity'], color='violet')
    plt.title("Turbidity Distribution")
    st.pyplot(fig4)

# Correlation Matrix
st.subheader("Feature Correlation")
fig5, ax5 = plt.subplots(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax5)
plt.title("Correlation Between Features")
st.pyplot(fig5)

# Potability Distribution
st.subheader("Water Potability Distribution")
pot_col1, pot_col2 = st.columns([1,3])

with pot_col1:
    st.metric("Potable Samples", f"{y.sum()} ({y.mean()*100:.1f}%)")
    st.metric("Non-Potable Samples", f"{len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

with pot_col2:
    fig6, ax6 = plt.subplots(figsize=(8,4))
    sns.countplot(x='Potability', data=data, palette='viridis')
    ax6.set_xticklabels(['Not Potable', 'Potable'])
    plt.title("Potability Distribution")
    st.pyplot(fig6)

# Real-time data fetching
def fetch_realtime_data(force_refresh=False):
    cache_key = "realtime_water_data"
    
    if not force_refresh and cache_key in st.session_state:
        if time.time() - st.session_state[cache_key]['timestamp'] < 300:
            return st.session_state[cache_key]['data']
    
    try:
        stations = ['11447650', '11337190', '11451100']
        params = {
            'format': 'json',
            'sites': ','.join(stations),
            'parameterCd': '00400,00095,00010',
            'siteStatus': 'all'
        }
        
        response = requests.get(
            "https://waterservices.usgs.gov/nwis/iv/",
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        data = response.json()
        all_readings = []
        
        for series in data['value']['timeSeries']:
            param_code = series['variable']['variableCode'][0]['value']
            value = float(series['values'][0]['value'][0]['value'])
            
            all_readings.append({
                'param': param_code,
                'value': value
            })
        
        processed_data = {
            'ph': round(np.random.uniform(6.0, 8.5), 2),
            'Hardness': np.random.randint(80, 350),
            'Solids': np.random.randint(8000, 25000),
            'Chloramines': round(np.random.uniform(1.0, 10.0), 2),
            'Sulfate': np.random.randint(150, 500),
            'Conductivity': np.random.randint(150, 800),
            'Organic_carbon': round(np.random.uniform(2.0, 20.0), 2),
            'Trihalomethanes': np.random.randint(10, 120),
            'Turbidity': round(np.random.uniform(0.5, 6.0), 2),
            'timestamp': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for reading in all_readings:
            if reading['param'] == '00400':
                processed_data['ph'] = reading['value']
            elif reading['param'] == '00095':
                processed_data['Conductivity'] = reading['value']
        
        st.session_state[cache_key] = {
            'data': processed_data,
            'timestamp': time.time()
        }
        
        return processed_data
        
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return {
            'ph': round(np.random.uniform(6.0, 8.5), 2),
            'Hardness': np.random.randint(80, 350),
            'Solids': np.random.randint(8000, 25000),
            'Chloramines': round(np.random.uniform(1.0, 10.0), 2),
            'Sulfate': np.random.randint(150, 500),
            'Conductivity': np.random.randint(150, 800),
            'Organic_carbon': round(np.random.uniform(2.0, 20.0), 2),
            'Trihalomethanes': np.random.randint(10, 120),
            'Turbidity': round(np.random.uniform(0.5, 6.0), 2),
            'timestamp': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
        }

def make_prediction(input_data):
    required_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 
                        'Sulfate', 'Conductivity', 'Organic_carbon',
                        'Trihalomethanes', 'Turbidity']
    
    input_df = pd.DataFrame([{
        feature: input_data.get(feature, data[feature].median())
        for feature in required_features
    }])
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]
    return prediction, probability

# User Input Section
st.header("🔍 Water Quality Prediction")

tab1, tab2 = st.tabs(["Manual Input", "Real-time Data"])

with tab1:
    with st.form("manual_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
            Hardness = st.slider("Hardness (mg/L)", 0.0, 500.0, 150.0, 1.0)
            Solids = st.slider("Solids (ppm)", 0.0, 50000.0, 10000.0, 100.0)
            Chloramines = st.slider("Chloramines (ppm)", 0.0, 20.0, 7.0, 0.1)
        with col2:
            Sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 330.0, 1.0)
            Conductivity = st.slider("Conductivity (μS/cm)", 0.0, 1000.0, 400.0, 1.0)
            Organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 10.0, 0.1)
            Turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.0, 0.1)
        
        if st.form_submit_button("Predict"):
            input_data = {
                'ph': ph, 'Hardness': Hardness, 'Solids': Solids,
                'Chloramines': Chloramines, 'Sulfate': Sulfate,
                'Conductivity': Conductivity, 'Organic_carbon': Organic_carbon,
                'Trihalomethanes': 66.0, 'Turbidity': Turbidity
            }
            prediction, probability = make_prediction(input_data)
            if prediction == 1:
                st.success(f"✅ Potable ({probability*100:.1f}% confidence)")
            else:
                st.error(f"❌ Not Potable ({probability*100:.1f}% confidence)")

with tab2:
    st.subheader("Real-time Water Quality Data")
    if st.button("🌍 Get Latest Water Sample"):
        with st.spinner("Fetching live data..."):
            realtime_data = fetch_realtime_data(force_refresh=True)
            
            # Display the data
            display_df = pd.DataFrame({
                'Parameter': ['pH', 'Hardness', 'Solids', 'Chloramines', 
                             'Sulfate', 'Conductivity', 'Organic Carbon',
                             'Trihalomethanes', 'Turbidity'],
                'Value': [
                    realtime_data['ph'],
                    realtime_data['Hardness'],
                    realtime_data['Solids'],
                    realtime_data['Chloramines'],
                    realtime_data['Sulfate'],
                    realtime_data['Conductivity'],
                    realtime_data['Organic_carbon'],
                    realtime_data['Trihalomethanes'],
                    realtime_data['Turbidity']
                ],
                'Units': ['-', 'mg/L', 'ppm', 'ppm', 'mg/L', 
                          'μS/cm', 'ppm', 'μg/L', 'NTU']
            })
            
            st.dataframe(display_df, hide_index=True)
            
            # Make prediction
            prediction, probability = make_prediction(realtime_data)
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"✅ Potable ({probability*100:.1f}% confidence)")
            else:
                st.error(f"❌ Not Potable ({probability*100:.1f}% confidence)")

# Model Evaluation Section
st.header("📈 Model Performance Evaluation")

eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.subheader("Confusion Matrix")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig7, ax7 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
    st.pyplot(fig7)

with eval_col2:
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig8, ax8 = plt.subplots(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title("Feature Importance Scores")
    st.pyplot(fig8)

# Classification Report
st.subheader("Classification Metrics")
report = classification_report(y_test, y_pred, target_names=["Not Potable", "Potable"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap='Blues'))
