import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# --- Load Model, Scaler, and Data ---
@st.cache_resource
def load_components():
    model = joblib.load(os.path.join('models', 'best_titanic_model.joblib'))
    scaler = joblib.load(os.path.join('models', 'scaler.joblib'))
    return model, scaler

@st.cache_data
def load_data():
    # Load the raw dataset for our insight graphs
    return pd.read_csv(os.path.join('data', 'titanic.csv'))

model, scaler = load_components()
df = load_data()

# --- UI Header ---
st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to predict survival, or explore the model insights in the tabs below.")
st.divider()

# --- Setup Tabs ---
tab1, tab2 = st.tabs(["🎯 Make a Prediction", "📊 Model Insights & Data"])

# ==========================================
# TAB 1: PREDICTION ENGINE
# ==========================================
with tab1:
    # --- Input Sections ---
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", options=[1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
        sex = st.selectbox("Sex", options=["Male", "Female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

    with col2:
        title = st.selectbox("Passenger Title", options=["Mr", "Mrs", "Miss", "Master", "Rare (e.g., Dr, Rev, Capt)"])
        sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1)
        parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0, step=1)
        embarked = st.selectbox("Port of Embarkation", options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

    # --- Prediction Logic ---
    st.write("") # Spacing
    predict_btn = st.button("Predict Survival", type="primary", use_container_width=True)

    if predict_btn:
        family_size = sibsp + parch
        
        sex_map = {"Female": 0, "Male": 1}
        embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        title_map = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare (e.g., Dr, Rev, Capt)": 4}
        
        input_data = pd.DataFrame([[
            pclass, sex_map[sex], age, fare, family_size, title_map[title], embarked_map[embarked]
        ]], columns=['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title', 'Embarked'])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]
        
        st.divider()
        st.markdown("### Prediction Result")
        if prediction[0] == 1:
            st.success(f"**Survived!** (Probability: {probability[1]*100:.1f}%)")
            st.balloons()
        else:
            st.error(f"**Did Not Survive.** (Probability: {probability[0]*100:.1f}%)")

# ==========================================
# TAB 2: DATA VISUALIZATIONS
# ==========================================
with tab2:
    st.header("Inside the Machine Learning Model")
    st.markdown("Understanding how the Random Forest makes its decisions.")
    
    # --- Graph 1: Feature Importance ---
    st.subheader("1. Feature Importance")
    st.markdown("This chart shows which data points hold the most weight in predicting survival.")
    
    # Extract importances from our trained Random Forest
    importances = model.feature_importances_
    features = ['Passenger Class', 'Sex', 'Age', 'Fare', 'Family Size', 'Title', 'Port of Embarkation']
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Plot
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax1)
    ax1.set_xlabel("Importance Score")
    ax1.set_ylabel("")
    st.pyplot(fig1)

    st.divider()

    # --- Graph 2 & 3: Historical Data ---
    st.subheader("2. Historical Survival Data")
    st.markdown("A look at the actual distribution of survivors in our dataset.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2', ax=ax2)
        ax2.set_title("Survival by Passenger Class")
        ax2.set_xlabel("Ticket Class (1st, 2nd, 3rd)")
        ax2.legend(title='Survived', labels=['No', 'Yes'])
        st.pyplot(fig2)

    with col_b:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1', ax=ax3)
        ax3.set_title("Survival by Gender")
        ax3.set_xlabel("Gender")
        ax3.legend(title='Survived', labels=['No', 'Yes'])
        st.pyplot(fig3)