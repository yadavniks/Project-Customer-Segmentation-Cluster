# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 23:59:54 2025

@author: Rajiv Anatwar
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="Marketing Segmentation App", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca.joblib')
    kmeans = joblib.load('kmeans_model.joblib')
    return scaler, pca, kmeans

try:
    scaler, pca, kmeans = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("🎯 Customer Segmentation Analysis")
st.markdown("Enter customer details to identify their marketing segment.")

# --- INPUT SECTION ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Details")
    income = st.number_input("Annual Income ($)", 1000, 200000, 50000)
    year_birth = st.number_input("Year of Birth", 1940, 2010, 1980)
    kidhome = st.number_input("Kids at home", 0, 2, 0)
    teenhome = st.number_input("Teens at home", 0, 2, 0)
    recency = st.slider("Days since last purchase", 0, 100, 50)

with col2:
    st.subheader("🍷 Product Spending")
    mnt_wines = st.number_input("Wines ($)", 0, 2000, 300)
    mnt_fruits = st.number_input("Fruits ($)", 0, 200, 20)
    mnt_meat = st.number_input("Meat ($)", 0, 2000, 150)
    mnt_fish = st.number_input("Fish ($)", 0, 200, 20)
    mnt_sweet = st.number_input("Sweets ($)", 0, 200, 20)
    mnt_gold = st.number_input("Gold ($)", 0, 200, 20)

with col3:
    st.subheader("🖱️ Channel Engagement")
    web_p = st.number_input("Web Purchases", 0, 20, 5)
    cat_p = st.number_input("Catalog Purchases", 0, 20, 2)
    store_p = st.number_input("Store Purchases", 0, 20, 5)
    web_v = st.number_input("Web Visits/Month", 0, 20, 7)
    deals = st.number_input("Deals Purchases", 0, 15, 2)
    # Categoricals
    education = st.selectbox("Education Level", ["Graduation", "PhD", "Master", "Basic", "2n Cycle"])
    marital = st.selectbox("Marital Status", ["Married", "Together", "Single", "Divorced", "Widow", "Alone", "Absurd", "YOLO"])

# --- PREDICTION LOGIC ---
if st.button("🚀 Run Segment Analysis", use_container_width=True):
    # 1. DERIVED CALCULATIONS (Must match notebook exactly)
    total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    total_purchases = web_p + cat_p + store_p
    age = 2024 - year_birth
    children = kidhome + teenhome
    
    # 2. CREATE FEATURE LIST (ORDER IS CRITICAL)
    # The scaler expects exactly 35 columns in this specific order:
    features = {
        'Year_Birth': year_birth, 'Income': income, 'Kidhome': kidhome, 'Teenhome': teenhome,
        'Recency': recency, 'MntWines': mnt_wines, 'MntFruits': mnt_fruits, 
        'MntMeatProducts': mnt_meat, 'MntFishProducts': mnt_fish, 'MntSweetProducts': mnt_sweet,
        'MntGoldProds': mnt_gold, 'NumDealsPurchases': deals, 'NumWebPurchases': web_p,
        'NumCatalogPurchases': cat_p, 'NumStorePurchases': store_p, 'NumWebVisitsMonth': web_v,
        'Total_Purchases': total_purchases,
        'Web_Share': web_p/total_purchases if total_purchases > 0 else 0,
        'Catalog_Share': cat_p/total_purchases if total_purchases > 0 else 0,
        'Store_Share': store_p/total_purchases if total_purchases > 0 else 0,
        'Children': children,
        'Tenure_days': 1000, # Fixed baseline from notebook median
        'Wine_Share': mnt_wines/total_spend if total_spend > 0 else 0,
        'Meat_Share': mnt_meat/total_spend if total_spend > 0 else 0,
        
        # Education Dummies
        'Education_Basic': 1 if education == "Basic" else 0,
        'Education_Graduation': 1 if education == "Graduation" else 0,
        'Education_Master': 1 if education == "Master" else 0,
        'Education_PhD': 1 if education == "PhD" else 0,
        
        # Marital Status Dummies
        'Marital_Status_Alone': 1 if marital == "Alone" else 0,
        'Marital_Status_Divorced': 1 if marital == "Divorced" else 0,
        'Marital_Status_Married': 1 if marital == "Married" else 0,
        'Marital_Status_Single': 1 if marital == "Single" else 0,
        'Marital_Status_Together': 1 if marital == "Together" else 0,
        'Marital_Status_Widow': 1 if marital == "Widow" else 0,
        'Marital_Status_YOLO': 1 if marital == "YOLO" else 0
    }
    
    input_df = pd.DataFrame([features])

    try:
        # Pipeline: Scale -> PCA -> KMeans
        scaled_data = scaler.transform(input_df)
        pca_data = pca.transform(scaled_data)
        cluster_id = kmeans.predict(pca_data)[0]

        # 3. MAPPING (Updated to match your notebook's Cluster Summary)
        results = {
            0: {"name": "Premium VIP Spenders", "desc": "Highest Income & Spending ($73k+). Heavy wine/meat buyers.", "strat": "Exclusive Events & Premium Catalog."},
            1: {"name": "Steady Moderate Spenders", "desc": "Average Income ($50k). Reliable store shoppers.", "strat": "Loyalty Programs & Refer-a-friend."},
            2: {"name": "Value Shoppers", "desc": "Lower Income ($34k). Extremely price-sensitive.", "strat": "Discount Vouchers & Bulk-buy deals."},
            3: {"name": "Digital Online Enthusiasts", "desc": "Solid Income ($54k). Primarily shop via website.", "strat": "Mobile App Notifications & Digital Ads."}
        }
        
        res = results.get(cluster_id)

        # Display Results
        st.divider()
        st.header(f"Result: {res['name']}")
        st.write(f"**Customer Profile:** {res['desc']}")
        st.success(f"**Marketing Action:** {res['strat']}")
        
        # Debug Section
        with st.expander("See Technical Details"):
            st.write(f"Predicted Cluster Index: {cluster_id}")
            st.write("Input Feature Vector:")
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")