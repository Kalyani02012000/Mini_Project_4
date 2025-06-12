import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === Title ===
st.set_page_config(page_title="Restaurant Recommender", layout="wide")
st.title("🍽️ Restaurant Recommender App")

# === Load Encoders and Scalers ===
@st.cache_resource
def load_encoders():
    with open("label_encoder_name.pkl", "rb") as f:
        le = pickle.load(f)
    with open("city_encoder.pkl", "rb") as f:
        city_encoder = pickle.load(f)
    with open("cuisine_encoder.pkl", "rb") as f:
        cuisine_encoder = pickle.load(f)
    return le, city_encoder, cuisine_encoder

# === Load Data ===
@st.cache_data
def load_data():
    original_df = pd.read_csv("cleaned_data.csv")
    encoded_df = pd.read_csv("final_encoded_data.csv")
    return original_df, encoded_df

le, city_encoder, cuisine_encoder = load_encoders()
original_df, encoded_df = load_data()

# === User Input Sidebar ===
st.sidebar.header("Filter Preferences")

# === Sidebar Filters ===
st.sidebar.subheader("📍 Filter by Location")

# Sub-city filter
sub_city_options = sorted(original_df['sub_city'].dropna().unique())
selected_sub_city = st.sidebar.selectbox("Select Sub-City", ["All"] + sub_city_options)

# Main-city filter
main_city_options = sorted(original_df['main_city'].dropna().unique())
selected_main_city = st.sidebar.selectbox("Select Main-City", ["All"] + main_city_options)

# Budget slider
st.sidebar.subheader("💸 Filter by Budget")
min_cost = int(original_df['cost'].min())
max_cost = int(original_df['cost'].max())
selected_cost = st.sidebar.slider("Max Budget (₹)", 1, max_cost, 300350)

# Rating slider
st.sidebar.subheader("⭐ Filter by Rating")
selected_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, step=0.1)

# Cuisine multiselect
st.sidebar.subheader("🍽️ Filter by Cuisine(s)")
cuisine_options = sorted(set(c for sublist in original_df['cuisine'].dropna().str.split(',') for c in sublist))
selected_cuisines = st.sidebar.multiselect("Select Cuisines", cuisine_options)

# Apply Filter Button
apply_filter = st.sidebar.button("✅ Apply Filters")

# === Filter and Display Data ===
if apply_filter:
    filtered_df = original_df.copy()

    if selected_sub_city != "All":
        filtered_df = filtered_df[filtered_df['sub_city'] == selected_sub_city]
    if selected_main_city != "All":
        filtered_df = filtered_df[filtered_df['main_city'] == selected_main_city]

    filtered_df = filtered_df[filtered_df['cost'] <= selected_cost]
    filtered_df = filtered_df[filtered_df['rating'] >= selected_rating]

    if selected_cuisines:
        filtered_df = filtered_df[filtered_df['cuisine'].apply(lambda x: any(c in x for c in selected_cuisines))]

    st.subheader(f"🔎 {len(filtered_df)} Restaurants Matching Your Preferences")
    st.dataframe(filtered_df[['name', 'sub_city', 'main_city', 'cuisine', 'cost', 'rating', 'rating_count']])
else:
    st.subheader("🔍 Use the sidebar to filter and click '✅ Apply Filters'")