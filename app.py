import streamlit as st
import pandas as pd
import numpy as np

# --- load dataset ---
df = pd.read_csv("gowalla_processed.csv")

# --- encoders ---
from sklearn.preprocessing import LabelEncoder
user_encoder = LabelEncoder()
poi_encoder = LabelEncoder()

df['user_id'] = user_encoder.fit_transform(df['user'])
df['poi_id'] = poi_encoder.fit_transform(df['poi'])

num_users = df['user_id'].nunique()
num_items = df['poi_id'].nunique()

# --- edges ---
edges = df[['user_id', 'poi_id']].values

# --- load embeddings (PRECOMPUTED) ---
user_emb = np.load("user_emb.npy")
item_emb = np.load("item_emb.npy")

# --- recommend ---
def recommend(user_id, top_k=5):
    scores = user_emb[user_id] @ item_emb.T

    interacted = edges[edges[:, 0] == user_id][:, 1]

    scores = scores.copy()
    scores[interacted] = -1e9

    return scores.argsort()[-top_k:][::-1]

# --- map helper ---
def get_poi_info(poi_ids):
    result = []
    for pid in poi_ids:
        poi_original = poi_encoder.inverse_transform([pid])[0]
        row = df[df['poi'] == poi_original].iloc[0]
        result.append([float(row['lat']), float(row['lon'])])
    return result

# --- UI ---
st.title("POI Recommendation System")

user_id = st.number_input("Enter User ID", min_value=0, step=1)

if st.button("Recommend"):
    recs = recommend(int(user_id))
    coords = get_poi_info(recs)

    st.write("Recommended Locations:")

    # Convert to DataFrame for Streamlit map
    map_df = pd.DataFrame(coords, columns=["lat", "lon"])

    # Built-in map (no folium)
    st.map(map_df)