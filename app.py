import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# --- load dataset ---
df = pd.read_csv("gowalla_processed.csv")

# --- encoding ---
df['user_id'] = df['user'].astype('category').cat.codes
df['poi_id'] = df['poi'].astype('category').cat.codes

poi_mapping = dict(enumerate(df['poi'].astype('category').cat.categories))

num_users = df['user_id'].nunique()

# --- edges ---
edges = df[['user_id', 'poi_id']].values

# --- embeddings ---
user_emb = np.load("user_emb.npy")
item_emb = np.load("item_emb.npy")

# --- recommend ---
def recommend(user_id, top_k=5):
    scores = user_emb[user_id] @ item_emb.T

    interacted = edges[edges[:, 0] == user_id][:, 1]
    scores = scores.copy()
    scores[interacted] = -1e9

    return scores.argsort()[-top_k:][::-1]

# --- UI ---
st.title("🌍 Smart POI Recommender")
st.caption("Graph-based recommendation using LightGCN embeddings")

user_id = st.number_input("Enter User ID", min_value=0, step=1)

if st.button("Recommend"):

    if user_id >= num_users:
        st.error("Invalid user ID")
    else:
        recs = recommend(int(user_id))

        st.subheader("📍 Recommended Places")

        poi_list = []
        coords = []

        for pid in recs:
            poi_original = poi_mapping[pid]
            row = df[df['poi'] == poi_original].iloc[0]

            lat = float(row['lat'])
            lon = float(row['lon'])

            poi_list.append({
                "Name": f"Location {poi_original}",
                "Latitude": lat,
                "Longitude": lon
            })

            coords.append((lat, lon))

        # --- show table ---
        st.dataframe(pd.DataFrame(poi_list))

        # --- better map ---
        center = coords[0]
        m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

        for i, (lat, lon) in enumerate(coords):
            folium.Marker(
                [lat, lon],
                popup=f"Location {poi_list[i]['Name']}",
                tooltip="Click for details",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)

        st_folium(m, width=700, height=500)
