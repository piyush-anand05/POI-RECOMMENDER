import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests

# --- config ---
st.set_page_config(page_title="POI Recommender", layout="wide")

# --- load data ---
df = pd.read_csv("gowalla_processed.csv")

# --- encoding ---
df['user_id'] = df['user'].astype('category').cat.codes
df['poi_id'] = df['poi'].astype('category').cat.codes

poi_mapping = dict(enumerate(df['poi'].astype('category').cat.categories))

num_users = df['user_id'].nunique()
edges = df[['user_id', 'poi_id']].values

# --- embeddings ---
user_emb = np.load("user_emb.npy")
item_emb = np.load("item_emb.npy")

# --- recommendation ---
def recommend(user_id, top_k=5):
    scores = user_emb[user_id] @ item_emb.T
    interacted = edges[edges[:, 0] == user_id][:, 1]

    scores = scores.copy()
    scores[interacted] = -1e9

    return scores.argsort()[-top_k:][::-1]

# --- API for real place names (OpenStreetMap) ---
def get_place_name(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        res = requests.get(url, headers={"User-Agent": "poi-app"})
        data = res.json()
        return data.get("display_name", "Unknown Location")
    except:
        return "Unknown Location"

# --- UI ---
st.title("🌍 Smart POI Recommender")
st.caption("Graph-based recommendation with user history + real-world mapping")

user_id = st.number_input("Enter User ID", min_value=0, step=1)

if st.button("Recommend"):

    if user_id >= num_users:
        st.error("❌ Invalid user ID")
    else:
        recs = recommend(int(user_id))

        # --- past visits ---
        user_history = edges[edges[:, 0] == user_id][:, 1]

        st.subheader("📊 User History vs Recommendations")

        rec_coords = []
        hist_coords = []

        rec_table = []
        hist_table = []

        # --- process recommendations ---
        for pid in recs:
            poi_original = poi_mapping[pid]
            row = df[df['poi'] == poi_original].iloc[0]

            lat = float(row['lat'])
            lon = float(row['lon'])

            name = get_place_name(lat, lon)

            rec_coords.append([lat, lon])

            rec_table.append({
                "Type": "Recommended",
                "Name": name,
                "Lat": lat,
                "Lon": lon
            })

        # --- process history ---
        for pid in user_history[:10]:  # limit for clarity
            poi_original = poi_mapping[pid]
            row = df[df['poi'] == poi_original].iloc[0]

            lat = float(row['lat'])
            lon = float(row['lon'])

            name = get_place_name(lat, lon)

            hist_coords.append([lat, lon])

            hist_table.append({
                "Type": "Visited",
                "Name": name,
                "Lat": lat,
                "Lon": lon
            })

        # --- combine ---
        all_df = pd.DataFrame(rec_table + hist_table)

        col1, col2 = st.columns([1, 2])

        # --- table ---
        with col1:
            st.markdown("### 📋 Details")
            st.dataframe(all_df, use_container_width=True)

        # --- map ---
        with col2:
            st.markdown("### 🗺️ Map View")

            map_df = pd.DataFrame(
                rec_coords + hist_coords,
                columns=["lat", "lon"]
            )

            center_lat = map_df["lat"].mean()
            center_lon = map_df["lon"].mean()

            # --- layers ---
            rec_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(rec_coords, columns=["lat", "lon"]),
                get_position='[lon, lat]',
                get_radius=80000,
                get_fill_color=[255, 0, 0],  # RED = recommended
                pickable=True,
            )

            hist_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(hist_coords, columns=["lat", "lon"]),
                get_position='[lon, lat]',
                get_radius=80000,
                get_fill_color=[0, 0, 255],  # BLUE = visited
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=2.5,
            )

            deck = pdk.Deck(
                layers=[rec_layer, hist_layer],
                initial_view_state=view_state,
                tooltip={"text": "Lat: {lat}\nLon: {lon}"}
            )

            st.pydeck_chart(deck)

        st.markdown("🔴 Red = Recommended | 🔵 Blue = Visited")
