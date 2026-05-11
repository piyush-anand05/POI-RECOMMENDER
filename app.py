import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# --- page config ---
st.set_page_config(page_title="POI Recommender", layout="wide")

# --- load dataset ---
df = pd.read_csv("gowalla_processed.csv")

# --- encoding (no sklearn) ---
df['user_id'] = df['user'].astype('category').cat.codes
df['poi_id'] = df['poi'].astype('category').cat.codes

poi_mapping = dict(enumerate(df['poi'].astype('category').cat.categories))

num_users = df['user_id'].nunique()

# --- edges ---
edges = df[['user_id', 'poi_id']].values

# --- load embeddings ---
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
st.caption("LightGCN-inspired recommendation system using Gowalla dataset")

# --- input ---
user_id = st.number_input("Enter User ID", min_value=0, step=1)

if st.button("Recommend"):

    if user_id >= num_users:
        st.error("❌ Invalid user ID. Please try a valid one.")
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

            coords.append([lat, lon])

        # --- layout ---
        col1, col2 = st.columns([1, 2])

        # --- table ---
        with col1:
            st.markdown("### 📋 Details")
            st.dataframe(pd.DataFrame(poi_list), use_container_width=True)

        # --- map ---
        with col2:
            st.markdown("### 🗺️ Map View")

            map_df = pd.DataFrame(coords, columns=["lat", "lon"])

            # center map globally (Gowalla is global dataset)
            center_lat = map_df["lat"].mean()
            center_lon = map_df["lon"].mean()

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[lon, lat]',
                get_radius=50000,   # large so visible
                get_fill_color=[255, 0, 0],
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=3,   # global zoom
            )

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/light-v9",
                tooltip={"text": "Lat: {lat}\nLon: {lon}"}
            )

            st.pydeck_chart(deck)
