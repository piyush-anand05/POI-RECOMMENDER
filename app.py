import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from recommender import Recommender
from utils import get_place_name

# --- config ---
st.set_page_config(page_title="POI Recommender", layout="wide")

# --- load data ---
df = pd.read_csv("data/gowalla_processed.csv")
user_emb = np.load("data/user_emb.npy")
item_emb = np.load("data/item_emb.npy")

rec = Recommender(df, user_emb, item_emb)

num_users = df['user'].nunique()

# --- UI ---
st.title("🌍 Smart POI Recommender")
st.caption("Graph-based recommendation with real-world mapping")

user_id = st.number_input("Enter User ID", min_value=0, step=1)

if st.button("Recommend"):

    if user_id >= num_users:
        st.error("Invalid user ID")
    else:
        recs = rec.recommend(user_id)
        history = rec.get_user_history(user_id)

        rec_coords, hist_coords = [], []
        rec_table, hist_table = [], []

        # --- recommendations ---
        for pid in recs:
            poi = rec.poi_mapping[pid]
            row = df[df['poi'] == poi].iloc[0]

            lat, lon = float(row['lat']), float(row['lon'])
            name = get_place_name(lat, lon)

            rec_coords.append([lat, lon])
            rec_table.append({"Type": "Recommended", "Name": name, "Lat": lat, "Lon": lon})

        # --- history ---
        for pid in history[:10]:
            poi = rec.poi_mapping[pid]
            row = df[df['poi'] == poi].iloc[0]

            lat, lon = float(row['lat']), float(row['lon'])
            name = get_place_name(lat, lon)

            hist_coords.append([lat, lon])
            hist_table.append({"Type": "Visited", "Name": name, "Lat": lat, "Lon": lon})

        col1, col2 = st.columns([1, 2])

        # --- table ---
        with col1:
            st.dataframe(pd.DataFrame(rec_table + hist_table))

        # --- map ---
        with col2:
            map_df = pd.DataFrame(rec_coords + hist_coords, columns=["lat", "lon"])

            center_lat = map_df["lat"].mean()
            center_lon = map_df["lon"].mean()

            rec_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(rec_coords, columns=["lat", "lon"]),
                get_position='[lon, lat]',
                get_radius=80000,
                get_fill_color=[255, 0, 0],
            )

            hist_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame(hist_coords, columns=["lat", "lon"]),
                get_position='[lon, lat]',
                get_radius=80000,
                get_fill_color=[0, 0, 255],
            )

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=2.5,
            )

            st.pydeck_chart(pdk.Deck(
                layers=[rec_layer, hist_layer],
                initial_view_state=view_state,
            ))

        st.markdown("🔴 Recommended | 🔵 Visited")
