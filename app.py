import streamlit as st
import torch
import pandas as pd
import folium

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

# --- graph ---
edge_list = []
for u, i in edges:
    edge_list.append([u, i + num_users])
    edge_list.append([i + num_users, u])

edge_index = torch.tensor(edge_list).t().contiguous()

# --- model ---
from torch_geometric.nn import LGConv
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32, num_layers=2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.embedding = nn.Embedding(num_users + num_items, emb_dim)
        self.time_emb = nn.Embedding(24, emb_dim)   # ✅ ADD THIS

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index, time_ids=None):
        x = self.embedding.weight

        if time_ids is not None:
            x = x + self.time_emb(time_ids % 24)

        all_embs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_embs.append(x)

        x = torch.stack(all_embs).mean(dim=0)

        return x[:self.num_users], x[self.num_users:]

# --- load model ---
model = LightGCN(num_users, num_items)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

time_ids = torch.zeros(num_users + num_items, dtype=torch.long)
user_emb, item_emb = model(edge_index, time_ids)

# --- recommend ---
def recommend(user_id, top_k=5):
    scores = torch.matmul(user_emb[user_id], item_emb.T)

    interacted = edges[edges[:, 0] == user_id][:, 1]
    for item in interacted:
        scores[item] = -1e9

    return torch.topk(scores, top_k).indices

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

    m = folium.Map(location=coords[0], zoom_start=12)

    for c in coords:
        folium.Marker(c).add_to(m)

    st.components.v1.html(m._repr_html_(), height=500)