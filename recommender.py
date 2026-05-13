import numpy as np
import pandas as pd

class Recommender:
    def __init__(self, df, user_emb, item_emb):
        self.df = df
        self.user_emb = user_emb
        self.item_emb = item_emb

        self.df['user_id'] = df['user'].astype('category').cat.codes
        self.df['poi_id'] = df['poi'].astype('category').cat.codes

        self.poi_mapping = dict(enumerate(df['poi'].astype('category').cat.categories))
        self.edges = self.df[['user_id', 'poi_id']].values

    def recommend(self, user_id, top_k=5):
        scores = self.user_emb[user_id] @ self.item_emb.T

        interacted = self.edges[self.edges[:, 0] == user_id][:, 1]
        scores = scores.copy()
        scores[interacted] = -1e9

        return scores.argsort()[-top_k:][::-1]

    def get_user_history(self, user_id):
        return self.edges[self.edges[:, 0] == user_id][:, 1]
