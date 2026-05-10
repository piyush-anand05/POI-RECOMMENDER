# POI Recommendation System (AdaGCL-inspired)

## Overview

This project is a graph-based recommendation system that suggests Points of Interest (POIs) to users using a Graph Neural Network (LightGCN) and contrastive learning.

## Features

* Graph-based user-item modeling
* LightGCN for embedding learning
* BPR loss for ranking
* Contrastive learning (dual graph views)
* Time embedding for temporal behavior
* Streamlit UI with map visualization

## How it works

* Users and POIs are represented as a bipartite graph
* The model learns embeddings using graph propagation
* Recommendations are generated using similarity scores
* Already visited POIs are filtered out

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Output

* Top recommended POIs
* Map visualization of locations

## Tech Stack

* PyTorch
* Torch Geometric
* Streamlit
* Folium
