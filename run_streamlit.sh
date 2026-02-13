#!/bin/bash
cd "$(dirname "$0")"
streamlit run ui/streamlit_app.py --server.port 8505
