#!/bin/bash

# Exit script on any error
set -e

# Start the Ray Server
echo "Starting Ray Server..."
bash manage_server.sh start_ray

# Start the application server with verbose logging
echo "Starting application server..."
bash manage_server.sh start_server -vvv

# Start the Streamlit app
echo "Starting Streamlit app..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
