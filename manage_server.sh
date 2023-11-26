#!/bin/bash

# Function to display usage guide
usage() {
    echo "Usage: bash manage_server.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start_ray               Start the Ray server."
    echo "  start_server [dir]      Start the application server with the specified index directory."
    echo "  stop_server             Stop the application server."
    echo "  stop_ray                Stop the Ray server."
    echo ""
    echo "Examples:"
    echo "  bash manage_server.sh start_ray"
    echo "  bash manage_server.sh start_server data/indices/stories"
    echo "  bash manage_server.sh stop_server"
    echo "  bash manage_server.sh stop_ray"
    echo ""
    echo "Please provide a valid command to manage the server."
}

# Function to start Ray server
start_ray() {
    echo "Starting Ray server..."
    ray start --head
}

# Function to start the application server with index directory
start_server() {
    if [ "$#" -ne 1 ]; then
        echo "Missing index directory for starting server."
        usage
        exit 1
    fi

    echo "Starting application server with index directory $1..."
    python serve.py --index_dir="$1"
}

# Function to stop the application server
stop_server() {
    echo "Stopping application server..."
    python -c "from ray import serve; serve.shutdown()"
}

# Function to stop Ray server
stop_ray() {
    echo "Stopping Ray server..."
    ray stop
}

# Check arguments and execute corresponding functions
case $1 in
    start_ray)
        start_ray
        ;;
    start_server)
        start_server $2
        ;;
    stop_server)
        stop_server
        ;;
    stop_ray)
        stop_ray
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo "Operation completed."
