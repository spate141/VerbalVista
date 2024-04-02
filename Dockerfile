FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    chromium \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip && rm -rf /root/.cache/pip

# Copy the .env file into the container at /app
#COPY .env ./

# Copy the rest of your application into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt && rm -rf /root/.cache/pip

# Expose the port Streamlit will run on
EXPOSE 8265 8000 8501 6379

# Health check for the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Define entrypoint and default command to run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
