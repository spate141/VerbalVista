FROM python:3.10-slim as base

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

# Upgrade pip and install Python dependencies in a single step, then clean up cache
# COPY only the files needed for pip install to leverage Docker's cache
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && rm -rf /root/.cache/pip

# Now copy the rest of your application
COPY . /app

# Make sure start.sh is executable
RUN chmod +x /app/entrypoint_start.sh

# Expose necessary ports
EXPOSE 8265 8000 8501 6379

# Health check for the container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Define the default command to run the app
CMD ["./entrypoint_start.sh"]
