# set base image
FROM python:3.11-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# get requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project files to the container
COPY . .
# set environment variable to the project directory so that imports work correctly
ENV PYTHONPATH=/app

# launch bash shell when the container starts
CMD ["/bin/bash"]
