# use the official typst image as a base image to get the typst binary
FROM ghcr.io/typst/typst:0.14.2 AS typst

# set base image
FROM python:3.11-slim

# set working directory
WORKDIR /app
# set environment variable to the project directory so that imports work correctly
ENV PYTHONPATH=/app

# install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# install typst
COPY --from=typst /bin/typst /usr/local/bin/typst

# get requirements file and install python dependencies
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# launch bash shell when the container starts
CMD ["/bin/bash"]
