#!/bin/bash

# Create config files from templates if they don't exist
echo "Checking configuration files..."
if [ ! -f config/fusion_config.json ]; then
    echo "Creating fusion_config.json from template..."
    cp config/fusion_config_template.json config/fusion_config.json
fi

if [ ! -f config/models_config.json ]; then
    echo "Creating models_config.json from template..."
    cp config/models_config_template.json config/models_config.json
fi

if [ ! -f config/predictor_params.json ]; then
    echo "Creating predictor_params.json from template..."
    cp config/predictor_params_template.json config/predictor_params.json
fi

if [ ! -f config/labelstudio_api_key.txt ]; then
    echo "Creating labelstudio_api_key.txt from template..."
    cp config/labelstudio_api_key_template.txt config/labelstudio_api_key.txt
fi

# Create labelstudio_data directory if it doesn't exist
mkdir -p labelstudio_data

echo "Fixing permissions..."
sudo chmod -R 777 labelstudio_data/

echo "Starting Docker Compose..."
DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker compose up --build
