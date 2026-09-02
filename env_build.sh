#!/bin/bash

ENV_NAME="videollava_tuning"
ENV_FILE="environment.yml"

echo "📦 Checking environment file..."
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found in current directory!"
    exit 1
fi

echo "🔍 Checking if environment '$ENV_NAME' already exists..."
if conda env list | grep -q "$ENV_NAME"; then
    echo "⚠️ Environment $ENV_NAME already exists. Deleting it first..."
    conda env remove -n $ENV_NAME -y
fi

echo "🚀 Creating conda environment: $ENV_NAME ..."
conda env create -f $ENV_FILE -n $ENV_NAME

echo "✨ Environment created successfully!"
echo "To activate it, run:"
echo "👉 conda activate $ENV_NAME"
