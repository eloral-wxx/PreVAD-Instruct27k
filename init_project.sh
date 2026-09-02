#!/bin/bash
# ===================================================
# Project Initialization Script
# Functionality:
# 1. Clone HAWK, HolmesVAU, and UniEval repositories into the 'external/' folder in the current directory
# 2. Skip repositories that already exist
# ===================================================

# Get the directory where the script is located
BASE_DIR="$(cd "$(dirname "$0")"; pwd)"
EXTERNAL_DIR="$BASE_DIR/external"

# Create the external directory if it does not exist
mkdir -p "$EXTERNAL_DIR"

# Define repositories
declare -A REPOS
REPOS["hawk"]="https://github.com/jqtangust/hawk.git"
REPOS["HolmesVAU"]="https://github.com/pipixin321/HolmesVAU.git"
REPOS["UniEval"]="https://github.com/maszhongming/UniEval.git"

# Loop through and clone repositories
for NAME in "${!REPOS[@]}"; do
    REPO_URL=${REPOS[$NAME]}
    TARGET_DIR="$EXTERNAL_DIR/$NAME"
    
    if [ -d "$TARGET_DIR" ]; then
        echo "[SKIP] $NAME already exists, skipping: $TARGET_DIR"
    else
        echo "[CLONE] Cloning $NAME ..."
        git clone "$REPO_URL" "$TARGET_DIR"
    fi
done

echo "Initialization complete. All repositories have been cloned to $EXTERNAL_DIR"
