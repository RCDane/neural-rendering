#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    echo "Example: $0 output configs_only"
    exit 1
fi

# Source and destination directories from arguments
SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Step 1: Copy folder structure with only default.yml files
echo "Copying config files from $SOURCE_DIR to $DEST_DIR..."
mkdir -p "$DEST_DIR"

find "$SOURCE_DIR" -name "default.yml" | while read config_file; do
    # Get relative path from source directory
    rel_path="${config_file#$SOURCE_DIR/}"
    dest_file="$DEST_DIR/$rel_path"
    
    # Create directory structure
    mkdir -p "$(dirname "$dest_file")"
    
    # Copy config file
    cp "$config_file" "$dest_file"
    echo "Copied: $config_file -> $dest_file"
done

# Step 2: Run training for each config file iteratively
echo ""
echo "Running training jobs sequentially..."

source ../.nr-env/bin/activate

find "$SOURCE_DIR" -name "default.yml" | while read config_file; do
    # Get the directory containing the config
    config_dir=$(dirname "$config_file")
    rel_path="${config_file#$SOURCE_DIR/}"
    dest_config_dir=$(dirname "$DEST_DIR/$rel_path")
    dest_config_file="$DEST_DIR/$rel_path"
    
    echo "========================================="
    echo "Starting training for: $config_dir"
    echo "Config file: $dest_config_file"
    echo "========================================="
    
    # Run training
    python3 training_combined.py --config "$dest_config_file" > "$dest_config_dir/output.log" 2>&1
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed successfully: $config_dir"
    else
        echo "✗ Failed with exit code $exit_code: $config_dir"
    fi
    echo ""
done

echo "All training jobs completed!"