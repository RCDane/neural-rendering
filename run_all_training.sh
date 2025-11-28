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

# Step 2: Find all config files and submit training jobs
echo ""
echo "Submitting training jobs..."

find "$SOURCE_DIR" -name "default.yml" | while read config_file; do
    # Get the directory containing the config
    config_dir=$(dirname "$config_file")
    
    echo "Submitting job for: $config_dir"
    
    # Create a temporary job script for this config
    temp_script=$(mktemp)
    
    cat > "$temp_script" << EOF
#!/bin/sh
#BSUB -q gpua10
#BSUB -J neural_rendering_$(basename "$config_dir")
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1"
#BSUB -M 5GB
#BSUB -W 24:00
#BSUB -o $config_dir/Output_%J.out
#BSUB -e $config_dir/Output_%J.err

module load cuda/12.9.1
nvidia-smi

echo "Starting training job for $config_dir..."

source ../.nr-env/bin/activate
python3 training_combined.py --config "$config_file" > "$config_dir/output.log"
EOF
    
    # Submit the job
    bsub < "$temp_script"
    
    # Clean up temp script
    rm "$temp_script"
done

echo "All jobs submitted!"