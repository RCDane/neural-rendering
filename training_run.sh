#!/bin/sh 

### Arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <output_folder> <training_config>"
  exit 1
fi

output_folder=$1
training_config=$2

# Ensure output folder exists

if [ ! -d "$output_folder" ]; then
  echo "Creating output folder: $output_folder"
  mkdir -p "$output_folder"
fi

# Verify training config file exists
if [ ! -f "$training_config" ]; then
  echo "Config file not found: $training_config"
  exit 1
fi


bsub < training.sh "$output_folder" "$training_config"