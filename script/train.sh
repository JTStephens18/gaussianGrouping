#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi


dataset_folder="$1"
scale="$2"
# dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi


# Gaussian Grouping training
python train.py    -s $dataset_folder -r ${scale}  -m $dataset_folder/output --config_file config/gaussian_dataset/train.json

# Segmentation rendering using trained model
python render.py -m $dataset_folder/output --num_classes 4
