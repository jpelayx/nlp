#!/bin/bash

while true; do
    python3 dataset.py
    if [ $? -eq 0 ]; then
        break
    fi
    echo "dataset.py ended with an error. Retrying..."
done
