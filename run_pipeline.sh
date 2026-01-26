#!/bin/bash
# Master execution script

echo "Setting up environment..."
# Optional: virtualenv activation
# source venv/bin/activate

echo "Ensuring output directory exists..."
mkdir -p output

echo "Running Pipeline..."
python3 main.py

echo "Done."
