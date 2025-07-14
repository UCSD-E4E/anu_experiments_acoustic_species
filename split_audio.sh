#!/bin/bash

# Duration for each chunk in seconds
CHUNK_DURATION=5

# Recursively find all .wav files
find . -type f -iname "*.wav" | while read -r file; do
    echo "Processing: $file"

    # Extract directory and clean filename (no path, no extension)
    dir=$(dirname "$file")
    base=$(basename "$file")
    filename="${base%.*}"  # Remove extension safely

    # Output pattern: same dir, no double extension
    output_pattern="$dir/${filename}_%03d.wav"

    # Split with ffmpeg: preserve metadata
    ffmpeg -hide_banner -loglevel error \
        -i "$file" \
        -f segment \
        -segment_time $CHUNK_DURATION \
        -map_metadata 0 \
        -c copy \
        "$output_pattern"

    # If successful, delete original
    if [ $? -eq 0 ]; then
        echo "Deleting original file: $file"
        rm "$file"
    else
        echo "Error splitting $file — original not deleted."
    fi
done
