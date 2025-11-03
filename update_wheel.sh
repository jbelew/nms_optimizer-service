#!/bin/bash

# This script automates the process of building a new release wheel for the
# nms-optimizer-service and updating the requirements.txt file to point to it.
# This is necessary for deploying the latest version of the Rust code to Heroku.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Building the release wheel..."
# Build the wheel using maturin and place it in the wheelhouse/ directory.
maturin build --release --out wheelhouse

echo "Finding the newly created wheel..."
# Find the newest .whl file in the wheelhouse directory.
# `ls -t` sorts files by modification time, newest first.
# `head -n 1` gets the first line, which is the newest file.
NEW_WHEEL_PATH=$(ls -t wheelhouse/*.whl | head -n 1)

if [ -z "$NEW_WHEEL_PATH" ]; then
    echo "Error: Could not find a wheel file in the wheelhouse/ directory after the build."
    exit 1
fi

echo "Found new wheel: $NEW_WHEEL_PATH"

echo "Updating requirements.txt..."
# Use sed to find the line containing the old wheelhouse path and replace it with the new one.
# The `|` character is used as a separator in sed to avoid issues with the '/' in the file path.
sed -i "s|wheelhouse/.*\.whl|$NEW_WHEEL_PATH|" requirements.txt

echo "Successfully updated requirements.txt."
echo ""
echo "Next steps:"
echo "1. Review the changes in requirements.txt."
echo "2. Commit the updated requirements.txt and the new wheel file in the wheelhouse/ directory."
echo "3. Push your changes to deploy the new version to Heroku."
