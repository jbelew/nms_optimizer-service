#!/usr/bin/env python3
import re
from pathlib import Path

# Directory containing the JSON files
modules_data_dir = Path("src/data_definitions/modules_data")

def replace_label(match):
    """Replace label if it doesn't already have a bracket prefix"""
    prefix = match.group(1)  # "Theta", "Tau", or "Sigma"
    label = match.group(2)   # the full label text
    
    # Check if label already starts with a bracket pattern like [R1], [C1], etc.
    if re.match(r'^\[[A-Z]\d+\]\s', label):
        return match.group(0)  # Return unchanged
    
    # Map prefix to bracket number
    bracket_map = {"Theta": "[1]", "Tau": "[2]", "Sigma": "[3]"}
    bracket = bracket_map[prefix]
    
    return f'"label": "{bracket} {label}"'

def process_file(file_path):
    """Process a single JSON file using regex"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Match "label": "...Theta..." and similar
    pattern = r'"label":\s*"([^"]*)(Theta|Tau|Sigma)([^"]*)"'
    
    def replacer(match):
        full_label = match.group(1) + match.group(2) + match.group(3)
        prefix_type = match.group(2)  # Theta, Tau, or Sigma
        
        # Check if label already starts with a bracket pattern
        if re.match(r'^\[[A-Z]\d+\]\s', full_label):
            return match.group(0)  # Return unchanged
        
        bracket_map = {"Theta": "[1]", "Tau": "[2]", "Sigma": "[3]"}
        bracket = bracket_map[prefix_type]
        
        return f'"label": "{bracket} {full_label}"'
    
    content = re.sub(pattern, replacer, content)
    
    # Only write if changes were made
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated: {file_path}")
    else:
        print(f"No changes: {file_path}")

# Process all JSON files
for json_file in sorted(modules_data_dir.glob("*.json")):
    process_file(json_file)

print("\nAll files processed successfully!")
