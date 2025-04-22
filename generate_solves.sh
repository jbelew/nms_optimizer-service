#!/bin/bash
python debugging_utils/generate_solves.py --generate-all
sed -E 's/"\(([^)]*)\)"/(\1)/g' new_solves.json > formatted_solves.json
