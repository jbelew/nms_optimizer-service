#!/bin/bash
python debugging_utils/generate_solves.py --generate-all --solver refine
sed -E 's/"\(([^)]*)\)"/(\1)/g' new_solves.json > formatted_solves.json
