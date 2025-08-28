import subprocess
import pkg_resources
import logging

# Step 1: Freeze current environment
with open("old-requirements.txt", "w") as f:
    subprocess.run(["pip", "freeze"], stdout=f)

# Step 2: Install pipreqs if not already installed
subprocess.run(["pip", "install", "pipreqs"])

# Step 3: Generate requirements from used imports
subprocess.run(["pipreqs", ".", "--force"])

# Step 4: Read both sets of requirements
def read_packages(path):
    with open(path, "r") as f:
        return set(line.strip().split("==")[0].lower() for line in f if "==" in line)

installed = read_packages("old-requirements.txt")
used = read_packages("requirements.txt")

# Step 5: Find unused packages
unused = installed - used

# Step 6: Uninstall unused packages
if unused:
    logging.info(f"Uninstalling unused packages: {unused}")
    subprocess.run(["pip", "uninstall", "-y", *unused])
else:
    logging.info("No unused packages found.")
