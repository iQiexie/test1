import sys
import os

<<<<<<< HEAD
print("Current working directory:", os.getcwd())
print("sys.path before:", sys.path)

sys.path.extend(["/stable-diffusion-webui-forge-main"])

print("sys.path after:", sys.path)
print("Checking if directory exists:", os.path.exists("/stable-diffusion-webui-forge-main"))
print("Listing directory:", os.listdir("/stable-diffusion-webui-forge-main") if os.path.exists("/stable-diffusion-webui-forge-main") else "Directory does not exist")
=======
sys.path.extend(["/stable-diffusion-webui-forge-main"])
>>>>>>> cdfba27b038fa7b50a797f99438ac98016eee4d4

from modules import launch_utils

with launch_utils.startup_timer.subcategory("prepare environment"):
    launch_utils.prepare_environment()
