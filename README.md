1. Install venv (if not already installed)

Most Python installations already include venv. Check with:

python3 -m venv --help

2. Create a new virtual environment

Decide on a folder name for your environment, e.g., cv_env:

python3 -m venv cv_env

This creates a folder cv_env containing a fresh Python environment.

3. Activate the virtual environment

Windows (cmd):

cv_env\Scripts\activate

macOS/Linux:

source cv_env/bin/activate

4. Install the required packages

Now install your packages in this clean environment:

pip install numpy opencv-python matplotlib pillow scikit-learn

5. Verify installation

Run Python:

python

Then try:

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN

print("All imports successful!")



運行專案：
source cv_env/bin/activate

python cv_env/app.py
