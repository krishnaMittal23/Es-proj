#!/bin/bash
# Installs ultralytics (YOLOv8) + CUDA torch into a venv; build ROS package
set -e
python3 -m venv --system-site-packages ~/camo_venv
~/camo_venv/bin/pip install -q --upgrade pip
~/camo_venv/bin/pip install -q "numpy<2" ultralytics
~/camo_venv/bin/python -c "import torch,ultralytics;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'ultralytics',ultralytics.__version__)"
echo ML_DONE
