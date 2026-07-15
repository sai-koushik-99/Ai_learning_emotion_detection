#!/bin/bash
# Force install torch CPU if not present
python -c "import torch" 2>/dev/null || pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
