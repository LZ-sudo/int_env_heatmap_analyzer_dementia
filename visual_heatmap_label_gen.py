#!/usr/bin/env python3
"""
Enhanced Dementia-Friendly Interior Assessment Tool
Analyzes household interior images for:
1. Contrast ratios (object-to-background boundaries using depth estimation)
2. Pattern complexity (busy patterns that increase cognitive load)

Uses YOLO for object detection, SAM for precise segmentation, and MiDaS for depth estimation.
Generates heatmaps showing risk areas with object-specific recommendations.

Installation:
    pip install ultralytics segment-anything timm opencv-python matplotlib scipy torch

Usage:
    python visual_heatmap_label_gen.py images/                    # Process all images in folder
    python visual_heatmap_label_gen.py images/ --recursive        # Include subdirectories
    python visual_heatmap_label_gen.py image1.jpg image2.jpg      # Process specific images
    python visual_heatmap_label_gen.py                            # Process all images in current folder
"""

# Import the modular implementation
from dementia_analyzer.main import main

if __name__ == "__main__":
    main()
