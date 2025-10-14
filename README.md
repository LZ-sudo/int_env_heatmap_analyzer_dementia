# Dementia-Friendly Interior Assessment Tool

A modular Python tool for analyzing household interiors for dementia-friendly design compliance.

## Features

- **YOLO Object Detection**: Identifies furniture and fixtures
- **SAM Segmentation**: Pixel-perfect object segmentation
- **MiDaS Depth Estimation**: Accurate floor/wall/surface separation
- **Contrast Analysis**: Measures contrast ratios for visibility
- **Pattern Complexity Analysis**: Detects cognitively overwhelming patterns
- **Visual Reports**: Generates detailed heatmaps and recommendations

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- ultralytics (YOLO)
- segment-anything (SAM)
- timm (MiDaS)
- opencv-python
- matplotlib
- scipy
- torch
- numpy

## Project Structure

```
dementia_analyzer/
├── __init__.py              # Package initialization
├── main.py                  # Main orchestrator
├── segmentation.py          # YOLO + SAM object detection
├── depth_estimation.py      # MiDaS depth estimation
├── contrast_analysis.py     # Contrast risk calculations
├── pattern_analysis.py      # Pattern complexity calculations
└── visualization.py         # Output generation

visual_heatmap_label_gen.py  # Standalone script wrapper
```

## Usage

### Option 1: Run as standalone script (like original)
```bash
python visual_heatmap_label_gen.py images/
python visual_heatmap_label_gen.py images/ --recursive
python visual_heatmap_label_gen.py image1.jpg image2.jpg
```

### Option 2: Run as module
```bash
python -m dementia_analyzer.main images/
```

### Option 3: Use in your own code
```python
from dementia_analyzer import DementiaFriendlyAnalyzer

analyzer = DementiaFriendlyAnalyzer()
risk_map, report = analyzer.analyze_image('path/to/image.jpg')
```

## Module Descriptions

### `main.py`
Main orchestrator that coordinates all components. Contains the `DementiaFriendlyAnalyzer` class which manages the analysis pipeline.

**Key Methods:**
- `__init__()`: Initialize all modules
- `analyze_image(image_path)`: Run complete analysis on an image
- `combine_risks()`: Combine contrast and pattern risks

### `segmentation.py`
Handles object detection and segmentation.

**Key Class:** `SegmentationModule`
- `detect_objects(image)`: YOLO object detection
- `get_sam_masks(image, detections)`: SAM precise segmentation

### `depth_estimation.py`
Estimates depth for spatial understanding.

**Key Class:** `DepthEstimationModule`
- `estimate_depth(image)`: MiDaS depth prediction
- `segment_with_depth()`: Depth-based region segmentation

### `contrast_analysis.py`
Analyzes contrast for visibility issues.

**Key Class:** `ContrastAnalysisModule`
- `analyze_contrast_with_depth()`: Depth-aware contrast analysis

### `pattern_analysis.py`
Detects complex patterns that increase cognitive load.

**Key Class:** `PatternAnalysisModule`
- `analyze_pattern_complexity_per_object()`: Object-wise pattern analysis

### `visualization.py`
Generates all visual outputs and reports.

**Key Class:** `VisualizationModule`
- `save_results()`: Create analysis visualizations
- `generate_detailed_report()`: Create text report with recommendations

## Output

The tool generates two output files per image:

1. **`{filename}_analysis.png`**: Comprehensive 6-panel analysis showing:
   - Detected objects with annotations
   - Depth map
   - Contrast risk heatmap
   - Pattern complexity heatmap
   - Overall risk heatmap
   - Risk overlay on original image

2. **`{filename}_overlay.png`**: Clean overlay with legend

Console output includes:
- Detection statistics
- Risk statistics
- Detailed findings
- Object-specific recommendations
- Overall risk assessment

## Example

```python
from dementia_analyzer import DementiaFriendlyAnalyzer

# Initialize analyzer
analyzer = DementiaFriendlyAnalyzer()

# Analyze an image
risk_map, report = analyzer.analyze_image('bedroom.jpg', output_dir='results')

# Access individual modules if needed
detections = analyzer.segmentation.detect_objects(image)
depth_map = analyzer.depth_estimation.estimate_depth(image)
```

## Customization

You can customize thresholds when initializing modules:

```python
from dementia_analyzer.main import DementiaFriendlyAnalyzer
from dementia_analyzer.contrast_analysis import ContrastAnalysisModule
from dementia_analyzer.pattern_analysis import PatternAnalysisModule

analyzer = DementiaFriendlyAnalyzer()
analyzer.contrast_analysis = ContrastAnalysisModule(contrast_threshold=2.0)
analyzer.pattern_analysis = PatternAnalysisModule(pattern_threshold=0.25)
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~6GB VRAM for GPU mode

## License

MIT License
