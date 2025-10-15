"""
Main Orchestrator Module
Coordinates all analysis components and provides the main interface.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import torch

from .segmentation import SegmentationModule
from .depth_estimation import DepthEstimationModule
from .contrast_analysis import ContrastAnalysisModule
from .pattern_analysis import PatternAnalysisModule
from .visualization import VisualizationModule


class DementiaFriendlyAnalyzer:
    """
    Main analyzer that orchestrates all components for dementia-friendly interior assessment
    """
    
    def __init__(self):
        """Initialize all analysis modules"""
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"INITIALIZATION DIAGNOSTICS")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Initialize all modules
        print("\nInitializing modules...")
        self.segmentation = SegmentationModule(device=str(self.device))
        self.depth_estimation = DepthEstimationModule(device=str(self.device))
        self.contrast_analysis = ContrastAnalysisModule(contrast_threshold=1.5)
        self.pattern_analysis = PatternAnalysisModule(complexity_threshold=0.45)  # Tune this: 0.35-0.45
        self.visualization = VisualizationModule()
        
        print(f"\n{'='*70}\n")
        
        # Print warnings if modules failed to load
        if not self.segmentation.yolo_loaded:
            print("⚠ WARNING: Running without YOLO - object detection disabled")
        if not self.depth_estimation.depth_loaded:
            print("⚠ WARNING: Running without MiDaS - using simple depth estimation")
        if not self.segmentation.sam_loaded:
            print("⚠ WARNING: Running without SAM - using bounding box approximations")
        print()
    
    def analyze_image(self, image_path, output_dir='output'):
        """
        Main analysis pipeline - analyzes a single image
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save outputs
            
        Returns:
            Tuple of (overall_risk_map, text_report)
        """
        print(f"\nAnalyzing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # # Step 1: Detect objects using YOLO + SAM
        # print("  - Detecting objects with YOLO...")
        # detections = self.segmentation.detect_objects(image_rgb)
        # print(f"    Found {len(detections)} objects")

        # Step 1: Detect objects using YOLO + SAM hybrid approach
        print("  - Detecting objects (hybrid: YOLO + SAM everything)...")
        detections = self.segmentation.detect_objects_hybrid(image_rgb)
        print(f"    Found {len(detections)} segments")
        
        # Step 2: Estimate depth
        print("  - Estimating depth...")
        depth_map = self.depth_estimation.estimate_depth(image_rgb)
        
        # Step 3: Segment image regions using depth
        print("  - Segmenting image regions...")
        segments = self.depth_estimation.segment_with_depth(image_rgb, depth_map, detections)
        
        # Step 4: Analyze contrast
        print("  - Analyzing contrast ratios...")
        contrast_risk, contrast_details = self.contrast_analysis.analyze_contrast_with_depth(
            image_rgb, depth_map, segments, detections
        )
        print(f"    Found {len(contrast_details)} contrast issues")
        
        # # Step 5: Analyze pattern complexity
        # print("  - Analyzing pattern complexity...")
        # pattern_risk, pattern_details = self.pattern_analysis.analyze_pattern_complexity_per_object(
        #     image_rgb, detections
        # )
        # print(f"    Found {len(pattern_details)} pattern issues")

        # Step 5: Analyze pattern complexity on all detected segments
        print("  - Analyzing pattern complexity...")
        pattern_risk, pattern_details = \
            self.pattern_analysis.analyze_pattern_complexity_per_object(
                image_rgb, detections
            )
        print(f"    Found {len(pattern_details)} pattern issues")
        
        # Step 6: Combine risks
        overall_risk = self.combine_risks(contrast_risk, pattern_risk)
        
        # Step 7: Generate visualizations
        self.visualization.save_results(
            image_rgb, contrast_risk, pattern_risk, overall_risk,
            depth_map, detections, contrast_details, image_path, output_dir
        )
        
        # Step 8: Generate text report
        report = self.visualization.generate_detailed_report(
            contrast_risk, pattern_risk, contrast_details,
            pattern_details, detections
        )
        print(report)
        
        return overall_risk, report
    
    def combine_risks(self, contrast_risk, pattern_risk):
        """
        Combine contrast and pattern risks into overall risk assessment
        
        Args:
            contrast_risk: Contrast risk map (0-1)
            pattern_risk: Pattern risk map (0-1)
            
        Returns:
            Combined risk map (0-1)
        """
        # Weighted combination: 50% contrast, 50% pattern
        overall_risk = (contrast_risk * 0.5 + pattern_risk * 0.5)
        return overall_risk


def main():
    """Main execution function"""
    import sys
    
    print("="*70)
    print("Enhanced Dementia-Friendly Interior Assessment Tool")
    print("="*70)
    print("\nFeatures:")
    print("  • YOLO object detection for identifying furniture and fixtures")
    print("  • SAM (Segment Anything) for pixel-perfect object segmentation")
    print("  • MiDaS depth estimation for accurate floor/wall separation")
    print("  • Contrast ratio analysis (object-to-background boundaries)")
    print("  • Pattern complexity analysis (using precise object masks)")
    print("  • Object-specific recommendations\n")
    
    # Process images
    if len(sys.argv) > 1:
        # Check if first argument is a directory
        first_arg = sys.argv[1]
        if os.path.isdir(first_arg):
            # Process all images in the specified directory
            image_paths = []
            dir_path = Path(first_arg)
            
            # Check for --recursive flag
            recursive = '--recursive' in sys.argv or '-r' in sys.argv
            
            if recursive:
                # Search recursively in subdirectories
                for ext in ['jpg', 'jpeg', 'png', 'bmp']:
                    image_paths.extend(dir_path.rglob(f'*.{ext}'))
                print(f"Searching recursively in {first_arg}")
            else:
                # Only search in specified directory
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_paths.extend(dir_path.glob(ext))
            
            if not image_paths:
                print(f"No images found in directory: {first_arg}")
                return
            print(f"Found {len(image_paths)} images in {first_arg}")
        else:
            # Process individual image files
            image_paths = sys.argv[1:]
    else:
        # Default: process all images in current directory
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(Path('.').glob(ext))
        
        if not image_paths:
            print("No images found.")
            print("\nUsage:")
            print("  python -m dementia_analyzer.main images/                    # Process all images in 'images' folder")
            print("  python -m dementia_analyzer.main images/ --recursive        # Include subdirectories")
            print("  python -m dementia_analyzer.main image1.jpg image2.jpg      # Process specific images")
            print("  python -m dementia_analyzer.main                            # Process all images in current folder")
            return
    
    # Create analyzer
    analyzer = DementiaFriendlyAnalyzer()
    
    # Process each image
    for image_path in image_paths:
        try:
            analyzer.analyze_image(str(image_path))
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("Analysis complete! Check the 'output' folder for results.")
    print("="*70)


if __name__ == "__main__":
    main()
