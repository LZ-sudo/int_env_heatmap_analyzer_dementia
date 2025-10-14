"""
Contrast Analysis Module
Calculates contrast risk between objects and backgrounds for dementia-friendly assessment.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class ContrastAnalysisModule:
    """Analyzes contrast ratios for dementia-friendly design compliance"""
    
    def __init__(self, contrast_threshold=1.5):
        self.CONTRAST_THRESHOLD = contrast_threshold
    
    def analyze_contrast_with_depth(self, image, depth_map, segments, detections):
        """
        Analyze contrast ratios using depth-aware segmentation
        
        Args:
            image: RGB image as numpy array
            depth_map: Depth map from depth estimation module
            segments: Segmented regions from depth estimation module
            detections: List of detected objects
            
        Returns:
            Tuple of (contrast_risk_map, contrast_details)
        """
        h, w = image.shape[:2]
        contrast_risk = np.zeros((h, w), dtype=np.float32)
        contrast_details = []
        
        # Convert to grayscale for luminance calculations
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
        
        # Analyze floor-wall transitions
        floor_mask = segments.get('floor_mask', np.zeros((h, w), dtype=bool))
        wall_mask = segments.get('wall_mask', np.zeros((h, w), dtype=bool))
        
        if floor_mask.any() and wall_mask.any():
            floor_lum = np.mean(gray[floor_mask])
            wall_lum = np.mean(gray[wall_mask])
            
            contrast_ratio = max(floor_lum, wall_lum) / (min(floor_lum, wall_lum) + 1e-6)
            
            if contrast_ratio < self.CONTRAST_THRESHOLD:
                # Mark floor-wall boundary as risky
                boundary = self._find_boundary(floor_mask, wall_mask)
                contrast_risk[boundary] = 1.0 - (contrast_ratio / self.CONTRAST_THRESHOLD)
                
                contrast_details.append({
                    'type': 'floor_wall_contrast',
                    'location': 'Floor-wall boundary',
                    'contrast_ratio': contrast_ratio,
                    'floor_luminance': floor_lum,
                    'wall_luminance': wall_lum,
                    'risk': 1.0 - (contrast_ratio / self.CONTRAST_THRESHOLD)
                })
        
        # Analyze object-background contrast for each detected object
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Get object region
            if det.get('mask') is not None:
                # Use precise SAM mask
                obj_mask = det['mask']
                obj_pixels = gray[obj_mask]
            else:
                # Fallback to bounding box
                obj_mask = np.zeros((h, w), dtype=bool)
                obj_mask[y1:y2, x1:x2] = True
                obj_pixels = gray[y1:y2, x1:x2].flatten()
            
            if len(obj_pixels) == 0:
                continue
            
            obj_lum = np.mean(obj_pixels)
            
            # Get surrounding background (dilate mask and subtract original)
            kernel = np.ones((15, 15), dtype=np.uint8)
            dilated_mask = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            bg_mask = dilated_mask & ~obj_mask
            
            if bg_mask.any():
                bg_pixels = gray[bg_mask]
                bg_lum = np.mean(bg_pixels)
                
                contrast_ratio = max(obj_lum, bg_lum) / (min(obj_lum, bg_lum) + 1e-6)
                
                if contrast_ratio < self.CONTRAST_THRESHOLD:
                    # Mark object as having poor contrast
                    risk_value = 1.0 - (contrast_ratio / self.CONTRAST_THRESHOLD)
                    contrast_risk[obj_mask] = np.maximum(contrast_risk[obj_mask], risk_value)
                    
                    contrast_details.append({
                        'type': 'object_background_contrast',
                        'location': f"{det['class']} at ({x1}, {y1})",
                        'object': det['class'],
                        'contrast_ratio': contrast_ratio,
                        'object_luminance': obj_lum,
                        'background_luminance': bg_lum,
                        'risk': risk_value
                    })
        
        # Smooth the risk map
        contrast_risk = gaussian_filter(contrast_risk, sigma=5)
        
        return contrast_risk, contrast_details
    
    def _find_boundary(self, mask1, mask2, thickness=10):
        """
        Find boundary between two masks
        
        Args:
            mask1: First boolean mask
            mask2: Second boolean mask
            thickness: Thickness of boundary region
            
        Returns:
            Boolean mask of boundary region
        """
        kernel = np.ones((thickness, thickness), dtype=np.uint8)
        
        # Dilate both masks
        dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1).astype(bool)
        dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1).astype(bool)
        
        # Boundary is where dilated masks overlap
        boundary = dilated1 & dilated2
        
        return boundary
