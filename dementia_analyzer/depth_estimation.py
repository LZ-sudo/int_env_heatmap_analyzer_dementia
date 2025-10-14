"""
Depth Estimation Module
Handles depth estimation using MiDaS model for accurate floor/wall/surface separation.
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter

try:
    import timm
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False


class DepthEstimationModule:
    """Handles depth estimation for spatial understanding"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.depth_model = None
        self.depth_transform = None
        self.depth_loaded = False
        
        self._load_midas()
    
    def _load_midas(self):
        """Load MiDaS depth estimation model"""
        if not MIDAS_AVAILABLE:
            print("✗ MiDaS not available - install with: pip install timm")
            return
        
        print("Loading MiDaS depth estimation model...")
        try:
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transform = midas_transforms.small_transform
            self.depth_loaded = True
            print("✓ MiDaS loaded successfully")
        except Exception as e:
            print(f"✗ MiDaS failed to load: {e}")
            self.depth_model = None
            self.depth_transform = None
    
    def estimate_depth(self, image):
        """
        Estimate depth map from RGB image
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Normalized depth map (0=far, 1=near)
        """
        if self.depth_model is None:
            return self._fallback_depth_estimation(image)
        
        h, w = image.shape[:2]
        
        # Prepare image for MiDaS
        input_batch = self.depth_transform(image).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-1 range (invert so 0=far, 1=near)
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = 1 - depth_map  # Invert
        
        return depth_map
    
    def _fallback_depth_estimation(self, image):
        """
        Simple fallback depth estimation when MiDaS is not available
        Uses gradient-based heuristic (bottom=near, top=far)
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Estimated depth map
        """
        h, w = image.shape[:2]
        
        # Simple gradient: assume bottom is floor (near) and top is ceiling/wall (far)
        depth_map = np.linspace(0, 1, h)[:, np.newaxis].repeat(w, axis=1)
        
        # Add some variation based on brightness (darker = potentially further)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
        depth_variation = (1 - gray) * 0.2
        depth_map = depth_map + depth_variation
        depth_map = np.clip(depth_map, 0, 1)
        
        return depth_map
    
    def segment_with_depth(self, image, depth_map, detections):
        """
        Segment image into regions using depth information and object detections
        
        Args:
            image: RGB image as numpy array
            depth_map: Depth map from estimate_depth
            detections: List of detected objects from segmentation module
            
        Returns:
            Dictionary of segmented regions
        """
        h, w = image.shape[:2]
        
        # Create depth-based regions
        depth_quantized = (depth_map * 5).astype(int)  # 5 depth levels
        
        segments = {
            'depth_map': depth_map,
            'depth_regions': depth_quantized,
            'detections': detections
        }
        
        # Identify likely floor region (bottom 40% of image with highest depth values)
        floor_mask = np.zeros((h, w), dtype=bool)
        floor_region = depth_map[int(h * 0.6):, :]
        floor_threshold = np.percentile(floor_region, 75)
        floor_mask[int(h * 0.6):, :] = depth_map[int(h * 0.6):, :] > floor_threshold
        segments['floor_mask'] = floor_mask
        
        # Identify likely wall regions (vertical surfaces with low depth variation)
        wall_mask = np.zeros((h, w), dtype=bool)
        wall_region = depth_map[:int(h * 0.7), :]
        
        # Detect vertical edges (walls typically have strong vertical gradients)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        vertical_strength = np.abs(sobely) > np.abs(sobelx)
        
        wall_mask[:int(h * 0.7), :] = (wall_region < np.percentile(wall_region, 50)) & vertical_strength[:int(h * 0.7), :]
        segments['wall_mask'] = wall_mask
        
        return segments
