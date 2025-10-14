"""
Pattern Analysis Module
Analyzes pattern complexity on surfaces and objects for dementia-friendly assessment.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, uniform_filter


class PatternAnalysisModule:
    """Analyzes pattern complexity for cognitive load assessment"""
    
    def __init__(self, pattern_threshold=0.18):
        self.PATTERN_COMPLEXITY_THRESHOLD = pattern_threshold
    
    def _analyze_frequency_domain(self, image, mask):
        """
        Analyze pattern using FFT to detect periodic/repeating patterns
        Particularly good for: wallpapers, striped fabrics, geometric patterns
        """
        if not mask.any():
            return 0.0
        
        # Apply FFT to detect periodic patterns
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Look at high-frequency content (excluding DC component)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Mask out low frequencies (center region)
        magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10] = 0
        
        # Calculate high-frequency energy (indicates busy patterns)
        high_freq_energy = np.sum(magnitude_spectrum[mask]) / np.sum(mask)
        
        # Detect peaks in frequency domain (indicates repeating patterns)
        # More peaks = more complex/regular pattern
        threshold = np.percentile(magnitude_spectrum[mask], 95)
        num_peaks = np.sum(magnitude_spectrum[mask] > threshold)
        peak_score = num_peaks / np.sum(mask)
        
        return np.clip((high_freq_energy * 0.0001 + peak_score * 10), 0, 1)

    def _analyze_chromatic_complexity(self, image_rgb, mask):
        """
        Analyze color-based patterns (floral prints, multi-colored fabrics)
        """
        if not mask.any():
            return 0.0
        
        # Convert to HSV for better color analysis
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Analyze hue variation (color diversity)
        hue = image_hsv[:, :, 0].astype(float)
        hue_variance = np.var(hue[mask])
        hue_score = np.clip(hue_variance / 1000, 0, 1)
        
        # Analyze saturation variation (color intensity patterns)
        saturation = image_hsv[:, :, 1].astype(float)
        sat_variance = np.var(saturation[mask])
        sat_score = np.clip(sat_variance / 2000, 0, 1)
        
        # Local color variation (indicates busy color patterns)
        from scipy.ndimage import uniform_filter
        local_hue_var = uniform_filter(hue ** 2, size=9) - uniform_filter(hue, size=9) ** 2
        local_color_complexity = np.mean(local_hue_var[mask])
        
        return (hue_score * 0.4 + sat_score * 0.3 + local_color_complexity * 0.01) * 2

    # def analyze_pattern_complexity_per_object(self, image, detections):
    #     """
    #     Analyze pattern complexity for each detected object
        
    #     Args:
    #         image: RGB image as numpy array
    #         detections: List of detected objects from segmentation module
            
    #     Returns:
    #         Tuple of (pattern_risk_map, pattern_details)
    #     """
    #     h, w = image.shape[:2]
    #     pattern_risk = np.zeros((h, w), dtype=np.float32)
    #     pattern_details = []
        
    #     # Convert to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    #     # Analyze each object
    #     for det in detections:
    #         x1, y1, x2, y2 = det['bbox']
            
    #         # Get object region
    #         if det.get('mask') is not None:
    #             # Use precise SAM mask
    #             obj_mask = det['mask']
                
    #             # Get bounding box of mask for cropping
    #             mask_coords = np.argwhere(obj_mask)
    #             if len(mask_coords) == 0:
    #                 continue
                
    #             y_min, x_min = mask_coords.min(axis=0)
    #             y_max, x_max = mask_coords.max(axis=0)
                
    #             # Crop to mask region
    #             mask_crop = obj_mask[y_min:y_max+1, x_min:x_max+1]
    #             region_crop = gray[y_min:y_max+1, x_min:x_max+1]
    #         else:
    #             # Fallback to bounding box
    #             obj_mask = np.zeros((h, w), dtype=bool)
    #             obj_mask[y1:y2, x1:x2] = True
    #             mask_crop = np.ones((y2-y1, x2-x1), dtype=bool)
    #             region_crop = gray[y1:y2, x1:x2]
            
    #         if region_crop.size == 0:
    #             continue
            
    #         # Analyze pattern complexity within this object only
    #         complexity_score, complexity_type = self._analyze_region_pattern(
    #             region_crop, mask_crop
    #         )
            
    #         if complexity_score > self.PATTERN_COMPLEXITY_THRESHOLD:
    #             # Mark object with pattern risk
    #             risk_value = (complexity_score - self.PATTERN_COMPLEXITY_THRESHOLD) / \
    #                         (1.0 - self.PATTERN_COMPLEXITY_THRESHOLD)
    #             risk_value = np.clip(risk_value, 0, 1)
                
    #             pattern_risk[obj_mask] = np.maximum(pattern_risk[obj_mask], risk_value)
                
    #             pattern_details.append({
    #                 'type': 'pattern_complexity',
    #                 'location': f"{det['class']} at ({x1}, {y1})",
    #                 'object': det['class'],
    #                 'complexity_score': complexity_score,
    #                 'pattern_type': complexity_type,
    #                 'risk': risk_value
    #             })
        
    #     # Smooth the risk map
    #     pattern_risk = gaussian_filter(pattern_risk, sigma=5)
        
    #     return pattern_risk, pattern_details

    def analyze_pattern_complexity_per_object(self, image, detections):
        """
        Analyze pattern complexity for each detected object (UPDATED)
        """
        h, w = image.shape[:2]
        pattern_risk = np.zeros((h, w), dtype=np.float32)
        pattern_details = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Analyze each object
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Get object region
            if det.get('mask') is not None:
                obj_mask = det['mask']
                mask_coords = np.argwhere(obj_mask)
                if len(mask_coords) == 0:
                    continue
                
                y_min, x_min = mask_coords.min(axis=0)
                y_max, x_max = mask_coords.max(axis=0)
                
                mask_crop = obj_mask[y_min:y_max+1, x_min:x_max+1]
                region_crop = gray[y_min:y_max+1, x_min:x_max+1]
                region_rgb_crop = image[y_min:y_max+1, x_min:x_max+1]  # NEW: Pass RGB
            else:
                obj_mask = np.zeros((h, w), dtype=bool)
                obj_mask[y1:y2, x1:x2] = True
                mask_crop = np.ones((y2-y1, x2-x1), dtype=bool)
                region_crop = gray[y1:y2, x1:x2]
                region_rgb_crop = image[y1:y2, x1:x2]  # NEW: Pass RGB
            
            if region_crop.size == 0:
                continue
            
            # Analyze pattern complexity with RGB data
            complexity_score, complexity_type = self._analyze_region_pattern(
                region_crop, mask_crop, image, region_rgb_crop  # Pass RGB
            )
            
            # Rest remains the same...
            if complexity_score > self.PATTERN_COMPLEXITY_THRESHOLD:
                risk_value = (complexity_score - self.PATTERN_COMPLEXITY_THRESHOLD) / \
                            (1.0 - self.PATTERN_COMPLEXITY_THRESHOLD)
                risk_value = np.clip(risk_value, 0, 1)
                
                pattern_risk[obj_mask] = np.maximum(pattern_risk[obj_mask], risk_value)
                
                pattern_details.append({
                    'type': 'pattern_complexity',
                    'location': f"{det['class']} at ({x1}, {y1})",
                    'object': det['class'],
                    'complexity_score': complexity_score,
                    'pattern_type': complexity_type,
                    'risk': risk_value
                })
        
        # Smooth the risk map
        pattern_risk = gaussian_filter(pattern_risk, sigma=5)
        
        return pattern_risk, pattern_details
    
    def _calculate_local_variance(self, image, mask, window=5):
        """
        Calculate local variance using a sliding window
        
        Args:
            image: Grayscale image
            mask: Boolean mask
            window: Window size for local variance
            
        Returns:
            Local variance map
        """
        
        # Local mean and variance
        local_mean = uniform_filter(image, size=window)
        local_mean_sq = uniform_filter(image ** 2, size=window)
        local_var = local_mean_sq - local_mean ** 2
        
        return local_var
    
    # def _detect_directional_patterns(self, image, mask):
    #     """
    #     Detect directional patterns (stripes, lines) using Gabor filters
        
    #     Args:
    #         image: Grayscale image (normalized 0-1)
    #         mask: Boolean mask
            
    #     Returns:
    #         Directional pattern score
    #     """
    #     # Test multiple orientations
    #     orientations = [0, 45, 90, 135]
    #     frequencies = [0.1, 0.2, 0.3]
        
    #     max_response = 0
        
    #     for theta in orientations:
    #         for frequency in frequencies:
    #             # Create Gabor kernel
    #             kernel = cv2.getGaborKernel(
    #                 ksize=(21, 21),
    #                 sigma=3.0,
    #                 theta=np.radians(theta),
    #                 lambd=1.0/frequency,
    #                 gamma=0.5,
    #                 psi=0
    #             )
                
    #             # Apply filter
    #             filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
                
    #             # Measure response strength in masked region
    #             response = np.std(filtered[mask])
    #             max_response = max(max_response, response)
        
    #     return np.clip(max_response * 2, 0, 1)
    
    def _detect_directional_patterns(self, image, mask):
        """
        Enhanced directional pattern detection with more scales and orientations
        Better for: stripes, plaids, geometric wallpapers
        """
        # More comprehensive orientation and scale testing
        orientations = [0, 30, 45, 60, 90, 120, 135, 150]  # More angles
        frequencies = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]    # More scales
        
        responses = []
        
        for theta in orientations:
            for frequency in frequencies:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    ksize=(31, 31),        # Larger kernel for fabric textures
                    sigma=4.0,             # Wider for softer patterns
                    theta=np.radians(theta),
                    lambd=1.0/frequency,
                    gamma=0.5,
                    psi=0
                )
                
                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
                
                # Measure response strength in masked region
                response = np.std(filtered[mask])
                responses.append(response)
        
        # Check for strong directional preference (indicates striped/geometric patterns)
        max_response = np.max(responses)
        mean_response = np.mean(responses)
        
        # High max with low mean = strong directional pattern
        directionality = (max_response - mean_response) / (mean_response + 1e-6)
        
        return np.clip(max_response * 2 + directionality * 0.5, 0, 1)
    
    def _detect_repeating_patterns(self, image, mask):
        """
        Use autocorrelation to detect repeating patterns
        Excellent for: wallpapers, tiled patterns, regular fabric prints
        """
        if not mask.any():
            return 0.0
        
        # Get the masked region
        coords = np.argwhere(mask)
        if len(coords) < 100:
            return 0.0
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        region = image[y_min:y_max+1, x_min:x_max+1]
        
        # Compute autocorrelation using FFT (faster)
        f = np.fft.fft2(region)
        f_conj = np.conj(f)
        autocorr = np.fft.ifft2(f * f_conj).real
        autocorr = np.fft.fftshift(autocorr)
        
        # Normalize
        autocorr = autocorr / autocorr.max()
        
        # Look for peaks away from center (indicates repetition)
        h, w = autocorr.shape
        center_h, center_w = h // 2, w // 2
        
        # Mask out center peak
        autocorr[center_h-5:center_h+5, center_w-5:center_w+5] = 0
        
        # Count significant peaks (repetition score)
        threshold = 0.3  # 30% of max correlation
        num_peaks = np.sum(autocorr > threshold)
        repetition_score = np.clip(num_peaks / 100, 0, 1)
        
        return repetition_score

    # def _analyze_region_pattern(self, region, mask):
    #     """
    #     Analyze pattern complexity within a masked region
        
    #     Args:
    #         region: Grayscale image region
    #         mask: Boolean mask indicating which pixels to analyze
            
    #     Returns:
    #         Tuple of (complexity_score, complexity_type)
    #     """
    #     if region.size == 0 or not mask.any():
    #         return 0.0, "none"
        
    #     # Create analysis region: fill exterior with mean to avoid edge artifacts
    #     analysis_region = region.copy().astype(float)
    #     interior_pixels = region[mask]
        
    #     # Convert mask to boolean properly
    #     mask_bool = mask.astype(bool)
    #     analysis_region[~mask_bool] = np.mean(interior_pixels)
        
    #     # Apply CLAHE to normalize lighting
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     analysis_region_uint8 = np.clip(analysis_region, 0, 255).astype(np.uint8)
    #     normalized = clahe.apply(analysis_region_uint8).astype(float) / 255.0
        
    #     # Multi-scale texture analysis
    #     complexity_scores = []
        
    #     # High-frequency texture (fine patterns)
    #     laplacian = cv2.Laplacian(normalized, cv2.CV_64F, ksize=3)
    #     high_freq_var = np.var(laplacian[mask_bool])
    #     complexity_scores.append(high_freq_var * 10)
        
    #     # Edge density
    #     edges = cv2.Canny(analysis_region_uint8, 50, 150)
    #     edge_density = np.sum(edges[mask_bool] > 0) / np.sum(mask_bool)
    #     complexity_scores.append(edge_density * 2)
        
    #     # Local variance (busy-ness)
    #     local_var = self._calculate_local_variance(normalized, mask_bool, window=5)
    #     complexity_scores.append(np.mean(local_var[mask_bool]) * 5)
        
    #     # Directional patterns (stripes, lines)
    #     directional_score = self._detect_directional_patterns(normalized, mask_bool)
    #     complexity_scores.append(directional_score)
        
    #     # Combine scores
    #     final_score = np.mean(complexity_scores)
        
    #     # Determine pattern type
    #     if directional_score > 0.3:
    #         pattern_type = "directional (stripes/lines)"
    #     elif high_freq_var > 0.05:
    #         pattern_type = "high-frequency (busy)"
    #     elif edge_density > 0.3:
    #         pattern_type = "complex (many edges)"
    #     else:
    #         pattern_type = "textured"
        
    #     return final_score, pattern_type

    # main analysis function
    def _analyze_region_pattern(self, region, mask, image_rgb=None, region_rgb=None):
        """
        Enhanced pattern complexity analysis within a masked region
        """
        if region.size == 0 or not mask.any():
            return 0.0, "none"
        
        # Existing preprocessing...
        analysis_region = region.copy().astype(float)
        interior_pixels = region[mask]
        mask_bool = mask.astype(bool)
        analysis_region[~mask_bool] = np.mean(interior_pixels)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        analysis_region_uint8 = np.clip(analysis_region, 0, 255).astype(np.uint8)
        normalized = clahe.apply(analysis_region_uint8).astype(float) / 255.0
        
        # Multi-scale texture analysis
        complexity_scores = []
        
        # 1. High-frequency texture (existing)
        laplacian = cv2.Laplacian(normalized, cv2.CV_64F, ksize=3)
        high_freq_var = np.var(laplacian[mask_bool])
        complexity_scores.append(high_freq_var * 10)
        
        # 2. Edge density (existing)
        edges = cv2.Canny(analysis_region_uint8, 50, 150)
        edge_density = np.sum(edges[mask_bool] > 0) / np.sum(mask_bool)
        complexity_scores.append(edge_density * 2)
        
        # 3. Local variance (existing)
        local_var = self._calculate_local_variance(normalized, mask_bool, window=5)
        complexity_scores.append(np.mean(local_var[mask_bool]) * 5)
        
        # 4. Directional patterns (IMPROVED)
        directional_score = self._detect_directional_patterns(normalized, mask_bool)
        complexity_scores.append(directional_score)
        
        # 5. NEW: Frequency domain analysis
        freq_score = self._analyze_frequency_domain(normalized, mask_bool)
        complexity_scores.append(freq_score * 1.5)  # Weight for importance
        
        # 6. NEW: Repeating pattern detection
        repeat_score = self._detect_repeating_patterns(normalized, mask_bool)
        complexity_scores.append(repeat_score * 1.2)
        
        # 7. NEW: Chromatic complexity (if RGB provided)
        if region_rgb is not None:
            chroma_score = self._analyze_chromatic_complexity(region_rgb, mask_bool)
            complexity_scores.append(chroma_score * 1.3)  # Weight for color patterns
        
        # Combine scores with weights
        final_score = np.mean(complexity_scores)
        
        # Enhanced pattern type classification
        if repeat_score > 0.4:
            pattern_type = "repeating (wallpaper/tiled)"
        elif chroma_score > 0.4 if region_rgb is not None else False:
            pattern_type = "colorful/floral"
        elif directional_score > 0.3:
            pattern_type = "directional (stripes/geometric)"
        elif freq_score > 0.3:
            pattern_type = "high-frequency (busy)"
        elif edge_density > 0.3:
            pattern_type = "complex (many edges)"
        else:
            pattern_type = "textured"
        
        return final_score, pattern_type