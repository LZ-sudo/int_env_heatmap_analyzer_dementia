"""
pattern_analysis.py - Contrast-focused CV complexity (NO TRAINING)
Specifically targets high color contrast patterns while ignoring:
- Subtle natural variations (wood grain, marble veining)
- Lighting differences (shadows, highlights)
- Solid colors
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
from scipy.fft import fft2, fftshift


class ContrastBasedComplexityClassifier:
    """
    Pattern complexity classifier focused on color contrast
    
    HIGH COMPLEXITY:
    - High color contrast patterns (blue stripes on yellow)
    - Obvious decorative patterns (red with white/gold)
    - Intentional bold designs
    
    LOW COMPLEXITY:
    - Subtle variations (dark brown on light brown wood)
    - Low contrast patterns (beige on white marble)
    - Solid colors
    - Lighting variations (shadows, highlights)
    """
    
    def __init__(self):
        print("\n" + "="*70)
        print("CONTRAST-BASED PATTERN COMPLEXITY CLASSIFIER")
        print("Focuses on color contrast, ignores lighting")
        print("="*70 + "\n")
    
    def remove_lighting_effects(self, image):
        """
        Normalize lighting to avoid confusing shadows with patterns
        Uses illumination-invariant representation
        
        Returns:
            Lighting-normalized image
        """
        # Convert to LAB color space (perceptually uniform)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Method 1: Normalize L channel (lightness)
        # Remove global illumination variations
        l_float = l.astype(float)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This normalizes local lighting while preserving actual color patterns
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_normalized = clahe.apply(l)
        
        # Reconstruct LAB with normalized lightness
        lab_normalized = cv2.merge([l_normalized, a, b])
        rgb_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
        
        return rgb_normalized, lab_normalized, (l_normalized, a, b)
    
    def calculate_color_contrast(self, image, lab_image, lab_channels):
        """
        Calculate color contrast (NOT lightness contrast)
        High contrast = visually complex pattern
        
        Args:
            image: RGB image
            lab_image: LAB color space image
            lab_channels: (L, a, b) channels
            
        Returns:
            Color contrast score (0-1)
        """
        l, a, b = lab_channels
        h, w = image.shape[:2]
        
        # Use a* and b* channels (color information, NOT lightness)
        # These are lighting-invariant color channels
        
        # 1. Calculate local color variance
        # High variance in a/b channels = strong color patterns
        
        # Smooth slightly to avoid noise
        a_smooth = cv2.GaussianBlur(a.astype(float), (5, 5), 1)
        b_smooth = cv2.GaussianBlur(b.astype(float), (5, 5), 1)
        
        # Calculate local standard deviation (color variation)
        kernel_size = 15
        
        # Local std for a* channel (green-red)
        a_mean = cv2.blur(a_smooth, (kernel_size, kernel_size))
        a_sq_mean = cv2.blur(a_smooth**2, (kernel_size, kernel_size))
        a_std = np.sqrt(np.maximum(a_sq_mean - a_mean**2, 0))
        
        # Local std for b* channel (blue-yellow)
        b_mean = cv2.blur(b_smooth, (kernel_size, kernel_size))
        b_sq_mean = cv2.blur(b_smooth**2, (kernel_size, kernel_size))
        b_std = np.sqrt(np.maximum(b_sq_mean - b_mean**2, 0))
        
        # Combined color variation (Euclidean distance in a*b* space)
        color_variation = np.sqrt(a_std**2 + b_std**2)
        
        # Average color variation across image
        mean_color_variation = np.mean(color_variation)
        
        # Normalize to 0-1 range
        # Typical values: 0-5 for low contrast, 10-30 for high contrast
        color_contrast_score = min(mean_color_variation / 20, 1.0)
        
        return color_contrast_score, color_variation
    
    def calculate_pattern_contrast_ratio(self, image, lab_channels):
        """
        Calculate maximum local color contrast ratio
        E.g., blue stripes on yellow = high ratio
        
        Returns:
            Contrast ratio score (0-1)
        """
        l, a, b = lab_channels
        
        # For each local region, find color contrast
        # Use sliding window to find max color difference
        
        kernel_size = 20
        
        # Find local min/max in a* and b* channels
        a_float = a.astype(float)
        b_float = b.astype(float)
        
        # Morphological operations to find local extrema
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        a_max = cv2.dilate(a_float, kernel)
        a_min = cv2.erode(a_float, kernel)
        a_range = a_max - a_min
        
        b_max = cv2.dilate(b_float, kernel)
        b_min = cv2.erode(b_float, kernel)
        b_range = b_max - b_min
        
        # Combined local color range
        local_color_range = np.sqrt(a_range**2 + b_range**2)
        
        # High percentile (90th) to capture strongest contrasts
        strong_contrast = np.percentile(local_color_range, 90)
        
        # Normalize (typical range: 0-50)
        contrast_ratio_score = min(strong_contrast / 40, 1.0)
        
        return contrast_ratio_score
    
    def check_repeating_pattern(self, image):
        """
        Check if there's a repeating pattern (stripes, dots, grid)
        Decorative patterns are often regular/repeating
        Natural variations are NOT regular
        
        Returns:
            Pattern repetition score (0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT to detect periodicity
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Log transform for better visualization
        magnitude_spectrum = np.log(magnitude_spectrum + 1)
        
        # Remove DC component (center)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10] = 0
        
        # Look for strong peaks indicating regular patterns
        threshold = np.percentile(magnitude_spectrum, 99.5)
        strong_peaks = np.sum(magnitude_spectrum > threshold)
        
        # More peaks = more regular pattern
        # Typical: 0-5 peaks for natural, 10+ for decorative patterns
        repetition_score = min(strong_peaks / 15, 1.0)
        
        return repetition_score
    
    def check_color_diversity(self, lab_channels):
        """
        Check diversity of colors (NOT lightness)
        Multi-colored patterns = complex
        Monochrome/similar colors = simple
        
        Returns:
            Color diversity score (0-1)
        """
        l, a, b = lab_channels
        
        # Create 2D histogram in a*b* color space
        # This captures color diversity independent of lightness
        
        # Quantize a and b channels
        a_quantized = ((a.astype(float) + 128) / 256 * 32).astype(int)
        b_quantized = ((b.astype(float) + 128) / 256 * 32).astype(int)
        
        # Clip to valid range
        a_quantized = np.clip(a_quantized, 0, 31)
        b_quantized = np.clip(b_quantized, 0, 31)
        
        # Create 2D histogram
        hist_2d = np.zeros((32, 32))
        for i in range(a_quantized.shape[0]):
            for j in range(a_quantized.shape[1]):
                hist_2d[a_quantized[i, j], b_quantized[i, j]] += 1
        
        # Normalize
        hist_2d = hist_2d / (hist_2d.sum() + 1e-7)
        
        # Calculate entropy (high = diverse colors)
        color_entropy = entropy(hist_2d.flatten() + 1e-7)
        
        # Normalize (max entropy for 32x32 bins ≈ 10)
        diversity_score = min(color_entropy / 8, 1.0)
        
        return diversity_score
    
    def is_solid_color(self, lab_channels):
        """
        Quick check if surface is solid color
        
        Returns:
            (is_solid, confidence)
        """
        l, a, b = lab_channels
        
        # Check standard deviation in all channels
        std_l = np.std(l)
        std_a = np.std(a)
        std_b = np.std(b)
        
        # Very low std in color channels = solid color
        # L can vary due to lighting, but a and b should be stable
        is_solid = (std_a < 3) and (std_b < 3)
        
        # Confidence based on how uniform
        uniformity = 1.0 - min((std_a + std_b) / 10, 1.0)
        
        return is_solid, uniformity
    
    def calculate_complexity_score(self, image):
        """
        Main complexity calculation
        
        Returns:
            (complexity_score, subscores_dict)
        """
        h, w = image.shape[:2]
        
        # Step 1: Remove lighting effects
        rgb_normalized, lab_normalized, lab_channels = self.remove_lighting_effects(image)
        
        # Step 2: Check if solid color (early exit)
        is_solid, solid_conf = self.is_solid_color(lab_channels)
        if is_solid and solid_conf > 0.8:
            # Definitely solid color
            return 0.0, {
                'solid_color': True,
                'color_contrast': 0.0,
                'contrast_ratio': 0.0,
                'repetition': 0.0,
                'diversity': 0.0
            }
        
        # Step 3: Calculate color contrast (MAIN METRIC)
        color_contrast, color_var_map = self.calculate_color_contrast(
            rgb_normalized, lab_normalized, lab_channels
        )
        
        # Step 4: Calculate contrast ratio (secondary metric)
        contrast_ratio = self.calculate_pattern_contrast_ratio(image, lab_channels)
        
        # Step 5: Check for repeating patterns
        repetition_score = self.check_repeating_pattern(rgb_normalized)
        
        # Step 6: Check color diversity
        diversity_score = self.check_color_diversity(lab_channels)
        
        # Weighted combination
        # Color contrast is MOST important for your use case
        weights = {
            'color_contrast': 0.35,      # Yellow with blue stripes
            'contrast_ratio': 0.20,      # Red with white/gold
            'repetition': 0.20,          # Regular patterns
            'diversity': 0.25            # Multi-colored
        }
        
        scores = {
            'solid_color': False,
            'color_contrast': color_contrast,
            'contrast_ratio': contrast_ratio,
            'repetition': repetition_score,
            'diversity': diversity_score
        }
        
        complexity_score = sum(scores[k] * weights[k] for k in weights if k in scores)
        
        return complexity_score, scores
    
    def classify(self, image):
        """
        Classify as simple or complex
        
        Returns:
            (category, confidence, is_complex, complexity_score, subscores)
        """
        complexity_score, subscores = self.calculate_complexity_score(image)
        
        # Threshold-based classification
        # Adjusted for color contrast focus
        is_complex = complexity_score > 0.4
        category = 'complex' if is_complex else 'simple'
        
        # Confidence based on distance from threshold
        distance_from_threshold = abs(complexity_score - 0.4)
        confidence = min(distance_from_threshold * 2 + 0.6, 1.0)
        
        return category, confidence, is_complex, complexity_score, subscores


class PatternAnalysisModule:
    """
    Pattern analysis module using contrast-based complexity
    NO TRAINING REQUIRED
    """
    
    def __init__(self, complexity_threshold=0.40):
        """
        Args:
            complexity_threshold: Complexity score threshold (0-1)
                                 Recommended: 0.35-0.45
                                 Lower = flag more patterns
                                 Higher = only flag obvious patterns
        """
        self.COMPLEXITY_THRESHOLD = complexity_threshold
        self.classifier = ContrastBasedComplexityClassifier()
        
        print(f"Complexity threshold: {complexity_threshold:.2f}")
        print("  Tuned for high color contrast patterns")
        print("  Ignores lighting variations and subtle textures")
        print("="*70 + "\n")
    
    def analyze_pattern_complexity_per_object(self, image, detections):
        """
        Analyze each detected object for pattern complexity
        
        Args:
            image: RGB image as numpy array
            detections: List of detected objects from segmentation
            
        Returns:
            (pattern_risk_map, pattern_details)
        """
        h, w = image.shape[:2]
        pattern_risk = np.zeros((h, w), dtype=np.float32)
        pattern_details = []
        
        print(f"    Analyzing {len(detections)} segments for color contrast patterns...")
        
        analyzed = 0
        flagged = 0
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Extract region
            if det.get('mask') is not None:
                obj_mask = det['mask']
                mask_coords = np.argwhere(obj_mask)
                if len(mask_coords) == 0:
                    continue
                
                y_min, x_min = mask_coords.min(axis=0)
                y_max, x_max = mask_coords.max(axis=0)
                region_rgb = image[y_min:y_max+1, x_min:x_max+1]
            else:
                obj_mask = np.zeros((h, w), dtype=bool)
                obj_mask[y1:y2, x1:x2] = True
                region_rgb = image[y1:y2, x1:x2]
            
            # Skip if too small
            if region_rgb.shape[0] < 32 or region_rgb.shape[1] < 32:
                continue
            
            analyzed += 1
            
            # Calculate complexity
            category, confidence, is_complex, complexity_score, subscores = \
                self.classifier.classify(region_rgb)
            
            # Flag if complex
            if complexity_score > self.COMPLEXITY_THRESHOLD:
                flagged += 1
                
                risk_value = complexity_score
                
                pattern_risk[obj_mask] = np.maximum(
                    pattern_risk[obj_mask],
                    risk_value
                )
                
                pattern_details.append({
                    'type': 'pattern_complexity',
                    'location': f"{det.get('class', 'object')} at ({x1}, {y1})",
                    'object': det.get('class', 'sam_segment'),
                    'category': 'visually_complex',
                    'pattern_type': 'high_contrast_pattern',  # ADD THIS LINE
                    'complexity_score': complexity_score,
                    'color_contrast': subscores.get('color_contrast', 0),
                    'contrast_ratio': subscores.get('contrast_ratio', 0),
                    'confidence': confidence,
                    'risk': risk_value
                })
        
        print(f"    Analyzed: {analyzed}, Flagged: {flagged}")
        
        if flagged > 0:
            pattern_risk = gaussian_filter(pattern_risk, sigma=3)
        
        return pattern_risk, pattern_details