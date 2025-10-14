"""
Segmentation Module
Handles YOLO object detection and SAM (Segment Anything Model) for precise segmentation.
"""

import numpy as np
import torch

# YOLO and SAM imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SegmentationModule:
    """Handles object detection and segmentation using YOLO and SAM"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.yolo = None
        self.sam_predictor = None
        self.yolo_loaded = False
        self.sam_loaded = False
        
        self._load_yolo()
        self._load_sam()
    
    def _load_yolo(self):
        """Load YOLO model for object detection"""
        if not YOLO_AVAILABLE:
            print("✗ YOLO not available - install with: pip install ultralytics")
            return
        
        print("Loading YOLO model...")
        try:
            self.yolo = YOLO('yolov8n.pt')  # Nano version for speed
            self.yolo_loaded = True
            print("✓ YOLO loaded successfully")
        except Exception as e:
            print(f"✗ YOLO failed to load: {e}")
            self.yolo = None
    
    def _load_sam(self):
        """Load SAM (Segment Anything Model) for precise segmentation"""
        if not SAM_AVAILABLE:
            print("✗ SAM not available - install with: pip install segment-anything")
            return
        
        print("Loading SAM (Segment Anything Model)...")
        try:
            # Download SAM checkpoint if needed
            sam_checkpoint = "sam_vit_b_01ec64.pth"
            
            # Try to load from local, if not available torch.hub will download
            try:
                sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            except:
                print("  Downloading SAM checkpoint (first time only)...")
                import urllib.request
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                urllib.request.urlretrieve(url, sam_checkpoint)
                sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            self.sam_loaded = True
            print("✓ SAM loaded successfully")
        except Exception as e:
            print(f"✗ SAM failed to load: {e}")
            print("  Falling back to bounding box analysis")
            self.sam_predictor = None
    
    def detect_objects(self, image):
        """
        Detect objects using YOLO and get precise masks with SAM
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            List of detection dictionaries containing class, confidence, bbox, and mask
        """
        if self.yolo is None:
            return []
        
        results = self.yolo(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                detection = {
                    'class': result.names[int(boxes.cls[i])],
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xyxy[i].cpu().numpy().astype(int),  # [x1, y1, x2, y2]
                    'class_id': int(boxes.cls[i]),
                    'mask': None  # Will be filled by SAM
                }
                detections.append(detection)
        
        # Get precise masks using SAM
        if self.sam_loaded and len(detections) > 0:
            print(f"    Getting precise masks with SAM for {len(detections)} objects...")
            detections = self.get_sam_masks(image, detections)
        
        return detections
    
    def detect_objects_hybrid(self, image):
        """
        Hybrid approach: Use YOLO for classification + SAM everything for comprehensive coverage.
        
        This combines the best of both:
        - YOLO identifies what objects are (chair, couch, etc.)
        - SAM finds ALL segments including cushions, pillows, decorations
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            List of detections with both YOLO-classified and SAM-only segments
        """
        # First, get YOLO detections
        yolo_detections = self.detect_objects(image) if self.yolo_loaded else []
        
        # Then, get SAM everything detections
        sam_detections = self.detect_everything_with_sam(image)
        
        if not sam_detections:
            return yolo_detections
        
        # Merge: Keep YOLO detections for their classifications, 
        # add SAM detections that don't significantly overlap with YOLO
        
        h, w = image.shape[:2]
        yolo_coverage = np.zeros((h, w), dtype=bool)
        
        # Mark areas covered by YOLO
        for det in yolo_detections:
            if det.get('mask') is not None:
                yolo_coverage |= det['mask']
            else:
                x1, y1, x2, y2 = det['bbox']
                yolo_coverage[y1:y2, x1:x2] = True
        
        # Add SAM segments that are mostly uncovered by YOLO
        additional_detections = []
        for sam_det in sam_detections:
            sam_mask = sam_det['mask']
            
            # Calculate overlap with YOLO coverage
            overlap = np.sum(sam_mask & yolo_coverage)
            sam_area = np.sum(sam_mask)
            
            overlap_ratio = overlap / sam_area if sam_area > 0 else 1.0
            
            # If less than 30% overlap with YOLO, it's a new object (likely cushion, pillow, etc.)
            if overlap_ratio < 0.3:
                additional_detections.append(sam_det)
        
        print(f"    Hybrid detection: {len(yolo_detections)} YOLO + {len(additional_detections)} additional SAM segments")
        
        return yolo_detections + additional_detections


    def detect_everything_with_sam(self, image):
        """
        Use SAM to segment EVERYTHING in the image automatically (no YOLO needed).
        This catches objects YOLO misses like cushions, pillows, wall art, etc.
        Slower but much more comprehensive.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            List of detection dictionaries with masks
        """
        if self.sam_predictor is None:
            print("    SAM not available - falling back to YOLO only")
            return []
        
        from segment_anything import SamAutomaticMaskGenerator
        
        print("    Running SAM automatic mask generation (this may take a moment)...")
        
        # Create automatic mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_predictor.model,
            points_per_side=24,              # Density of point grid (32x32 = 1024 points)
            pred_iou_thresh=0.86,            # Quality threshold
            stability_score_thresh=0.92,     # Stability threshold
            crop_n_layers=1,                 # Crop layers for better detection
            crop_n_points_downscale_factor=2,
            min_mask_region_area=150,        # Minimum 200 pixels (filters tiny segments)
        )
        
        # Generate masks for everything
        masks = mask_generator.generate(image)
        
        print(f"    SAM detected {len(masks)} segments")
        
        # Convert to detection format
        detections = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            
            # Get bounding box from mask
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue
                
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Calculate mask area
            area = mask_data['area']
            
            detection = {
                'class': 'sam_segment',  # Generic name since we don't have YOLO classification
                'confidence': mask_data['stability_score'],
                'bbox': np.array([x_min, y_min, x_max, y_max]),
                'class_id': -1,
                'mask': mask,
                'area': area
            }
            detections.append(detection)
        
        # Sort by area (largest first) for better visualization
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        return detections

    def get_sam_masks(self, image, detections):
        """
        Use SAM to get precise pixel-level masks for detected objects
        
        Args:
            image: RGB image as numpy array
            detections: List of detection dictionaries from YOLO
            
        Returns:
            Updated detections with masks
        """
        if self.sam_predictor is None:
            return detections
        
        h, w = image.shape[:2]
        
        # Set image for SAM
        self.sam_predictor.set_image(image)
        
        # Get mask for each detection using its bounding box as prompt
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Convert bbox to format SAM expects: [x1, y1, x2, y2]
            box_prompt = np.array([x1, y1, x2, y2])
            
            try:
                # Predict mask using bounding box prompt
                masks, scores, _ = self.sam_predictor.predict(
                    box=box_prompt,
                    multimask_output=False  # Single mask per object
                )
                
                # Take the first (and only) mask
                mask = masks[0]
                
                # Store mask in detection
                detection['mask'] = mask
                detection['mask_score'] = float(scores[0])
            except Exception as e:
                print(f"    Warning: SAM failed for {detection['class']}: {e}")
                detection['mask'] = None
        
        return detections
