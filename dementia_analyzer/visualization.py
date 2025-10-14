"""
Visualization Module
Generates visual outputs and reports for dementia-friendly assessment.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os


class VisualizationModule:
    """Handles all visualization and report generation"""
    
    def __init__(self):
        pass
    
    def save_results(self, image, contrast_risk, pattern_risk, overall_risk, 
                    depth_map, detections, contrast_details, image_path, output_dir='output'):
        """
        Generate and save all visualization outputs
        
        Args:
            image: RGB image
            contrast_risk: Contrast risk heatmap
            pattern_risk: Pattern risk heatmap
            overall_risk: Combined risk heatmap
            depth_map: Depth estimation map
            detections: List of detected objects
            contrast_details: List of contrast issues
            image_path: Path to original image
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.splitext(os.path.basename(image_path))[0]
        h, w = image.shape[:2]
        
        # Calculate statistics
        stats = self._calculate_statistics(contrast_risk, pattern_risk, overall_risk, depth_map)
        
        print(f"\n  Risk Statistics:")
        print(f"    Contrast risk - Mean: {stats['contrast_mean']:.3f}, Max: {stats['contrast_max']:.3f}, High-risk: {stats['contrast_high']:.1f}%")
        print(f"    Pattern risk  - Mean: {stats['pattern_mean']:.3f}, Max: {stats['pattern_max']:.3f}, High-risk: {stats['pattern_high']:.1f}%")
        print(f"    Overall risk  - Mean: {stats['overall_mean']:.3f}, Max: {stats['overall_max']:.3f}, High-risk: {stats['overall_high']:.1f}%")
        
        # Create main analysis figure
        self._create_main_analysis(image, contrast_risk, pattern_risk, overall_risk,
                                   depth_map, detections, contrast_details, stats,
                                   filename, output_dir)
        
        # Create clean overlay
        self._create_overlay(image, overall_risk, stats, filename, output_dir)
    
    def _calculate_statistics(self, contrast_risk, pattern_risk, overall_risk, depth_map):
        """Calculate statistics for risk maps"""
        stats = {
            'contrast_mean': np.mean(contrast_risk),
            'contrast_max': np.max(contrast_risk),
            'contrast_high': np.sum(contrast_risk > 0.7) / contrast_risk.size * 100,
            'pattern_mean': np.mean(pattern_risk),
            'pattern_max': np.max(pattern_risk),
            'pattern_high': np.sum(pattern_risk > 0.7) / pattern_risk.size * 100,
            'overall_mean': np.mean(overall_risk),
            'overall_max': np.max(overall_risk),
            'overall_high': np.sum(overall_risk > 0.7) / overall_risk.size * 100,
            'depth_range': np.max(depth_map) - np.min(depth_map)
        }
        return stats
    
    def _create_main_analysis(self, image, contrast_risk, pattern_risk, overall_risk,
                             depth_map, detections, contrast_details, stats, filename, output_dir):
        """Create main analysis figure with all views"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dementia-Friendly Assessment: {filename}', fontsize=16, fontweight='bold')
        
        # Original image with detections and SAM masks
        img_with_annotations = self._annotate_image(image, detections, contrast_details)
        
        contrast_issues_count = len([d for d in contrast_details if d['type'] == 'object_background_contrast'])
        sam_count = len([d for d in detections if d.get('mask') is not None])
        
        axes[0, 0].imshow(img_with_annotations)
        title_str = f'Detected Objects ({len(detections)} items)\n'
        if sam_count > 0:
            title_str += f'{sam_count} with SAM masks | '
        title_str += f'{contrast_issues_count} with contrast issues'
        axes[0, 0].set_title(title_str, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Depth map
        im1 = axes[0, 1].imshow(depth_map, cmap='plasma')
        depth_title = f'Depth Estimation\n(range: {stats["depth_range"]:.3f})'
        axes[0, 1].set_title(depth_title, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Contrast risk heatmap
        im2 = axes[0, 2].imshow(contrast_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Contrast Risk\nMean: {stats["contrast_mean"]:.3f} | High: {stats["contrast_high"]:.1f}%',
                            fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Pattern complexity heatmap
        im3 = axes[1, 0].imshow(pattern_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1, 0].set_title(f'Pattern Complexity Risk\nMean: {stats["pattern_mean"]:.3f} | High: {stats["pattern_high"]:.1f}%',
                            fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Overall risk heatmap
        im4 = axes[1, 1].imshow(overall_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Overall Risk Assessment\nMean: {stats["overall_mean"]:.3f} | High: {stats["overall_high"]:.1f}%',
                            fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        # Enhanced overlay
        overlay = image.copy().astype(np.float32) / 255.0
        heatmap_colored = plt.cm.RdYlGn_r(overall_risk)[:, :, :3]
        overlay = overlay * 0.5 + heatmap_colored * 0.5
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Risk Overlay on Image\n(50% original + 50% heatmap)', fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{filename}_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - Saved analysis to: {output_path}")
        plt.close()
    
    def _annotate_image(self, image, detections, contrast_details):
        """Annotate image with detections and contrast issues"""
        img_annotated = image.copy()
        
        # Build lookup for objects with contrast issues
        contrast_issues_lookup = {}
        for detail in contrast_details:
            if detail['type'] == 'object_background_contrast':
                obj_key = detail['location']
                contrast_issues_lookup[obj_key] = detail
        
        # Create mask overlay
        mask_overlay = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Check if this object has contrast issues
            obj_key = f"{det['class']} at ({x1}, {y1})"
            has_contrast_issue = obj_key in contrast_issues_lookup
            
            if has_contrast_issue:
                issue = contrast_issues_lookup[obj_key]
                risk = issue['risk']
                # Color code by severity
                if risk > 0.7:
                    color = (255, 0, 0)
                    mask_color = (255, 0, 0, 100)
                    thickness = 3
                elif risk > 0.5:
                    color = (255, 165, 0)
                    mask_color = (255, 165, 0, 80)
                    thickness = 3
                else:
                    color = (255, 255, 0)
                    mask_color = (255, 255, 0, 60)
                    thickness = 2
                
                label = f"{det['class']}: ⚠ C:{issue['contrast_ratio']:.2f}"
            else:
                color = (0, 255, 0)
                mask_color = (0, 255, 0, 40)
                thickness = 2
                label = f"{det['class']}: ✓"
            
            # Draw SAM mask if available
            if det.get('mask') is not None:
                mask_overlay[det['mask']] = mask_color
                
                # Draw mask boundary
                mask_uint8 = det['mask'].astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_annotated, contours, -1, color, thickness)
            else:
                # Fallback: draw bounding box
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(img_annotated,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            cv2.putText(img_annotated, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend mask overlay
        alpha_mask = mask_overlay[:, :, 3:4] / 255.0
        img_annotated = (img_annotated * (1 - alpha_mask) + mask_overlay[:, :, :3] * alpha_mask).astype(np.uint8)
        
        return img_annotated
    
    def _create_overlay(self, image, overall_risk, stats, filename, output_dir):
        """Create clean overlay with legend"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        overlay = image.copy().astype(np.float32) / 255.0
        heatmap_colored = plt.cm.RdYlGn_r(overall_risk)[:, :, :3]
        overlay = overlay * 0.5 + heatmap_colored * 0.5
        
        ax.imshow(overlay)
        ax.axis('off')
        
        # Title with statistics
        title = f'Dementia-Friendly Risk Assessment: {filename}\n'
        title += f'Overall Risk Score: {stats["overall_mean"]:.3f} | High-Risk Areas: {stats["overall_high"]:.1f}%'
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')
        
        # Legend
        legend_elements = [
            Patch(facecolor='darkred', label=f'High Risk (>0.7) - {stats["overall_high"]:.1f}% of image'),
            Patch(facecolor='orange', label='Medium Risk (0.4-0.7)'),
            Patch(facecolor='yellow', label='Low Risk (0.2-0.4)'),
            Patch(facecolor='lightgreen', label='Minimal Risk (<0.2)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{filename}_overlay.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  - Saved overlay to: {output_path}")
        plt.close()
    
    def generate_detailed_report(self, contrast_risk, pattern_risk, contrast_details, 
                                pattern_details, detections):
        """
        Generate detailed text report with recommendations
        
        Args:
            contrast_risk: Contrast risk map
            pattern_risk: Pattern risk map
            contrast_details: List of contrast issues
            pattern_details: List of pattern issues
            detections: List of detected objects
            
        Returns:
            Formatted text report
        """
        report = "\n" + "="*70 + "\n"
        report += "DEMENTIA-FRIENDLY ASSESSMENT REPORT\n"
        report += "="*70 + "\n\n"
        
        # Overall statistics
        avg_risk = (np.mean(contrast_risk) + np.mean(pattern_risk)) / 2
        high_contrast_risk = np.sum(contrast_risk > 0.7) / contrast_risk.size * 100
        high_pattern_risk = np.sum(pattern_risk > 0.7) / pattern_risk.size * 100
        
        report += f"DETECTED OBJECTS: {len(detections)}\n"
        report += f"CONTRAST ISSUES: {len(contrast_details)}\n"
        report += f"PATTERN ISSUES: {len(pattern_details)}\n\n"
        
        # Detailed findings
        if contrast_details:
            report += "CONTRAST ISSUES:\n"
            report += "-" * 70 + "\n"
            for i, detail in enumerate(contrast_details, 1):
                report += f"{i}. {detail['location']}\n"
                report += f"   Contrast Ratio: {detail['contrast_ratio']:.2f} (target: >1.5)\n"
                report += f"   Risk Level: {'HIGH' if detail['risk'] > 0.7 else 'MEDIUM' if detail['risk'] > 0.4 else 'LOW'}\n"
                report += f"   Recommendation: Increase contrast between object and background\n\n"
        
        if pattern_details:
            report += "PATTERN COMPLEXITY ISSUES:\n"
            report += "-" * 70 + "\n"
            for i, detail in enumerate(pattern_details, 1):
                report += f"{i}. {detail['location']}\n"
                report += f"   Complexity Score: {detail['complexity_score']:.3f} (threshold: <0.18)\n"
                report += f"   Pattern Type: {detail['pattern_type']}\n"
                report += f"   Risk Level: {'HIGH' if detail['risk'] > 0.7 else 'MEDIUM' if detail['risk'] > 0.4 else 'LOW'}\n"
                report += f"   Recommendation: Replace with simpler, solid-colored alternative\n\n"
        
        # Overall assessment
        report += "OVERALL ASSESSMENT:\n"
        report += "-" * 70 + "\n"
        
        if avg_risk > 0.6:
            report += "🔴 HIGH RISK: Immediate action recommended\n\n"
            report += "PRIORITY ACTIONS:\n"
            if high_contrast_risk > 15:
                report += "1. Address floor-wall contrast immediately\n"
            if high_pattern_risk > 15:
                report += "2. Replace items with complex patterns\n"
        elif avg_risk > 0.4:
            report += "⚠ MEDIUM RISK: Several areas need attention\n\n"
            report += "RECOMMENDED ACTIONS:\n"
            if contrast_details:
                report += "1. Improve contrast in flagged areas\n"
            if pattern_details:
                report += "2. Consider simplifying patterned surfaces\n"
        elif avg_risk > 0.2:
            report += "⚡ LOW-MEDIUM RISK: Some minor improvements recommended\n\n"
        else:
            report += "✓ LOW RISK: Environment appears generally dementia-friendly\n\n"
        
        report += f"\nAverage risk score: {avg_risk:.3f} (0=safe, 1=unsafe)\n"
        report += "="*70 + "\n"
        
        return report
