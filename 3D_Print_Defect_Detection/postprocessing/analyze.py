"""
InSight3D - Defect Analysis & Quality Decision Engine
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Workflow:
1. Extract contours from binary defect segmentation mask using OpenCV.
2. Compute defect statistics:
   - Defect count
   - Area of each defect (pixels)
   - Total defect area & largest defect
   - Defect percentage relative to component slice area
   - Bounding box coordinates [x, y, w, h]
   - Centroid coordinates (cx, cy)
3. Evaluate component quality using configurable rules engine:
   - Defect Percentage < 1.0%: PASS
   - 1.0% <= Defect Percentage <= 3.0%: WARNING
   - Defect Percentage > 3.0%: FAIL
4. Return structured JSON / dict summary report.
"""

import json
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


class QualityDecisionEngine:
    """
    Configurable Rule Engine for Component Quality Assessment.
    """

    def __init__(
        self,
        pass_threshold: float = 1.0,
        warning_threshold: float = 3.0,
        max_single_defect_area: Optional[float] = None
    ):
        """
        Args:
            pass_threshold (float): Defect percentage upper limit for PASS status.
            warning_threshold (float): Defect percentage upper limit for WARNING status.
            max_single_defect_area (float, optional): Maximum allowed area for a single defect before forcing FAIL.
        """
        self.pass_threshold = pass_threshold
        self.warning_threshold = warning_threshold
        self.max_single_defect_area = max_single_defect_area

    def evaluate(self, defect_percentage: float, largest_defect_area: float) -> str:
        """
        Evaluate Quality Status based on configured threshold rules.
        
        Returns:
            str: 'PASS', 'WARNING', or 'FAIL'
        """
        if self.max_single_defect_area is not None and largest_defect_area > self.max_single_defect_area:
            return "FAIL"

        if defect_percentage < self.pass_threshold:
            return "PASS"
        elif defect_percentage <= self.warning_threshold:
            return "WARNING"
        else:
            return "FAIL"


def analyze_defects(
    binary_mask: np.ndarray,
    min_defect_size: int = 3,
    component_mask: Optional[np.ndarray] = None,
    decision_engine: Optional[QualityDecisionEngine] = None
) -> Dict[str, Any]:
    """
    Analyze binary defect mask and compute statistical metrics.
    
    Args:
        binary_mask (np.ndarray): Binary segmentation mask (256, 256) where 1/255 represents defect pixels.
        min_defect_size (int): Minimum pixel area to consider as a valid defect (filters out single noise pixels).
        component_mask (np.ndarray, optional): Binary mask of total component region for precise area % calculation.
        decision_engine (QualityDecisionEngine, optional): Custom quality decision rules instance.
        
    Returns:
        Dict[str, Any]: Structured dictionary with defect metrics and quality status.
    """
    if decision_engine is None:
        decision_engine = QualityDecisionEngine(pass_threshold=1.0, warning_threshold=3.0)

    # Standardize mask to uint8 {0, 255}
    if binary_mask.dtype != np.uint8:
        mask_uint8 = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1.0 else binary_mask.astype(np.uint8)
    else:
        mask_uint8 = binary_mask.copy()

    # Find external contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects_list: List[Dict[str, Any]] = []
    total_defect_area = 0.0
    largest_defect_area = 0.0

    defect_id = 1
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_defect_size:
            continue  # Filter noise

        total_defect_area += area
        if area > largest_defect_area:
            largest_defect_area = area

        # Compute Bounding Box
        x, y, w, h = cv2.boundingRect(cnt)

        # Compute Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        defects_list.append({
            "id": defect_id,
            "area": area,
            "centroid": [cx, cy],
            "bbox": [x, y, w, h]
        })
        defect_id += 1

    # Calculate component slice area for percentage calculation
    if component_mask is not None:
        total_component_area = float(np.sum(component_mask > 0))
    else:
        # Default to entire slice dimension area (e.g. 256 * 256 = 65536)
        total_component_area = float(binary_mask.shape[0] * binary_mask.shape[1])

    defect_percentage = round((total_defect_area / total_component_area) * 100.0, 2)
    defect_count = len(defects_list)

    # Evaluate Quality Status
    status = decision_engine.evaluate(defect_percentage, largest_defect_area)

    analysis_result = {
        "defect_count": defect_count,
        "total_defect_area": round(total_defect_area, 2),
        "largest_defect": round(largest_defect_area, 2),
        "defect_percentage": defect_percentage,
        "status": status,
        "defects": defects_list
    }

    return analysis_result


def export_analysis_to_json(analysis_result: Dict[str, Any], output_filepath: str) -> str:
    """Save analysis dictionary to JSON file."""
    with open(output_filepath, "w") as f:
        json.dump(analysis_result, f, indent=4)
    return output_filepath


if __name__ == "__main__":
    print("InSight3D Defect Analysis Module Test")
    
    # Create synthetic binary mask with 3 defects
    synthetic_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(synthetic_mask, (60, 60), 12, 1, -1)   # Defect 1: Area ~452
    cv2.circle(synthetic_mask, (150, 150), 18, 1, -1) # Defect 2: Area ~1017 (Largest)
    cv2.rectangle(synthetic_mask, (180, 40), (200, 60), 1, -1) # Defect 3: Area 400

    results = analyze_defects(synthetic_mask)
    print("\nAnalysis JSON Output:")
    print(json.dumps(results, indent=4))
