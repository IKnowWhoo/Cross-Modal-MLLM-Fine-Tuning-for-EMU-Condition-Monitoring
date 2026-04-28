import os
import re
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from PIL import Image
from mmseg.apis import inference_segmentor, init_segmentor
from backbone import beit 
import math
from scipy import ndimage
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import asyncio
import psutil
from dataclasses import dataclass
import traceback
torch.backends.cudnn.benchmark = False



import os
import re
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import json
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
from PIL import Image
from mmseg.apis import inference_segmentor, init_segmentor
from backbone import beit
import math
from scipy import ndimage
import time
import logging
import sys
from datetime import datetime
import traceback
import shutil


torch.backends.cudnn.benchmark = False


class LogManager:
    """Unified logging manager: all GPUs log into a single file"""

    def __init__(self, log_dir: str, log_level: int = logging.INFO):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{self.timestamp}.log")

        self.logger = logging.getLogger("multi_gpu_processor")
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for errors
        console_handler = logging.StreamHandler(sys.stderr)
        console_formatter = logging.Formatter("%(asctime)s - ERROR - %(message)s")
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.log_file = log_file

    def log_info(self, message: str, gpu_id: int = None):
        if gpu_id is not None:
            self.logger.info(f"[GPU-{gpu_id}] {message}\n")
        else:
            self.logger.info(f'{message}\n')

    def log_error(self, message: str, gpu_id: int = None, exc_info: bool = False):
        if gpu_id is not None:
            self.logger.error(f"[GPU-{gpu_id}] {message}\n", exc_info=exc_info)
        else:
            self.logger.error(f'{message}\n', exc_info=exc_info)

class MultiGPULongImageProcessor:
    """
    Processor optimized for handling a single sequence on a dedicated GPU.
    Includes pixel count validation for connected components.
    """

    def __init__(self, config_path: str, checkpoint_path: str, gpu_id: int = 0, load_model: bool = True,
                 log_manager: LogManager = None):
        """
        Initialize the processor for a specific GPU.

        Args:
            config_path: Path to model config.
            checkpoint_path: Path to model checkpoint.
            gpu_id: The ID of the GPU to use for this instance.
            load_model: If False, skips loading the model to create a lightweight instance for helper tasks.
            log_manager: LogManager instance for logging
        """
        self.gpu_id = gpu_id
        self.log_manager = log_manager

        if load_model:
            self.device = f'cuda:{self.gpu_id}'
            self._log_info(f"Initializing processor...")

            # Initialize the model on the assigned GPU
            try:
                self.model = init_segmentor(config_path, checkpoint_path, device=self.device)
                self._log_info(f"Model loaded successfully on {self.device}")
            except Exception as e:
                self._log_error(f"Failed to load model on {self.device}: {e}", exc_info=True)
                raise
        else:
            self._log_info("Initializing helper instance (no model loaded).")
            self.model = None
            self.device = 'cpu'

        self.class_names = ['background', 'gearbox', 'axle', 'motor', 'wheel', 'air_duct']
        self.class_id_map = {name: idx for idx, name in enumerate(self.class_names)}

        # Rule-based target definitions
        self.neighboring_patterns_ch567 = [
            ['wheel', 'gearbox', 'motor', 'axle'],
            ['wheel', 'gearbox', 'axle'],
            ['wheel', 'gearbox'],
            ['wheel', 'axle', 'motor']
        ]
        self.non_neighboring_patterns_ch567 = [['wheel', 'axle']]
        self.non_neighboring_patterns_crh5 = [['wheel', 'gearbox']]

        # Default patterns for unspecified channels
        self.default_neighboring_patterns = [
            ['wheel', 'gearbox', 'motor', 'axle'],
            ['wheel', 'gearbox', 'axle'],
            ['wheel', 'gearbox'],
            ['wheel', 'axle', 'motor'],
            ['wheel', 'axle']
        ]
        
        self.palette = [
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128]
        ]

        self.pixel_ranges = {
            'air_duct': (1, 10), 'axle': (2, 20),
            'gearbox': (3, 30), 'motor': (4, 40),
            'wheel': (5, 50), 'background': (0, float('inf'))
        }

        # Chunking and processing parameters
        self.max_chunk_height = 512
        self.max_chunk_width = 512
        self.overlap = 512
        self.min_chunk_merge_ratio = 0.00
        self.max_connection_distance = 1
        self.boundary_padding = 1

        try:
            with open('/home/suma/bin/try_detectron2/multipleGPUs/station_rotation.json', 'r') as f:
                self.station_dict = json.load(f)
        except Exception as e:
            self._log_error(f"Warning: Could not load station rotation data: {e}")
            self.station_dict = {}

    def _log_info(self, message: str):
        """Log info message"""
        if self.log_manager:
            self.log_manager.log_info(message, self.gpu_id)
        else:
            print(f"[GPU-{self.gpu_id}] {message}")

    def _log_error(self, message: str, exc_info: bool = False):
        """Log error message"""
        if self.log_manager:
            self.log_manager.log_error(message, self.gpu_id, exc_info)
        else:
            print(f"[GPU-{self.gpu_id}] ERROR: {message}", file=sys.stderr)
            if exc_info:
                traceback.print_exc()

    def find_non_neighboring_groups_by_height(self, mask: np.ndarray, target_classes: List[str], height_tolerance: int = 30) -> List[Dict]:
        """
        Finds target groups based on vertical alignment (height overlap) rather than direct connectivity.
        This is used for components that are expected to be near each other vertically but not touching.

        Args:
            mask (np.ndarray): The segmentation mask.
            target_classes (List[str]): A list of class names that constitute the group (e.g., ['wheel', 'axle']).
            height_tolerance (int): The number of pixels to extend the vertical search range.

        Returns:
            List[Dict]: A list of found groups, each a dictionary with components and metadata.
        """
        self._log_info(f"Searching for non-neighboring groups: {target_classes}")
        all_components = defaultdict(list)
        for class_name in target_classes:
            class_id = self.class_id_map[class_name]
            components = self.find_connected_components_for_class(mask, class_id)
            for comp_mask, bbox in components:
                all_components[class_name].append({
                    'class_name': class_name,
                    'mask': comp_mask,
                    'bbox': bbox,
                    'height_range': (bbox[0], bbox[1])
                })

        # The primary component must be 'wheel' for this logic
        if 'wheel' not in all_components or not all_components['wheel']:
            return []

        other_class_names = [cls for cls in target_classes if cls != 'wheel']
        found_groups = []

        # Use merged wheel groups as the anchor for finding other parts
        wheel_groups = self.merge_wheels_by_tolerance(all_components['wheel'], tolerance=height_tolerance)

        for wheel_group in wheel_groups:
            wheel_min_row = min(w['height_range'][0] for w in wheel_group)
            wheel_max_row = max(w['height_range'][1] for w in wheel_group)

            group_components = list(wheel_group) # Start with the wheels
            
            # Find compatible components from other classes
            for class_name in other_class_names:
                for comp in all_components[class_name]:
                    comp_min_row, comp_max_row = comp['height_range']
                    # Check for vertical overlap within tolerance
                    if max(wheel_min_row, comp_min_row) <= min(wheel_max_row, comp_max_row) + height_tolerance:
                         group_components.append(comp)

            # Validate if the found group is satisfactory
            found_class_set = {c['class_name'] for c in group_components}
            if set(target_classes).issubset(found_class_set):
                all_rows, all_cols = [], []
                for comp in group_components:
                    rows, cols = np.where(comp['mask'])
                    all_rows.extend(rows)
                    all_cols.extend(cols)
                
                if not all_rows or not all_cols: continue

                min_row, max_row = min(all_rows), max(all_rows)
                min_col, max_col = min(all_cols), max(all_cols)

                found_groups.append({
                    'target_pattern': target_classes,
                    'found_classes': [c['class_name'] for c in group_components],
                    'components': group_components,
                    'bbox': (min_row, max_row, min_col, max_col),
                    'num_components': len(group_components),
                })
        
        return found_groups

    def find_single_component_groups(self, mask: np.ndarray, class_name: str) -> List[Dict]:
        """
        Finds all individual connected components of a single class and formats them as 'groups'.
        
        Args:
            mask (np.ndarray): The segmentation mask.
            class_name (str): The name of the class to find (e.g., 'wheel').

        Returns:
            List[Dict]: A list of groups, where each group contains just one component.
        """
        self._log_info(f"Searching for single-component groups: {class_name}")
        groups = []
        class_id = self.class_id_map[class_name]
        components = self.find_connected_components_for_class(mask, class_id)

        for comp_mask, bbox in components:
            groups.append({
                'target_pattern': [class_name],
                'found_classes': [class_name],
                'components': [{'class_name': class_name, 'mask': comp_mask, 'bbox': bbox}],
                'bbox': bbox,
                'num_components': 1
            })
        
        return groups
    
    def find_connected_components_for_class(self, mask: np.ndarray, class_id: int) -> List[
        Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Find connected components for a specific class and return component masks with bounding boxes.

        Args:
            mask: Segmentation mask
            class_id: Class ID to find components for

        Returns:
            List of tuples (component_mask, (min_row, max_row, min_col, max_col))
        """
        class_mask = (mask == class_id).astype(np.uint8)
        num_components, labels = cv2.connectedComponents(class_mask, connectivity=8)

        components = []
        for component_id in range(1, num_components):  # Skip background component 0
            component_mask = (labels == component_id)

            # Find bounding box
            rows, cols = np.where(component_mask)
            if len(rows) > 0:
                min_row, max_row = int(rows.min()), int(rows.max())
                min_col, max_col = int(cols.min()), int(cols.max())
                components.append((component_mask, (min_row, max_row, min_col, max_col)))

        return components

    def get_component_centroid(self, component_mask: np.ndarray) -> Tuple[int, int]:
        """Get centroid of a connected component."""
        rows, cols = np.where(component_mask)
        if len(rows) > 0:
            return int(rows.mean()), int(cols.mean())
        return 0, 0

    def is_within_distance(self, component1: Dict, component2: Dict, max_distance: int) -> bool:
        """
        Check if the minimum distance between any two points of two component masks is within max_distance.
        This uses a distance transform for efficient and accurate calculation.

        Args:
            component1 (Dict): Dictionary for the first component, must contain 'mask' and 'bbox'.
            component2 (Dict): Dictionary for the second component, must contain 'mask' and 'bbox'.
            max_distance (int): The maximum pixel distance to be considered "neighboring".

        Returns:
            bool: True if the components are within the specified distance, False otherwise.
        """
        # 1. Quick optimization: check if bounding boxes are close enough to potentially connect.
        min_r1, max_r1, min_c1, max_c1 = component1['bbox']
        min_r2, max_r2, min_c2, max_c2 = component2['bbox']

        # Pad the first bounding box by max_distance. If it doesn't overlap with the second,
        # the components cannot be within max_distance.
        padded_min_r1, padded_max_r1 = min_r1 - max_distance, max_r1 + max_distance
        padded_min_c1, padded_max_c1 = min_c1 - max_distance, max_c1 + max_distance

        # Check for intersection between the padded bbox1 and the original bbox2
        if (padded_max_r1 < min_r2 or padded_min_r1 > max_r2 or
                padded_max_c1 < min_c2 or padded_min_c1 > max_c2):
            return False

        # 2. Create a combined Region of Interest (ROI) to minimize computation
        combined_min_r = min(min_r1, min_r2)
        combined_max_r = max(max_r1, max_r2)
        combined_min_c = min(min_c1, min_c2)
        combined_max_c = max(max_c1, max_c2)

        # Crop the full masks to this combined ROI
        mask1_roi = component1['mask'][combined_min_r:combined_max_r + 1, combined_min_c:combined_max_c + 1]
        mask2_roi = component2['mask'][combined_min_r:combined_max_r + 1, combined_min_c:combined_max_c + 1]

        # Ensure masks are not empty within the ROI
        if not np.any(mask1_roi) or not np.any(mask2_roi):
            return False

        # 3. Use distance transform for precise distance calculation.
        # ndimage.distance_transform_edt computes the distance to the nearest zero pixel.
        # By inverting mask1_roi, we get the distance from every pixel to the nearest True pixel in mask1_roi.
        dist_transform = ndimage.distance_transform_edt(np.logical_not(mask1_roi))

        # 4. Find the minimum distance on the transform map where mask2 exists.
        # These values represent the distance from each pixel of mask2 to the closest pixel of mask1.
        distances_on_mask2 = dist_transform[mask2_roi]

        if distances_on_mask2.size == 0:
            return False

        min_dist = np.min(distances_on_mask2)

        return min_dist <= max_distance

    def merge_wheels_uncrh5(self, components: List[Dict], max_distance: int) -> List[List[Dict]]:
        """
        Merge wheel components that are close together (within max_distance).
        Returns a list of wheel groups (each group is a list of components).
        """
        wheels = [c for c in components if c['class_name'] == 'wheel']
        if not wheels:
            return []

        wheels_sorted = sorted(wheels, key=lambda x: x['bbox'][0])  # sort by min_row
        merged_groups = []

        current_group = [wheels_sorted[0]]
        current_min, current_max = wheels_sorted[0]['bbox'][0], wheels_sorted[0]['bbox'][1]

        for wheel in wheels_sorted[1:]:
            w_min, w_max = wheel['bbox'][0], wheel['bbox'][1]
            if w_min <= current_max + max_distance:  # overlaps/touching within distance
                current_group.append(wheel)
                current_min = min(current_min, w_min)
                current_max = max(current_max, w_max)
            else:
                merged_groups.append(current_group)
                current_group = [wheel]
                current_min, current_max = w_min, w_max

        merged_groups.append(current_group)
        return merged_groups

    def find_target_groups_connected(self, mask: np.ndarray, target_classes: List[str], target_groups: List[List[str]],
                                     max_distance: int = 30) -> List[Dict]:
        """
        Find connected target groups using BFS, but merge nearby wheels first.
        """
        # Step 1: Extract components
        all_components = {}
        for class_name in target_classes:
            class_id = self.class_id_map[class_name]
            components = self.find_connected_components_for_class(mask, class_id)
            all_components[class_name] = []
            for comp_mask, bbox in components:
                centroid = self.get_component_centroid(comp_mask)
                all_components[class_name].append({
                    'class_name': class_name,
                    'mask': comp_mask,
                    'bbox': bbox,
                    'centroid': centroid,
                    'used': False
                })

        # Flatten all components
        component_list = []
        for comps in all_components.values():
            component_list.extend(comps)

        if not component_list:
            return []

        groups = []

        # Step 2: Merge wheels first
        wheel_groups = self.merge_wheels_uncrh5(component_list, max_distance)

        # Step 3: BFS per wheel group
        visited_components = set()
        for wheel_group in wheel_groups:
            queue = deque(wheel_group)
            group_components = []
            while queue:
                comp = queue.popleft()
                comp_id = id(comp)
                if comp_id in visited_components:
                    continue
                visited_components.add(comp_id)
                group_components.append(comp)
                # check neighbors
                for neighbor in component_list:
                    neighbor_id = id(neighbor)
                    if neighbor_id in visited_components:
                        continue
                    if self.is_within_distance(comp, neighbor, max_distance):
                        queue.append(neighbor)

            # Check if group matches any target pattern
            group_classes = [c['class_name'] for c in group_components]
            group_class_set = set(group_classes)
            for target_group in target_groups:  # self.target_groups
                if set(target_group).issubset(group_class_set):
                    # calculate bounding box
                    all_rows, all_cols = [], []
                    for comp in group_components:
                        rows, cols = np.where(comp['mask'])
                        all_rows.extend(rows)
                        all_cols.extend(cols)
                    if all_rows and all_cols:
                        min_row, max_row = min(all_rows), max(all_rows)
                        min_col, max_col = min(all_cols), max(all_cols)
                        groups.append({
                            'target_pattern': target_group,
                            'found_classes': group_classes,
                            'components': group_components,
                            'bbox': (min_row, max_row, min_col, max_col),
                            'num_components': len(group_components)
                        })
                    break  # one pattern matched

        # Step 4: Remove duplicates
        groups = self.remove_duplicate_groups(groups)
        return groups

    def merge_wheels_by_tolerance(self, wheels: List[Dict], tolerance: int = 30) -> List[List[Dict]]:
        """
        Merge wheels whose height ranges overlap within tolerance.
        Returns a list of wheel groups (each group is a list of wheel components).
        """
        if not wheels:
            return []

        wheels_sorted = sorted(wheels, key=lambda x: x['height_range'][0])  # sort by min_row
        merged_groups = []

        current_group = [wheels_sorted[0]]
        current_min, current_max = wheels_sorted[0]['height_range']

        for wheel in wheels_sorted[1:]:
            w_min, w_max = wheel['height_range']
            if w_min <= current_max + tolerance:  # overlaps/touching within tolerance
                current_group.append(wheel)
                current_min = min(current_min, w_min)
                current_max = max(current_max, w_max)
            else:
                merged_groups.append(current_group)
                current_group = [wheel]
                current_min, current_max = w_min, w_max

        merged_groups.append(current_group)
        return merged_groups

    def remove_duplicate_cr5_groups(self, groups: List[Dict], overlap_threshold: float = 0.1) -> List[Dict]:
        """
        Remove duplicate CR5 groups that have significant overlap.
        Prefer groups with more components (wheel+axle+gearbox > wheel+axle/wheel+gearbox).
        """
        if not groups:
            return groups

        # Sort groups by priority: 3-component groups first, then 2-component groups
        priority_order = {'wheel_axle_gearbox': 3, 'wheel_axle': 2, 'wheel_gearbox': 2}
        groups.sort(key=lambda x: (priority_order.get(x.get('cr5_group_type'), 1),
                                   x.get('num_components', 0)), reverse=True)

        filtered_groups = []

        for group in groups:
            bbox1 = group['bbox']
            comp_ids1 = set(group.get('component_ids', [c.get('id') for c in group.get('components', [])]))
            is_duplicate = False

            for existing_group in filtered_groups:
                # if they share identical component ids -> duplicate
                comp_ids2 = set(
                    existing_group.get('component_ids', [c.get('id') for c in existing_group.get('components', [])]))
                if comp_ids1 == comp_ids2:
                    is_duplicate = True
                    break

                bbox2 = existing_group['bbox']

                # Calculate overlap (rows x cols)
                overlap_h = max(0, min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0]))
                overlap_w = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2]))
                overlap_area = overlap_h * overlap_w

                area1 = max(1, (bbox1[1] - bbox1[0])) * max(1, (bbox1[3] - bbox1[2]))
                area2 = max(1, (bbox2[1] - bbox2[0])) * max(1, (bbox2[3] - bbox2[2]))

                union_area = area1 + area2 - overlap_area
                overlap_ratio = overlap_area / union_area if union_area > 0 else 0

                if overlap_ratio > overlap_threshold:
                    # consider duplicate (we've sorted by priority so existing_group is equal or higher priority)
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_groups.append(group)

        return filtered_groups

    def cut_target_groups(self, image: np.ndarray, mask: np.ndarray, prefix: str, seq_id: int,
                          output_dir: str, parsed_data: Dict) -> List[Dict]:
        """
        Finds and cuts target groups from the image and mask based on channel and train type rules.

        Args:
            image: Original image.
            mask: Segmentation mask.
            prefix: Original image name prefix.
            seq_id: Split image ID.
            output_dir: Output directory.
            parsed_data: Dictionary containing metadata like 'channel', 'train_group', 'row'.

        Returns:
            List of dictionaries with cut group information.
        """
        channel_str = str(parsed_data.get('channel', ''))
        row_str = parsed_data.get('row', '') # For 'L' channels, e.g., 'L4'
        train_group = parsed_data.get('train_group', '').upper()
        
        self._log_info(f"Applying rules for Train: {train_group}, Channel: {channel_str}, Row: {row_str}")

        all_found_groups = []

        # Rule 4: No groups for channels 1, 2
        if channel_str in ['1', '2']:
            self._log_info("Rule matched: Channels 1 & 2. No target groups will be searched.")
            return []

        # Rule 3: Single wheel for channels 3, 4, 8, 9, L3, L7
        elif channel_str in ['3', '4', '8', '9', 'L3', 'L7']:
            self._log_info("Rule matched: Channels 3,4,8,9,L3,L7. Searching for single wheels with 3x pixel range.")
            all_found_groups = self.find_single_component_groups(mask, 'wheel')
            # original_wheel_range = self.pixel_ranges['wheel']
            # try:
            #     # Temporarily increase the minimum pixel requirement for wheels
            #     new_min_pixels = original_wheel_range[0] * 3
            #     self.pixel_ranges['wheel'] = (new_min_pixels, original_wheel_range[1])
            #     self._log_info(f"Temporarily changed wheel pixel range to: {self.pixel_ranges['wheel']}")
            #
            #     all_found_groups = self.find_single_component_groups(mask, 'wheel')
            # finally:
            #     # Always restore the original pixel range
            #     self.pixel_ranges['wheel'] = original_wheel_range
            #     self._log_info(f"Restored wheel pixel range to: {self.pixel_ranges['wheel']}")

        # Rule 1: Channels 5, 6, 7, L4, L5, L6
        elif channel_str in ['5', '6', '7', 'L4', 'L5', 'L6']:
            self._log_info("Rule matched: Channels 5,6,7,L4-L6.")
            # Find neighboring groups
            for pattern in self.neighboring_patterns_ch567:
                groups = self.find_target_groups_connected(mask, pattern, self.neighboring_patterns_ch567, self.max_connection_distance)
                all_found_groups.extend(groups)
            # Find non-neighboring groups
            for pattern in self.non_neighboring_patterns_ch567:
                groups = self.find_non_neighboring_groups_by_height(mask, pattern)
                all_found_groups.extend(groups)
            all_found_groups.extend(self.find_single_component_groups(mask, 'air_duct'))
        
        # Fallback to default rules for any other channels not explicitly handled
        else:
            self._log_info(f"No specific rule for Channel {channel_str}/Row {row_str}. Applying default neighboring rules.")
            for pattern in self.default_neighboring_patterns:
                groups = self.find_target_groups_connected(mask, pattern, self.default_neighboring_patterns, self.max_connection_distance)
                all_found_groups.extend(groups)
        
        # Rule 2: Special case for CRH5, applied in addition to channel rules
        if 'CRH5' in train_group:
            self._log_info("Special Rule Matched: CRH5. Searching for non-neighboring ['wheel', 'gearbox'].")
            crh5_groups = self.find_non_neighboring_groups_by_height(mask, self.non_neighboring_patterns_crh5[0])
            all_found_groups.extend(crh5_groups)

        # Remove duplicates from all collected groups
        target_groups = self.remove_duplicate_groups(all_found_groups)
        self._log_info(f"Found {len(target_groups)} unique target groups in {prefix} after applying all rules.")

        if not target_groups:
            return []

        # --- The rest of the function (saving cuts and creating metadata) remains the same ---
        cuts_dir = os.path.join(output_dir, 'target_cuts')
        os.makedirs(cuts_dir, exist_ok=True)
        cut_metadata = []

        for group_idx, group in enumerate(target_groups):
            try:
                min_row, max_row, min_col, max_col = group['bbox']
                padded_min_row = max(0, min_row - self.boundary_padding)
                padded_max_row = min(mask.shape[0], max_row + self.boundary_padding + 1)
                cut_min_col, cut_max_col = 0, mask.shape[1]

                cut_image = image[padded_min_row:padded_max_row, cut_min_col:cut_max_col]
                cut_mask = mask[padded_min_row:padded_max_row, cut_min_col:cut_max_col]
                
                group_name = '_'.join(sorted(list(set(g['class_name'] for g in group['components']))))
                cut_filename_base = f"{prefix}_{seq_id}_group{group_idx}_{group_name}"

                # Save cut image (Commented out as per user request)
                cut_image_path = os.path.join(cuts_dir, f"{cut_filename_base}_image.png")
                cut_image_bgr = cut_image[:, :, ::-1]  # Convert RGB to BGR for OpenCV
                cv2.imwrite(cut_image_path, cut_image_bgr)
                # cut_image_path = "Image saving is disabled."

                # Save cut mask (colored) (Commented out as per user request)
                # Create colored mask for the cut
                cut_colored_mask = self.create_colored_mask(cut_mask)
                cut_mask_path = os.path.join(cuts_dir, f"{cut_filename_base}_mask.png")
                cv2.imwrite(cut_mask_path, cut_colored_mask)
                # cut_mask_path = "Mask saving is disabled."

                cut_info = {
                    'prefix': prefix,
                    'group_index': group_idx,
                    'target_pattern': group['target_pattern'],
                    'found_classes': group['found_classes'],
                    'num_components': group['num_components'],
                    'original_bbox': {
                        'min_row': int(min_row), 'max_row': int(max_row),
                        'min_col': int(min_col), 'max_col': int(max_col)
                    },
                    'cut_bbox': {
                        'min_row': int(padded_min_row), 'max_row': int(padded_max_row),
                        'min_col': int(cut_min_col), 'max_col': int(cut_max_col)
                    },
                    'cut_dimensions': {
                        'height': int(padded_max_row - padded_min_row),
                        'width': int(cut_max_col - cut_min_col)
                    },
                    'files': {'cut_image': cut_image_path, 'cut_mask': cut_mask_path}
                }
                cut_metadata.append(cut_info)
            except Exception as e:
                self._log_error(f"Error processing target group {group_idx}: {e}", exc_info=True)

        return cut_metadata

    def identify_and_copy_no_target_images(self, filtered_paths: List[str], image_heights: List[int],
                                            cut_metadata: List[Dict], trim_start_row: int, output_dir: str):
        """
        Identifies original images without any target groups and copies them to a 'no_target' folder.

        Args:
            filtered_paths: List of file paths for the original images in the sequence.
            image_heights: List of the heights of each image as they appear in the concatenated strip.
            cut_metadata: The metadata from cut_target_groups, containing bboxes of found targets.
            trim_start_row: The vertical offset (y-coordinate) where the trimmed image begins.
            output_dir: The main output directory.
        """
        # Step 1: Get the vertical ranges of all found target groups in the full concatenated image space.
        target_vertical_ranges = []
        for item in cut_metadata:
            bbox = item['original_bbox']
            # Adjust for the trimming that happened to the concatenated image
            start_y = bbox['min_row'] + trim_start_row
            end_y = bbox['max_row'] + trim_start_row
            target_vertical_ranges.append((start_y, end_y))

        # Step 2: Calculate the vertical ranges of each original image in the concatenated space.
        original_image_ranges = []
        current_y = 0
        for height in image_heights:
            original_image_ranges.append((current_y, current_y + height))
            current_y += height

        # Step 3: Find which original images do NOT overlap with any target group.
        no_target_paths = []
        for i, (img_path, (img_start_y, img_end_y)) in enumerate(zip(filtered_paths, original_image_ranges)):
            has_overlap = False
            for (target_start_y, target_end_y) in target_vertical_ranges:
                # Check for any vertical overlap
                if max(img_start_y, target_start_y) < min(img_end_y, target_end_y):
                    has_overlap = True
                    break  # This image has a target, move to the next image
            
            if not has_overlap:
                no_target_paths.append(img_path)

        # Step 4: Copy the identified images.
        if not no_target_paths:
            self._log_info(f"No 'no-target' images found for this sequence.")
            return

        no_target_dir = os.path.join(output_dir, 'no_target')
        os.makedirs(no_target_dir, exist_ok=True)
        
        copied_count = 0
        for src_path in no_target_paths:
            try:
                shutil.copy(src_path, no_target_dir)
                copied_count += 1
            except Exception as e:
                self._log_error(f"Failed to copy no-target image {src_path} to {no_target_dir}: {e}")
        
        if copied_count > 0:
            self._log_info(f"Copied {copied_count} no-target images to {no_target_dir} for this sequence.")


    def remove_duplicate_groups(self, groups: List[Dict], overlap_threshold: float = 0.1) -> List[Dict]:
        """
        Remove duplicate groups that have significant bbox overlap.

        Args:
            groups: List of found groups
            overlap_threshold: Minimum overlap ratio to consider as duplicate

        Returns:
            Filtered list of groups
        """
        if not groups:
            return groups

        # Sort groups by number of components (prefer groups with more components)
        groups.sort(key=lambda x: x['num_components'], reverse=True)

        filtered_groups = []

        for group in groups:
            bbox1 = group['bbox']
            is_duplicate = False

            for existing_group in filtered_groups:
                bbox2 = existing_group['bbox']

                # Calculate overlap
                overlap_area = max(0, min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0])) * \
                               max(0, min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2]))

                area1 = (bbox1[1] - bbox1[0]) * (bbox1[3] - bbox1[2])
                area2 = (bbox2[1] - bbox2[0]) * (bbox2[3] - bbox2[2])

                union_area = area1 + area2 - overlap_area
                overlap_ratio = overlap_area / union_area if union_area > 0 else 0

                if overlap_ratio > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_groups.append(group)

        return filtered_groups

    def save_cut_visualization(self, cut_image: np.ndarray, cut_mask: np.ndarray, output_path: str, alpha: float = 0.1):
        """
        Create and save visualization for cut target group.

        Args:
            cut_image: Cut image region
            cut_mask: Cut mask region
            output_path: Path to save visualization
            alpha: Transparency for overlay
        """
        try:
            # Convert to BGR for OpenCV
            image_bgr = cut_image[:, :, ::-1] if len(cut_image.shape) == 3 else cv2.cvtColor(cut_image,
                                                                                             cv2.COLOR_GRAY2BGR)
            colored_mask = self.create_colored_mask(cut_mask)

            # Create overlay
            overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)

            # Create side-by-side visualization
            visualization = np.hstack((image_bgr, colored_mask, overlay))

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, font_color, thickness = 0.8, (255, 255, 255), 2

            cv2.putText(visualization, 'Original', (10, 30), font, font_scale, font_color, thickness)
            cv2.putText(visualization, 'Mask', (cut_image.shape[1] + 10, 30), font, font_scale, font_color, thickness)
            cv2.putText(visualization, 'Overlay', (cut_image.shape[1] * 2 + 10, 30), font, font_scale, font_color,
                        thickness)

            cv2.imwrite(output_path, visualization)

        except Exception as e:
            self._log_error(f"Failed to create cut visualization: {e}")
            # Fallback: save just the colored mask
            try:
                colored_mask = self.create_colored_mask(cut_mask)
                cv2.imwrite(output_path, colored_mask)
            except Exception as e2:
                self._log_error(f"Failed to save even simplified cut visualization: {e2}")

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a single chunk on the instance's assigned GPU.
        Includes fallback strategies for memory errors.
        """
        try:
            h, w = chunk.shape[:2]
            # Safety check
            if h > self.max_chunk_height or w > self.max_chunk_width:
                chunk, _ = self.resize_to_safe_dimensions(chunk)
                h, w = chunk.shape[:2]

            chunk_bgr = chunk[:, :, ::-1] if len(chunk.shape) == 3 else chunk

            # Strategy 1: Inference with autocast for efficiency
            try:
                with torch.cuda.amp.autocast():
                    result = inference_segmentor(self.model, chunk_bgr)
                return result[0].astype(np.uint8)
            except RuntimeError as e:
                self._log_info(f"Autocast inference failed: {e}. Trying without.")

            # Strategy 2: Standard inference
            try:
                result = inference_segmentor(self.model, chunk_bgr)
                return result[0].astype(np.uint8)
            except RuntimeError as e:
                self._log_info(f"Standard inference failed: {e}. Trying with resize.")

            # Strategy 3: Resize, infer, and scale back
            try:
                safe_h, safe_w = 512, 512
                resized_chunk = cv2.resize(chunk_bgr, (safe_w, safe_h), interpolation=cv2.INTER_LINEAR)
                result = inference_segmentor(self.model, resized_chunk)
                small_mask = result[0].astype(np.uint8)
                return cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                self._log_error(f"Resized inference also failed: {e}. Returning blank mask.")

            # Final fallback
            return np.zeros((h, w), dtype=np.uint8)

        except Exception as e:
            self._log_error(f"Critical error processing chunk: {e}", exc_info=True)
            h, w = chunk.shape[:2] if len(chunk.shape) >= 2 else (512, 512)
            return np.zeros((h, w), dtype=np.uint8)

    def _process_chunks(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a list of chunks sequentially on the instance's dedicated GPU.
        """
        if not chunks:
            return []

        start_time = time.time()
        masks = [self._process_chunk(chunk) for chunk in chunks]
        total_time = time.time() - start_time

        self._log_info(
            f"Processed {len(chunks)} chunks in {total_time:.2f}s ({total_time / len(chunks):.3f}s per chunk)")
        return masks

    def convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable Python types recursively."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, defaultdict):
            return {key: self.convert_to_json_serializable(value) for key, value in dict(obj).items()}
        else:
            return obj

    def validate_connected_components(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Validate connected components based on pixel count ranges and remove invalid ones.

        Args:
            mask: Input segmentation mask

        Returns:
            Tuple of (filtered_mask, validation_stats)
        """
        filtered_mask = mask.copy()
        validation_stats = {
            'removed_components': defaultdict(int),
            'kept_components': defaultdict(int),
            'total_pixels_removed': defaultdict(int),
            'total_pixels_kept': defaultdict(int)
        }

        # Process each class (skip background)
        for class_id in range(1, len(self.class_names)):
            class_name = self.class_names[class_id]
            min_pixels, max_pixels = self.pixel_ranges[class_name]

            # Extract binary mask for current class
            class_mask = (mask == class_id).astype(np.uint8)

            if np.sum(class_mask) == 0:
                continue

            # Find connected components
            num_components, labels = cv2.connectedComponents(class_mask, connectivity=8)

            # Process each component (skip component 0 which is background)
            for component_id in range(1, num_components):
                component_mask = (labels == component_id)
                component_pixels = int(np.sum(component_mask))

                # Check if component is within valid pixel range
                if min_pixels <= component_pixels <= max_pixels:
                    # Keep this component
                    validation_stats['kept_components'][class_name] += 1
                    validation_stats['total_pixels_kept'][class_name] += component_pixels
                else:
                    # Remove this component (set to background)
                    filtered_mask[component_mask] = 0
                    validation_stats['removed_components'][class_name] += 1
                    validation_stats['total_pixels_removed'][class_name] += component_pixels

        validation_stats = self.convert_to_json_serializable(validation_stats)
        return filtered_mask, validation_stats

    def is_blank_image(self, image_np: np.ndarray, threshold: float = 10.0) -> bool:
        """Determines if an image is blank (i.e., mostly black)."""
        return np.mean(image_np) < threshold

    def resize_to_safe_dimensions(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it exceeds safe dimensions while maintaining aspect ratio.
        Returns resized image and scale factor.
        """
        h, w = image.shape[:2]

        # Calculate scale factor to fit within limits
        scale_h = self.max_chunk_height / h if h > self.max_chunk_height else 1.0
        scale_w = self.max_chunk_width / w if w > self.max_chunk_width else 1.0
        scale_factor = min(scale_h, scale_w)

        if scale_factor < 1.0:
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            # Ensure dimensions are even (helps with some models)
            new_h = new_h - (new_h % 2)
            new_w = new_w - (new_w % 2)

            self._log_info(f"Resizing from {h}x{w} to {new_h}x{new_w} (scale: {scale_factor:.3f})")
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, scale_factor

        return image, 1.0

    def intelligent_concatenation(self, image_paths: List[str]) -> Tuple[
        Optional[np.ndarray], List[np.ndarray], int, List[str], float, List[int]]:
        """
        Improved concatenation with aspect-ratio aware size management.

        Returns:
            Tuple of (concatenated_image, chunks, chunk_height, valid_paths, scale_factor, image_heights)
        """
        # Load and prepare all images
        processed_images = []
        for path in image_paths:
            try:
                img_pil = Image.open(path)

                # Apply rotation logic if needed
                try:
                    parsed_data = self.parse_full_image_metadata(path)
                    station_name = parsed_data['station']
                    channel_int = parsed_data['channel']
                    needs_rotation = self.get_rotation_flag(str(channel_int), station_name, self.station_dict)
                    
                    if needs_rotation:
                        # Apply 90 degrees anti-clockwise rotation (equivalent to 90 degrees clockwise)
                        img_pil = img_pil.transpose(Image.ROTATE_90)
                except ValueError as e:
                    self._log_error(
                        f"Warning: Could not parse metadata for {os.path.basename(path)}: {e}. Skipping rotation.")

                img_np = np.array(img_pil.convert('RGB'))
                processed_images.append(img_np)

            except Exception as e:
                self._log_error(f"Warning: Could not load {path}: {e}. Skipping.")
                continue

        if not processed_images:
            return None, [], 0, [], 1.0, []

        # Filter blank images from start and end
        start_index = -1
        for i, img in enumerate(processed_images):
            if not self.is_blank_image(img):
                start_index = i
                break

        if start_index == -1:
            self._log_error("Warning: All images in the sequence are blank. Skipping.")
            return None, [], 0, [], 1.0, []

        end_index = -1
        for i in range(len(processed_images) - 1, -1, -1):
            if not self.is_blank_image(processed_images[i]):
                end_index = i
                break

        num_original = len(processed_images)
        valid_processed_images = processed_images[start_index: end_index + 1]
        valid_image_paths = image_paths[start_index: end_index + 1]

        if len(valid_processed_images) < num_original:
            self._log_info(
                f"Filtered out {num_original - len(valid_processed_images)} blank images from the start/end of the sequence.")

        if not valid_processed_images:
            return None, [], 0, [], 1.0, []

        # Ensure all images have the same width for vertical stacking
        target_width = valid_processed_images[0].shape[1]

        # Smart width management - resize if too wide
        if target_width > self.max_chunk_width:
            width_scale = self.max_chunk_width / target_width
            target_width = self.max_chunk_width
            self._log_info(f"Width too large, scaling down by factor {width_scale:.3f}")

            # Resize all images to the new target width
            for i, img in enumerate(valid_processed_images):
                new_height = int(img.shape[0] * width_scale)
                valid_processed_images[i] = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LINEAR)

        resized_images = []
        for i, img in enumerate(valid_processed_images):
            if img.shape[1] != target_width:
                self._log_error(f"Warning: Image {os.path.basename(valid_image_paths[i])} width mismatch. Resizing.")
                new_height = int(img.shape[0] * (target_width / img.shape[1]))
                img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
            resized_images.append(img)

        # Get the heights of each image before stacking
        image_heights = [img.shape[0] for img in resized_images]

        # Concatenate all valid images into one long strip
        full_concatenated = np.vstack(resized_images)
        total_height, width = full_concatenated.shape[:2]
        self._log_info(f"Full concatenated image size: {total_height}x{width}")

        # Global scaling for extremely tall images
        global_scale_factor = 1.0
        max_allowed_width = self.max_chunk_width
        if width > max_allowed_width:
            global_scale_factor = width / max_allowed_width
            new_height = int(total_height * global_scale_factor)
            new_width = int(width * global_scale_factor)

            self._log_info(f"Image too wide ({width}), scaling down to {new_height}x{new_width}")
            full_concatenated = cv2.resize(full_concatenated, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            total_height, width = new_height, new_width


        # Aspect-ratio aware chunk height calculation
        if width > 1536:
            chunk_height = min(self.max_chunk_height, max(1536, int(width * 1.0)))
        elif width > 1024:
            chunk_height = min(int(width * 1.0), max(1280, int(width * 1.0)))
        else:
            chunk_height = min(int(width * 1.0), max(768, int(width * 1.0)))

        # Ensure chunk height is reasonable and divisible by xx
        chunk_height = min(chunk_height, self.max_chunk_height)
        chunk_height = max(chunk_height, 512)  # Minimum chunk size
        chunk_height = (chunk_height // 512) * 512  # Make divisible by xx

        chunks = self._split_into_improved_chunks(full_concatenated, chunk_height)
        self._log_info(f"Created {len(chunks)} chunks for multi-GPU processing")

        return full_concatenated, chunks, chunk_height, valid_image_paths, global_scale_factor, image_heights

    def _split_into_improved_chunks(self, image: np.ndarray, chunk_height: int) -> List[np.ndarray]:
        """
        Improved chunking strategy that avoids creating oversized chunks.
        """
        total_height, width = image.shape[:2]
        chunks = []
        stride = chunk_height - self.overlap

        y_start = 0
        while y_start < total_height:
            y_end = min(y_start + chunk_height, total_height)
            chunk = image[y_start:y_end, :]

            # Intelligent merging logic
            remaining_height = total_height - y_start
            is_last_chunk = y_end >= total_height
            chunk_ratio = chunk.shape[0] / chunk_height

            if is_last_chunk and chunk_ratio < self.min_chunk_merge_ratio and len(chunks) > 0:
                # Only merge very small last chunks
                self._log_info(
                    f"Merging small last chunk ({chunk.shape[0]} px, ratio: {chunk_ratio:.2f}) with previous chunk")
                merged_height = chunks[-1].shape[0] + chunk.shape[0]

                # Ensure merged chunk doesn't exceed limits
                if merged_height <= self.max_chunk_height:
                    chunks[-1] = np.vstack([chunks[-1], chunk])
                else:
                    # If last chunk would be too large, keep as separate chunk
                    chunks.append(chunk)
            else:
                chunks.append(chunk)

            if y_end >= total_height:
                break
            y_start += stride

        # Final safety check - resize any oversized chunks
        for i, chunk in enumerate(chunks):
            if chunk.shape[0] > self.max_chunk_height or chunk.shape[1] > self.max_chunk_width:
                safe_chunk, _ = self.resize_to_safe_dimensions(chunk)
                chunks[i] = safe_chunk
                self._log_info(f"Resized chunk {i} to safe dimensions: {safe_chunk.shape[:2]}")

        return chunks

    def merge_chunk_masks(self, chunk_masks: List[np.ndarray], chunk_positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Merge overlapping chunk masks into final result using weighted averaging in overlap regions.
        """
        if not chunk_masks:
            return None

        max_y_end = max(pos[1] for pos in chunk_positions)
        width = chunk_masks[0].shape[1]

        final_mask = np.zeros((max_y_end, width), dtype=np.float32)
        weight_map = np.zeros((max_y_end, width), dtype=np.float32)

        for mask, (y_start, y_end) in zip(chunk_masks, chunk_positions):
            chunk_height = y_end - y_start
            actual_mask = mask[:chunk_height, :]

            chunk_weight = np.ones((chunk_height, width), dtype=np.float32)
            if self.overlap > 0:
                fade_region = min(self.overlap, chunk_height // 2)
                if y_start > 0:
                    for i in range(fade_region):
                        chunk_weight[i, :] *= (i + 1) / fade_region
                if y_end < max_y_end:
                    for i in range(fade_region):
                        chunk_weight[chunk_height - 1 - i, :] *= (fade_region - i) / fade_region

            final_mask[y_start:y_end, :] += actual_mask.astype(np.float32) * chunk_weight
            weight_map[y_start:y_end, :] += chunk_weight

        mask_nonzero = weight_map > 0
        final_mask[mask_nonzero] /= weight_map[mask_nonzero]

        return np.round(final_mask).astype(np.uint8)

    def create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(len(self.class_names)):
            mask_indices = (mask == class_id)
            colored_mask[mask_indices] = self.palette[class_id][::-1]

        return colored_mask

    def save_visualization(self, image: np.ndarray, mask: np.ndarray, output_path: str, alpha: float = 0.6):
        """
        Create visualization with proper size matching between image and mask.
        """
        # Ensure image and mask have matching dimensions
        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape[:2]

        if h_img != h_mask or w_img != w_mask:
            self._log_info(f"Size mismatch detected. Resizing mask from {mask.shape} to match image {image.shape[:2]}")
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        try:
            image_bgr = image[:, :, ::-1] if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            colored_mask = self.create_colored_mask(mask)

            # Ensure both images have the same dimensions before addWeighted
            if image_bgr.shape != colored_mask.shape:
                if len(colored_mask.shape) == 2:
                    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)
                elif colored_mask.shape[2] == 1:
                    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2BGR)

                # Final size check
                if image_bgr.shape[:2] != colored_mask.shape[:2]:
                    colored_mask = cv2.resize(colored_mask, (image_bgr.shape[1], image_bgr.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)

            overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)
            visualization = np.vstack((colored_mask, overlay))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale, font_color, thickness = 1.0, (255, 255, 255), 2
            h_orig = image_bgr.shape[0]

            cv2.putText(visualization, 'Segmentation', (20, 50), font, font_scale, font_color, thickness)
            cv2.putText(visualization, 'Overlay', (20, h_orig + 50), font, font_scale, font_color, thickness)

            cv2.imwrite(output_path, visualization)
            self._log_info(f"Visualization saved successfully to {output_path}")

        except Exception as e:
            self._log_error(f"Failed to create visualization overlay: {e}")
            try:
                image_bgr = image[:, :, ::-1] if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                colored_mask = self.create_colored_mask(mask)

                # Simple vertical stack without overlay
                visualization = np.vstack((image_bgr, colored_mask))
                cv2.imwrite(output_path, visualization)
                self._log_info(f"Simplified visualization saved to {output_path}")
            except Exception as e2:
                self._log_error(f"Failed to save even simplified visualization: {e2}")

    def group_image_pieces(self, image_dir: str) -> Dict[str, List[List[str]]]:
        all_files = defaultdict(list)
        for img_file in sorted(os.listdir(image_dir)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    parsed_data = self.parse_full_image_metadata(img_file)
                    prefix = parsed_data['prefix']
                    if prefix is None: continue
                    all_files[prefix].append((parsed_data['pic_no'], os.path.join(image_dir, img_file)))
                except Exception as e:
                    self._log_error(f"Warning: Could not parse {img_file}: {e}")

        grouped_files = {}
        for prefix, files_with_numbers in all_files.items():
            files_with_numbers.sort(key=lambda x: x[0])
            sequences, current_sequence, last_number = [], [], None
            for number, filepath in files_with_numbers:
                if last_number is None or number == last_number + 1:
                    current_sequence.append(filepath)
                else:
                    if current_sequence: sequences.append(current_sequence)
                    current_sequence = [filepath]
                last_number = number
            if current_sequence: sequences.append(current_sequence)
            grouped_files[prefix] = sequences
            self._log_info(
                f"Prefix '{prefix}': Found {len(sequences)} sequences with lengths {[len(seq) for seq in sequences]}")
        return grouped_files

    def parse_full_image_metadata(self, filepath: str) -> Dict:
        filename = os.path.basename(filepath)
        filename_no_ext, _ = os.path.splitext(filename)
        parts = filename_no_ext.split('_')

        if len(parts) >= 2 and re.match(r'^L\d+$', parts[-2]) and re.match(r'^\d{4}$', parts[-1]):
            return {'bureau': parts[0], 'train_group': parts[1], 'pass_time': parts[2],
                    'station': "NO_ROTATION_STATION_L", 'row': parts[4], 'car_seq': -1, 'channel': parts[5],
                    'pic_no': int(parts[6]), 'prefix': f'{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}'}

        if len(parts) < 7:
            raise ValueError(f"Filename {filename} does not have enough parts.")

        car_seq_str, channel_pic_no_str = parts[-2], parts[-1]
        if not (car_seq_str.isdigit() and channel_pic_no_str.startswith('10X') and len(channel_pic_no_str) >= 7):
            raise ValueError(f"Invalid format in {filename}")

        try:
            channel, pic_no = int(channel_pic_no_str[3]), int(channel_pic_no_str[4:])
        except ValueError:
            raise ValueError(f"Numeric conversion failed for {filename}")

        prefix = f'{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_10X{channel}'
        return {
            'bureau': parts[0], 'train_group': parts[1], 'pass_time': parts[2],
            'station': parts[3], 'row': parts[4], 'car_seq': int(car_seq_str),
            'channel': channel, 'pic_no': pic_no, 'prefix': prefix
        }

    def get_rotation_flag(self, channel_str: str, station_name: str, station_dict: dict) -> bool:
        if station_name == "NO_ROTATION_STATION_L":
            return True
            # False
        station_key = ""
        if channel_str in '1234':
            station_key = f'{station_name}_1'
        elif channel_str in '56789':
            candidate_key = f'{station_name}_{channel_str}'
            if candidate_key in station_dict:
                station_key = candidate_key
            else:
                station_key = f'{station_name}_5'
        return station_dict.get(station_key) == 1

    def cleanup_gpu_memory(self):
        """Clean up GPU memory and clear cache"""
        try:
            torch.cuda.empty_cache()
            self._log_info(f"GPU memory cleaned up successfully")
        except Exception as e:
            self._log_error(f"Warning: Failed to clean up GPU memory: {e}")

    def get_gpu_memory_info(self):
        """Print GPU memory information for the assigned GPU"""
        try:
            allocated = torch.cuda.memory_allocated(self.device) / 512 ** 3
            cached = torch.cuda.memory_reserved(self.device) / 512 ** 3
            total = torch.cuda.get_device_properties(self.device).total_memory / 512 ** 3
            self._log_info(
                f"GPU {self.gpu_id}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
        except Exception as e:
            self._log_error(f"GPU {self.gpu_id}: Error getting memory info - {e}")

    def process_single_sequence(self, prefix: str, seq_id: int, image_paths: List[str], output_dir: str):
        """
        The complete end-to-end processing for one sequence of images on a single GPU.
        This function is designed to be run in a separate process.
        """
        sequence_name = f"{prefix}_seq{seq_id}"
        self._log_info(f"Processing {sequence_name} with {len(image_paths)} images...")

        # Parse metadata from the first image path to determine rules
        try:
            parsed_data = self.parse_full_image_metadata(image_paths[0])
            is_large_image = (parsed_data["station"] == "NO_ROTATION_STATION_L" and str(parsed_data["channel"]).startswith("L"))
        except Exception as e:
            self._log_error(f"Metadata parsing failed for {sequence_name}, using defaults: {e}")
            parsed_data = {'channel': -1, 'train_group': 'UNKNOWN', 'row': ''}
            is_large_image = False
        
        # Backup original pixel ranges
        original_pixel_ranges = self.pixel_ranges.copy()

        if is_large_image:
            self._log_info(f"Large image detected in {sequence_name}. Applying 5x pixel ranges.")
            scale_factor = 5
            scaled_ranges = {}
            for cls, (min_px, max_px) in original_pixel_ranges.items():
                new_min = int(min_px * scale_factor) if min_px not in (None, float("inf")) else min_px
                if max_px == float("inf"):
                    new_max = max_px
                else:
                    new_max = int(max_px * scale_factor)
                scaled_ranges[cls] = (new_min, new_max)
            self.pixel_ranges = scaled_ranges

        try:
            # Step 1: Concatenate images and create chunks
            original_concatenated, chunks, chunk_height, filtered_paths, scale_factor, image_heights = self.intelligent_concatenation(
                image_paths)
            if not chunks or original_concatenated is None:
                self._log_error(f"Failed to create chunks for {sequence_name}. Skipping.")
                return None, None

            # Step 2: Run inference on all chunks
            chunk_masks = self._process_chunks(chunks)

            # Step 3: Merge chunk masks
            chunk_positions = []
            stride = chunk_height - self.overlap
            for i, mask in enumerate(chunk_masks):
                y_start = i * stride
                y_end = y_start + mask.shape[0]
                chunk_positions.append((y_start, y_end))
            raw_final_mask = self.merge_chunk_masks(chunk_masks, chunk_positions)

            if raw_final_mask is None:
                self._log_error(f"Failed to merge masks for {sequence_name}. Skipping.")
                return None, None

            h_orig, w_orig = original_concatenated.shape[:2]
            if raw_final_mask.shape[0] > h_orig:
                raw_final_mask = raw_final_mask[:h_orig, :]
            elif raw_final_mask.shape[0] < h_orig:
                padding = np.zeros((h_orig - raw_final_mask.shape[0], w_orig), dtype=np.uint8)
                raw_final_mask = np.vstack([raw_final_mask, padding])

            # Step 4: Trim blank areas
            row_means = np.mean(original_concatenated, axis=(1, 2))
            non_blank_rows = np.where(row_means >= 10.0)[0]
            start_row, end_row = (non_blank_rows[0], non_blank_rows[-1] + 1) if len(non_blank_rows) > 0 else (0, h_orig)

            img_trimmed = original_concatenated[start_row:end_row, :]
            mask_trimmed_raw = raw_final_mask[start_row:end_row, :]

            # Step 5: Validate components
            mask_trimmed_filtered, validation_stats = self.validate_connected_components(mask_trimmed_raw)

            # Step 6: Save artifacts (Disabled as per user's code state)
            # colored_mask_path = os.path.join(output_dir, 'masks', f"{sequence_name}_colored_mask.png")
            # vis_path = os.path.join(output_dir, 'visualizations', f"{sequence_name}_visualization.png")
            # cv2.imwrite(colored_mask_path, self.create_colored_mask(mask_trimmed_filtered))
            # self.save_visualization(img_trimmed, mask_trimmed_filtered, vis_path)

            # Step 7: Cut target groups and get their metadata (MODIFIED CALL)
            sequence_cut_metadata = self.cut_target_groups(
                img_trimmed, 
                mask_trimmed_filtered, 
                prefix,
                seq_id,
                output_dir,
                parsed_data  # Pass the parsed metadata
            )

            # Step 7a: NEW - Identify and copy original images without any targets
            self.identify_and_copy_no_target_images(
                filtered_paths=filtered_paths,
                image_heights=image_heights,
                cut_metadata=sequence_cut_metadata,
                trim_start_row=start_row,
                output_dir=output_dir
            )
            
            all_cut_metadata_for_sequence = sequence_cut_metadata

            # Step 8: Compile metadata
            raw_unique, final_unique = np.unique(mask_trimmed_raw), np.unique(mask_trimmed_filtered)
            metadata = {
                'sequence_name': sequence_name, 'original_prefix': prefix, 'sequence_id': seq_id,
                'num_input_images': len(filtered_paths), 'num_processing_chunks': len(chunks),
                'chunk_size': f"{chunk_height}x{chunks[0].shape[1]}", 'overlap': self.overlap,
                'global_scale_factor': scale_factor, 'gpu_used': self.gpu_id,
                'input_image_paths': filtered_paths,
                'start_row': start_row, 'end_row': end_row,
                'final_image_shape': img_trimmed.shape[:2],
                'classes_found_raw': [self.class_names[c] for c in raw_unique],
                'class_pixel_counts_raw': {self.class_names[c]: int(np.sum(mask_trimmed_raw == c)) for c in raw_unique},
                'classes_found_filtered': [self.class_names[c] for c in final_unique],
                'class_pixel_counts_filtered': {self.class_names[c]: int(np.sum(mask_trimmed_filtered == c)) for c in
                                                final_unique},
                'validation_stats': dict(validation_stats),
                'pixel_ranges_applied': self.pixel_ranges,
                'total_pixels': int(mask_trimmed_filtered.size)
            }

            self._log_info(f"Successfully processed {sequence_name}.")
            print(f"Successfully processed {sequence_name}.")
            return self.convert_to_json_serializable(metadata), all_cut_metadata_for_sequence

        finally:
            # Always restore original pixel ranges
            self.pixel_ranges = original_pixel_ranges


def gpu_worker(original_gpu_id: int,
               task_queue: mp.Queue,
               result_queue: mp.Queue,
               config_path: str,
               checkpoint_path: str,
               log_manager: LogManager):
    """
    Dedicated process for a single GPU.
    Loads the model once and processes tasks pulled from a shared queue.
    """
    import os
    import torch
    import traceback

    # Hard-isolate the device so this process only ever sees ONE GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(original_gpu_id)

    # IMPORTANT: after masking, the only visible GPU is '0' to this process.
    visible_gpu_id = 0

    processor = None
    try:
        torch.cuda.set_device(visible_gpu_id)
        # Load one model per GPU process
        processor = MultiGPULongImageProcessor(
            config_path,
            checkpoint_path,
            gpu_id=original_gpu_id,  # keep original id for logging
            log_manager=log_manager
        )

        while True:
            item = task_queue.get()
            if item is None:  # sentinel for shutdown
                task_queue.task_done()
                break

            try:
                prefix, seq_id, image_paths, output_dir = item
                meta, cut_meta = processor.process_single_sequence(prefix, seq_id, image_paths, output_dir)
                result_queue.put((meta, cut_meta))
            except Exception as e:
                log_manager.log_error(
                    f"Critical error on GPU {original_gpu_id} while processing {prefix}_seq{seq_id}: {e}",
                    original_gpu_id, exc_info=True)
                result_queue.put((None, None))
            finally:
                # Keep memory tidy to avoid fragmentation over long runs
                try:
                    processor.cleanup_gpu_memory()
                    torch.cuda.ipc_collect()
                except Exception as _:
                    pass

            task_queue.task_done()

    finally:
        if processor:
            try:
                processor.cleanup_gpu_memory()
            except Exception:
                pass
            del processor
        # Final flush
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# Worker function to be executed by each process in the pool
def process_sequence_worker(args):
    """
    Initializes a processor for a specific GPU and runs the processing for one sequence.
    """
    config_path, checkpoint_path, gpu_id, prefix, seq_id, image_paths, output_dir, log_manager = args

    processor = None
    try:
        # Each process gets its own processor instance, pinned to a specific GPU
        processor = MultiGPULongImageProcessor(config_path, checkpoint_path, gpu_id=gpu_id, log_manager=log_manager)
        return processor.process_single_sequence(prefix, seq_id, image_paths, output_dir)
    except Exception as e:
        log_manager.log_error(f"Critical error in worker for GPU {gpu_id} processing {prefix}_seq{seq_id}: {e}", gpu_id,
                              exc_info=True)
        return None, None
    finally:
        # Clean up GPU memory in the worker process before it exits
        if processor:
            processor.cleanup_gpu_memory()
            del processor


def process_sequences_parallel(image_dirs: Union[str, List[str]], output_dir: str,
                               config_path: str, checkpoint_path: str,
                               gpu_ids: List[int], log_manager: LogManager):
    """
    Orchestrator that launches one dedicated process per GPU.
    Each GPU process loads the model once and pulls work items from a shared queue.
    Supports one or multiple input directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target_cuts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'no_target'), exist_ok=True) # Ensure no_target dir exists

    # Normalize input into a list
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    # Lightweight helper for grouping (CPU only, no model)
    helper_processor = MultiGPULongImageProcessor(config_path, checkpoint_path, gpu_id=gpu_ids[0], load_model=False,
                                                  log_manager=log_manager)

    # Collect image groups from all dirs
    all_image_groups = {}
    for image_dir in image_dirs:
        groups = helper_processor.group_image_pieces(image_dir)
        for prefix, seqs in groups.items():
            if prefix in all_image_groups:
                all_image_groups[prefix].extend(seqs)
            else:
                all_image_groups[prefix] = seqs
        log_manager.log_info(f"Scanned directory {image_dir}, found {len(groups)} prefixes.")

    del helper_processor

    # Tunable parameters for handling very long sequences
    VERY_LONG_THRESHOLD = 50
    SUB_SEQUENCE_SIZE = 15

    # Build flat task list with subdivision logic
    tasks = []
    for prefix, sequences in all_image_groups.items():
        for seq_id, image_paths in enumerate(sequences):
            if len(image_paths) > VERY_LONG_THRESHOLD:
                log_manager.log_info(
                    f"Prefix '{prefix}' seq {seq_id} is very long ({len(image_paths)} images). "
                    f"Subdividing into chunks of {SUB_SEQUENCE_SIZE}.")

                for i in range(0, len(image_paths), SUB_SEQUENCE_SIZE):
                    sub_image_paths = image_paths[i:i + SUB_SEQUENCE_SIZE]
                    if not sub_image_paths:
                        continue
                    sub_seq_id = f"{seq_id}_sub{i // SUB_SEQUENCE_SIZE}"
                    tasks.append((prefix, sub_seq_id, sub_image_paths, output_dir))
                    log_manager.log_info(f"  -> Created sub-task {sub_seq_id} with {len(sub_image_paths)} images.")
            else:
                tasks.append((prefix, seq_id, image_paths, output_dir))

    num_gpus = len(gpu_ids)
    log_manager.log_info(f"Found {len(tasks)} total sequences. Spawning {num_gpus} GPU workers: {gpu_ids}")

    if len(tasks) == 0:
        log_manager.log_info("No work to do.")
        return

    # Shared queues
    task_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=max(4 * num_gpus, 8))
    result_queue: mp.Queue = mp.Queue()

    # Start one worker process per GPU
    workers: List[mp.Process] = []
    for gid in gpu_ids:
        p = mp.Process(
            target=gpu_worker,
            args=(gid, task_queue, result_queue, config_path, checkpoint_path, log_manager),
            daemon=True
        )
        p.start()
        workers.append(p)

    # Feed tasks
    for item in tasks:
        task_queue.put(item)

    # Add sentinel Nones to stop workers
    for _ in gpu_ids:
        task_queue.put(None)

    all_metadata = []
    all_cut_metadata = []
    completed = 0
    total = len(tasks)

    # Consume results
    while completed < total:
        meta, cut_meta = result_queue.get()
        if meta:
            all_metadata.append(meta)
        if cut_meta:
            all_cut_metadata.extend(cut_meta)
        completed += 1
        log_manager.log_info(f"Progress: {completed}/{total} sequences completed.")

    task_queue.join()

    for p in workers:
        p.join()

    # Save results
    metadata_path = os.path.join(output_dir, 'processing_metadata.json')
    cutdata_path = os.path.join(output_dir, 'cut_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(sorted(all_metadata, key=lambda x: x['sequence_name']), f, indent=2)
    with open(cutdata_path, 'w') as f:
        json.dump(all_cut_metadata, f, indent=2)

    log_manager.log_info(f"Parallel processing complete. {len(all_metadata)} sequences processed.")
    log_manager.log_info(f"Results saved to {output_dir}")
    log_manager.log_info(f"Metadata saved to {metadata_path}")

def main():
    """Main function to run the multi-GPU processor with pixel validation and logging"""
    config_path = "/home/suma/bin/beit2/unilm/beit2/semantic_segmentation/configs/beit/upernet/upernet_beit_base_12_512_slide_160k_21ktoade20k.py"
    checkpoint_path = "/data1/beit2/finetune4/iter_52000.pth"

    gpu_ids_to_use = [0, 1, 2, 3, 4, 5, 6, 7]

    available_gpus = []
    if torch.cuda.is_available():
        for i in gpu_ids_to_use:
            try:
                torch.cuda.get_device_name(i)
                available_gpus.append(i)
            except Exception:
                print(f"Warning: GPU {i} is not available or invalid.", file=sys.stderr)

    if not available_gpus:
        print("Error: No CUDA-enabled GPUs available. Exiting.", file=sys.stderr)
        return

    # ✅ Accept one or multiple input folders
    image_dirs = [
        '/data1/beit2/data/xxx',
    ]

    output_dir = '/data1/beit2/log/xxx'
    log_dir = os.path.join(output_dir, 'logs')

    log_manager = LogManager(log_dir)

    log_manager.log_info(f"Starting parallel image processing with {len(available_gpus)} GPUs: {available_gpus}")
    start_time = time.time()

    print(f"Processing started. Logs are being written to: {log_dir}")
    print(f"Check main log: {os.path.join(log_dir, f'main_{log_manager.timestamp}.log')}")

    # ✅ Pass one or multiple dirs
    process_sequences_parallel(image_dirs, output_dir, config_path, checkpoint_path, available_gpus, log_manager)

    total_time = time.time() - start_time
    log_manager.log_info(f"Processing completed successfully!")
    log_manager.log_info(f"Total time: {total_time:.2f}s")

    print(f"Processing completed successfully! Total time: {total_time:.2f}s")
    print(f"Check logs in {log_dir} for detailed information.")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    # 'spawn' is safer and recommended for CUDA
    mp.set_start_method('spawn', force=True)
    main()
