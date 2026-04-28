# filename: generate_unified_prompts.py
import os
import re
import json
import argparse
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
from collections import defaultdict

# --- Configuration & Global Mappings ---

CHANNEL_MAPPING = {
    '1': "right top", '2': "left top", '3': "right bottom", '4': "left bottom",
    '5': "bottom center", '6': "bottom left", '7': "bottom right",
    '8': "trackside left", '9': "trackside right",
    'L5': "bottom center", 'L3': "trackside", 'L7': "trackside",
    'L4': "inner", 'L6': "inner"
}

# --- Generic Helper Functions ---

def convert_time_to_description(time_str: str) -> str:
    """Converts a timestamp to a time-of-day string."""
    try:
        time_str = str(time_str) # Ensure it's a string
        hour = int(time_str[8:10])
        if 5 <= hour < 12: return 'in the morning'
        if 12 <= hour < 17: return 'in the afternoon'
        if 17 <= hour < 21: return 'in the evening'
        return 'at night'
    except (ValueError, IndexError):
        return 'during the daytime'

def get_position_from_channel(channel: str) -> str:
    """Gets the camera position description from its channel number."""
    return CHANNEL_MAPPING.get(str(channel), 'an unknown position')

def create_qwen_vl_instance(image_paths: List[Path], prompt: str, base_dir: Path) -> Dict:
    """Creates a dictionary in the Qwen-VL format for single or multi-image instances."""
    human_prompts = [
        "<image>\nDescribe this image in detail.",
        "<image>\nWhat do you see in this picture?",
        "<image>\nProvide a detailed description of the image."
    ]
    
    # Adjust format for single vs. multi-image
    if len(image_paths) == 1:
        instance = {
            "image": str(image_paths[0].relative_to(base_dir)),
            "conversations": [
                {"from": "human", "value": random.choice(human_prompts)},
                {"from": "gpt", "value": prompt}
            ]
        }
    else:
        relative_paths = [str(p.relative_to(base_dir)) for p in image_paths]
        human_value = "\n".join(["<image>"] * len(image_paths)) + "\nDescribe these images."
        instance = {
            "images": relative_paths,
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": prompt}
            ]
        }
    return instance

# --- Case 1: Target Group Image Processing ---

def parse_target_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parses filenames for target group images (Case 1)."""
    patterns = [
        re.compile(r'^(?P<bureau>.+?)_(?P<train_group>.+?)_(?P<pass_time>\d+)_(?P<location>.+?)_(?P<row>\d+)_(?P<car_seq>\d+)_10X(?P<channel>\d)_.+?_group\d+_(?P<components>.+?)_image\.png$'),
        re.compile(r'^(?P<bureau>.+?)_(?P<train_group>.+?)_(?P<pass_time>\d+)_(?P<location>.+?)_(?P<row>\d+)_(?P<channel>L\d)_.+?_sub\d+_group\d+_(?P<components>.+?)_image\.png$'),
        re.compile(r'^(?P<bureau>.+?)_(?P<train_group>.+?)_(?P<pass_time>\d+)_(?P<location>.+?)_(?P<row>\d+)_(?P<channel>L\d)_.+?_group\d+_(?P<components>.+?)_image\.png$')
    ]
    for pattern in patterns:
        match = pattern.match(filename)
        if match: return match.groupdict()
    return None

def process_target_images(target_dirs: List[str], base_dir: Path) -> List[Dict]:
    """Finds and processes all target group images."""
    instances = []
    wheel_channels = {'1', '2', '3', '4', '8', '9', 'L1', 'L2', 'L3', 'L7', 'L8', 'L9'}

    for directory in target_dirs:
        for image_path in tqdm(Path(directory).rglob('*_image.png'), desc="⚙️ Processing Target Images (Case 1)"):
            if not Path(str(image_path).replace('_image.png', '_mask.png')).exists():
                continue

            parsed_info = parse_target_filename(image_path.name)
            if not parsed_info:
                continue

            try:
                channel = parsed_info['channel']
                train_type = parsed_info['train_group'].rsplit('-', 1)[0]
                time_desc = convert_time_to_description(parsed_info['pass_time'])
                position = get_position_from_channel(channel)

                # --- normalize components ---
                components_raw = parsed_info['components'].replace('_', ' ')
                components_list = components_raw.split()

                # ✅ Wheel-specific filtering
                if channel in wheel_channels:
                    if "wheel" not in components_list:
                        continue  # Skip if wheel not in components
                    components_list = ["wheel"]  # Keep only wheel

                # ✅ CRH5 axle filtering
                if train_type.startswith("CRH5") and channel in {"5", "L5"}:
                    if "axle" in components_list and "wheel" in components_list and "gearbox" in components_list:
                        components_list = [c for c in components_list if c != "axle"]

                # Rebuild components string
                if not components_list:
                    continue  # Skip if nothing left
                components = " ".join(components_list)

                prompt_location = "undercarriage" if channel in (
                    '5', '6', '7', '8', '9', 'L3', 'L4', 'L5', 'L6', 'L7'
                ) else "side"

                prompt = (
                    f"A photo of a train's {prompt_location}, showing {components} "
                    f"at the {position} of a {train_type} train {time_desc}."
                )
                instances.append(create_qwen_vl_instance([image_path], prompt, base_dir))

            except (KeyError, IndexError):
                continue

    return instances



# --- Case 2: Background Image Processing ---

def parse_background_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parses filenames for background images (Case 2)."""
    patterns = [
        re.compile(r'^(?P<bureau>.+?)_(?P<train_group>.+?)_(?P<pass_time>\d+)_(?P<location>.+?)_(?P<row>\d+)_(?P<car_seq>\d+)_10X(?P<channel>\d).*\.(jpg|jpeg|png)$'),
        re.compile(r'^(?P<bureau>.+?)_(?P<train_group>.+?)_(?P<pass_time>\d+)_(?P<location>.+?)_(?P<row>\d+)_(?P<channel>L\d)_.+\.(jpg|jpeg|png)$')
    ]
    for pattern in patterns:
        match = pattern.match(filename)
        if match: return match.groupdict()
    return None

def process_background_images(background_dirs: List[str], base_dir: Path) -> List[Dict]:
    """Finds and processes all background images."""
    instances = []
    for directory in background_dirs:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for image_path in tqdm(Path(directory).rglob(ext), desc="⚙️ Processing Background Images (Case 2)"):
                parsed_info = parse_background_filename(image_path.name)
                if parsed_info:
                    try:
                        train_type = parsed_info['train_group'].rsplit('-', 1)[0]
                        time_desc = convert_time_to_description(parsed_info['pass_time'])
                        position = get_position_from_channel(parsed_info['channel'])
                        prompt_location = "undercarriage" if parsed_info['channel'] in ('5', '6', '7', '8', '9', 'L3', 'L4', 'L5', 'L6', 'L7') else "side"
                        prompt = f"A background view from the {position} of a {train_type} train's {prompt_location} {time_desc}, with no key components visible."
                        instances.append(create_qwen_vl_instance([image_path], prompt, base_dir))
                    except (KeyError, IndexError): continue
    return instances

# --- Case 3: Fault Image Processing ---

class FaultPromptGenerator:
    """Loads metadata and generates prompts for fault images."""
    def __init__(self, csv_path, component_map_path, translation_cache_path):
        self.is_ready = False
        try:
            df = pd.read_csv(csv_path, dtype={'TYPE': str}, low_memory=False)
            self.metadata = df.set_index('ALERTID').to_dict('index')
            with open(translation_cache_path, 'r', encoding='utf-8') as f: self.translation_cache = json.load(f)
            with open(component_map_path, 'r', encoding='utf-8') as f: component_map = json.load(f)
            self.component_id_to_en = {k: v.replace('(', '').replace(')', '').strip() for k, v in component_map.items() if ',' not in k}
            self.is_ready = True
        except FileNotFoundError as e:
            print(f"Error: Could not initialize FaultPromptGenerator. Missing metadata file: {e}")

    def _get_component_name_from_code(self, code: str) -> str:
        if code in self.component_id_to_en: return self.component_id_to_en[code]
        for length in range(len(code), 0, -1):
            if code[:length] in self.component_id_to_en: return self.component_id_to_en[code[:length]]
        return 'unknown component'

    def generate_prompt(self, alert_id: str, parsed_filename_info: Dict) -> Optional[str]:
        record = self.metadata.get(alert_id)
        if not record: return None
        
        train_group = record.get('SETNO', '')
        train_type = train_group.rsplit('-', 1)[0] if '-' in train_group else train_group
        position = get_position_from_channel(parsed_filename_info['channel'])
        time_desc = convert_time_to_description(record.get('TIME', ''))
        comp_code = record.get('TYPECODE', '')
        comp_name = self._get_component_name_from_code(comp_code)
        fault_name_cn = record.get('ALARMCODE', '')
        fault_en = self.translation_cache.get(fault_name_cn, "an unspecified issue")

        prompt_location = "undercarriage" if parsed_filename_info['channel'] in ('5', '6', '7', '8', '9', 'L3', 'L4', 'L5', 'L6', 'L7') else "side"
        return f'A photo of a train\'s {prompt_location} showing a "{fault_en}" fault on a {comp_name}, viewed from the {position} of a {train_type} train {time_desc}.'

def parse_fault_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parses filenames for the new fault image format (Case 3)."""
    pattern = re.compile(
        r'^(?P<comp_name>.+?)_'
        r'(?P<fault_code>.+?)_'
        r'(?P<alert_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_'
        r'ori\d+_'
        r'(?P<channel>\d)_'
        r'(?P<location>.+?)\.jpg$'
    )
    match = pattern.match(filename)
    return match.groupdict() if match else None

def process_fault_images(fault_dirs: List[str], base_dir: Path, prompt_generator: FaultPromptGenerator) -> List[Dict]:
    """Finds, groups, and processes all fault images."""
    if not prompt_generator.is_ready: return []
    
    fault_groups = defaultdict(list)
    for directory in fault_dirs:
        for image_path in Path(directory).rglob('*_ori*_*.jpg'):
            parsed_info = parse_fault_filename(image_path.name)
            if parsed_info:
                alert_id = parsed_info['alert_id']
                fault_groups[alert_id].append(image_path)
    
    instances = []
    if not fault_groups: return []

    for alert_id, image_paths in tqdm(fault_groups.items(), desc="⚙️ Processing Fault Images (Case 3)"):
        first_image_info = parse_fault_filename(image_paths[0].name)
        if not first_image_info: continue
        
        prompt = prompt_generator.generate_prompt(alert_id, first_image_info)
        if prompt:
            instances.append(create_qwen_vl_instance(image_paths, prompt, base_dir))
    return instances

# --- Case 4: Manually Sorted Image Processing ---

def process_manual_images(manual_dirs: List[str], base_dir: Path) -> List[Dict]:
    """Finds and processes manually sorted images that were missed by the model."""
    instances = []
    for directory in manual_dirs:
        dir_path = Path(directory)
        # The component name is derived from the folder's name
        components = dir_path.name.replace('_', ' ')
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for image_path in tqdm(dir_path.rglob(ext), desc=f"⚙️ Processing Manual '{components}' (Case 4)"):
                # Use the background parser as the filename format is the same
                parsed_info = parse_background_filename(image_path.name)
                if parsed_info:
                    try:
                        train_type = parsed_info['train_group'].rsplit('-', 1)[0]
                        time_desc = convert_time_to_description(parsed_info['pass_time'])
                        position = get_position_from_channel(parsed_info['channel'])
                        prompt_location = "undercarriage" if parsed_info['channel'] in ('5', '6', '7', '8', '9', 'L3', 'L4', 'L5', 'L6', 'L7') else "side"
                        # Create a prompt similar to Case 1, using the folder name for components
                        prompt = f"A photo of a train's {prompt_location}, showing {components} at the {position} of a {train_type} train {time_desc}."
                        instances.append(create_qwen_vl_instance([image_path], prompt, base_dir))
                    except (KeyError, IndexError):
                        continue
    return instances

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate a unified Qwen-VL prompt dataset for train components, backgrounds, and faults.")
    parser.add_argument('--target-dirs', nargs='*', help="Directories for target group images (Case 1).")
    parser.add_argument('--background-dirs', nargs='*', help="Directories for background images (Case 2).")
    parser.add_argument('--fault-dirs', nargs='*', help="Directories for fault images (Case 3).")
    parser.add_argument('--manual-dirs', nargs='*', help="Directories for manually selected false category images (Case 4).")
    parser.add_argument('--csv-file', default='/data1/detectron2/backup/all_fault.csv', type=str,
                        help="Path to metadata CSV for faults (Case 3).")
    parser.add_argument('--component-map-file', default='/data1/detectron2/feature/component_map.json', type=str,
                        help="Path to component map JSON for faults (Case 3).")
    parser.add_argument('--translation-cache-file', default='/data1/detectron2/feature/fault_map.json', type=str,
                        help="Path to translation cache JSON for faults (Case 3).")
    parser.add_argument('--output-file', type=str, default='/data1/beit2/log/qwen_vl_simulation.json',
                        help="Path to the final output JSON file.")
    parser.add_argument('--base-dir', type=str, default='/data1/beit2',
                        help="Base directory for making image paths relative.")
    args = parser.parse_args()

    all_instances = []
    base_dir = Path(args.base_dir) if args.base_dir else Path(args.output_file).parent

    if args.target_dirs:
        all_instances.extend(process_target_images(args.target_dirs, base_dir))
    
    if args.background_dirs:
        all_instances.extend(process_background_images(args.background_dirs, base_dir))
        
    if args.fault_dirs:
        if not all([args.csv_file, args.component_map_file, args.translation_cache_file]):
            print("\nWarning: To process fault images, you must provide --csv-file, --component-map-file, and "
                  "--translation-cache-file. Skipping fault processing.")
        else:
            prompt_generator = FaultPromptGenerator(args.csv_file, args.component_map_file, args.translation_cache_file)
            all_instances.extend(process_fault_images(args.fault_dirs, base_dir, prompt_generator))
            
    if args.manual_dirs:
        all_instances.extend(process_manual_images(args.manual_dirs, base_dir))

    if not all_instances:
        print("\nNo images were processed. Please check your input directories and arguments. Exiting.")
        return

    random.shuffle(all_instances)
    
    print("\n--- 📝 Generation Summary ---")
    print(f"✔️ Total instances generated: {len(all_instances)}")
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_instances, f, indent=2, ensure_ascii=False)
        
    print(f"\n✨ Success! Unified dataset saved to '{output_path}'")

if __name__ == "__main__":
    main()