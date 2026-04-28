# filename: batch_inference.py
import torch
import argparse
import json
import re
import gc
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Optional, Any

# Allows loading of potentially truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- ------------------------------------------------------------------ ---
# --- Logic Ported from generate_prompt.py for Image Rotation            ---
# --- ------------------------------------------------------------------ ---

# This folder path determines rotation based on the image's absolute location
SPECIAL_ROTATION_FOLDER = '/data1/beit2/data/guangzhou_simulation_large'

def load_station_dict(json_path: str) -> dict:
    """Loads the station rotation JSON file safely."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Station rotation file not found at '{json_path}'. No name-based rotation will be applied.")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from '{json_path}'. Please check the file's format.")
    except Exception as e:
        print(f"Warning: An error occurred while loading '{json_path}': {e}")
    return {}

def parse_inference_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parses filenames like 'ori017_5_NANCHANGXIZHANCHANGJIUXIAXING.jpg'
    to extract channel and station information for rotation logic.
    """
    # Regex for the format: ori..._CHANNEL_STATION.jpg
    pattern = re.compile(r'ori\d+_(?P<channel>\d)_(?P<station>.+?)\.jpg$')
    match = pattern.match(filename)
    if match:
        return match.groupdict()
    return None

def get_rotation_flag_from_name(channel_str: str, station_name: str, station_dict: dict) -> bool:
    """
    Determines if rotation is needed based on channel, station name, and the station dictionary.
    This logic is directly from `get_rotation_flag` in your provided script.
    """
    if station_name == "NO_ROTATION_STATION_L":
        return False
    station_key = ""
    if channel_str in '1234':
        station_key = f'{station_name}_1'
    elif channel_str in '56789':
        candidate_key = f'{station_name}_{channel_str}'
        station_key = candidate_key if candidate_key in station_dict else f'{station_name}_5'
    
    return station_dict.get(station_key) == 1

def needs_rotation(image_path: Path, station_dict: dict) -> bool:
    """
    Checks if an image needs rotation based on its file path or filename.
    This combines the rotation logic from `create_qwen_vl_instance`.
    """
    # 1. Check if the image is in the special rotation folder
    if SPECIAL_ROTATION_FOLDER in str(image_path.resolve()):
        return True

    # 2. Check based on filename parsing and station dictionary
    parsed_info = parse_inference_filename(image_path.name)
    if parsed_info:
        station_name = parsed_info.get('station', '')
        channel = parsed_info.get('channel', '')
        if station_name and channel:
            return get_rotation_flag_from_name(str(channel), station_name, station_dict)

    return False

# --- ------------------------------------------------------------------ ---
# --- Main Inference and File Processing Logic                           ---
# --- ------------------------------------------------------------------ ---

def find_images(root_dirs: List[str]) -> List[Dict[str, str]]:
    """
    Finds all 'ori*.jpg' images in the provided directories and pairs them
    with their corresponding 'bbox.jpg' file if it exists.
    """
    image_pairs = []
    for root_dir in root_dirs:
        p_root = Path(root_dir)
        if not p_root.is_dir():
            print(f"Warning: Directory not found, skipping: {root_dir}")
            continue
        
        print(f"🔍 Searching for images in '{p_root}'...")
        # Use rglob to find all ori*.jpg files recursively
        for ori_path in tqdm(list(p_root.rglob("ori*.jpg")), desc=f"Scanning {p_root.name}"):
            # Look for bbox.jpg in the same directory
            bbox_path = ori_path.parent / 'bbox.jpg'
            image_pairs.append({
                'ori_path': str(ori_path),
                'bbox_path': str(bbox_path) if bbox_path.exists() else None
            })
    print(f"✅ Found {len(image_pairs)} images to process.")
    return image_pairs

def run_inference(args):
    """
    Main function to load the model, process images in batches, and save results.
    """
    # --- 1. Setup ---
    print("🚀 Initializing model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # IMPROVEMENT 1: Added offload_buffers=True to help with memory management.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto", offload_buffers=True
        )
        # THE FIX: Added padding_side='left' to the processor.
        processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')
        
    except Exception as e:
        print(f"❌ Error: Could not load the model from '{args.model_path}'. Please check the path.")
        print(f"Details: {e}")
        return
    print(f"✅ Model loaded successfully on device: {device}")

    # Load station dictionary for rotation logic
    station_dict = load_station_dict(args.station_rotation_file)

    # --- 2. Image Discovery ---
    # MODIFICATION: Load image paths from the JSON file
    print(f"📚 Loading image list from '{args.image_list_json}'...")
    try:
        with open(args.image_list_json, 'r', encoding='utf-8') as f:
            image_files = json.load(f)
    except Exception as e:
        print(f"❌ Error: Could not read or parse the JSON file: {e}")
        return
        
    if not image_files:
        print("No images found in the JSON file. Exiting.")
        return
    print(f"✅ Found {len(image_files)} images to process.")

    # --- 3. Batch Processing ---
    all_results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(image_files), batch_size), desc="🔥 Processing batches"):
        batch_files_info = image_files[i:i+batch_size]
        batch_pil_images = []
        valid_batch_files_info = []

        for file_info in batch_files_info:
            try:
                img_path = Path(file_info['ori_path'])
                img = Image.open(img_path).convert('RGB')
                
                if needs_rotation(img_path, station_dict):
                    img = img.rotate(270, expand=True) 
                
                batch_pil_images.append(img)
                valid_batch_files_info.append(file_info)

            except Exception as e:
                print(f"\nWarning: Skipping image {file_info['ori_path']} due to error: {e}")

        if not batch_pil_images:
            continue

        prompts = []
        for _ in batch_pil_images:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        
        try:
            inputs = processor(
                text=prompts,
                images=batch_pil_images,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            
            # IMPROVEMENT 2: More robust way to remove the prompt from the output.
            # This slices the input tokens from the generated tokens before decoding.
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for idx, result in enumerate(output_text):
                file_info = valid_batch_files_info[idx]
                all_results.append({
                    'original_image_path': file_info['ori_path'],
                    'bbox_image_path': file_info['bbox_path'],
                    'inference_result': result.strip()
                })

        except Exception as e:
            print(f"\n❌ Error during model inference on a batch: {e}")
        
        del batch_pil_images, prompts, inputs, generated_ids, generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 4. Save Results ---
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\n✨ Success! Inference complete. {len(all_results)} results saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference script for Qwen 2.5 VL on fault images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # MODIFICATION: Change --input-dirs to --image-list-json
    parser.add_argument(
        '--image-list-json', 
        type=str,
        default='/data1/qwen/inference/output.json',
        help="Path to the JSON file containing the list of image paths."
    )
    parser.add_argument(
        '--model-path', 
        type=str, 
        default="/data1/qwen/train", 
        help="Path to the pretrained Qwen 2.5 VL model directory."
    )
    parser.add_argument(
        '--station-rotation-file', 
        type=str, 
        default="/home/suma/bin/try_detectron2/multipleGPUs/station_rotation.json", 
        help="Path to the 'station_rotation.json' file for rotation logic."
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default='/data1/qwen/inference/inference_results.json',
        help="Path to save the final JSON file with all results."
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=8, 
        help="Number of images to process in a single batch."
    )
    parser.add_argument(
        '--prompt', 
        type=str, 
        default="Describe this image in detail.", 
        help="The text prompt to use for each image."
    )
    parser.add_argument(
        '--max-new-tokens', 
        type=int, 
        default=512, 
        help="Maximum number of new tokens for the model to generate."
    )
    
    args = parser.parse_args()
    run_inference(args)

if __name__ == "__main__":
    main()