import os
import pickle
import numpy as np
from PIL import Image

# Set up output directory.
dest_root = '/data1/beit2/finetune1/all_52000/'
pkl_root = '/data1/beit2/finetune1/all_52000.pkl'
data_root = '/data1/beit2/all'  # '/data1/beit2/ADE/images/test'
os.makedirs(dest_root, exist_ok=True)

# Load predictions (assumed to be binary masks in 0 or 1).
with open(pkl_root, 'rb') as f:
    predictions = pickle.load(f)

# Get list of test image paths.
image_names = [
    entry.path
    for entry in os.scandir(data_root)
    if entry.name.endswith('png') or entry.name.endswith('jpg')
]
assert len(image_names) == len(predictions), "Mismatch in image/prediction counts!"

# Parameters.
alpha_val = 0.3  # blending factor: 0 => original, 1 => fully overlay color.
overlay_color = np.array([255, 182, 193], dtype=np.float32)  # light overlay color (RGB).

for image_name, pred_mask in zip(image_names, predictions):
    # Load original image as a NumPy array.
    orig_img = np.array(Image.open(image_name).convert("RGB"))

    # Ensure the mask is float32 and has the same spatial dimensions.
    # If your predictions are binary (0 and 1) but not boolean, convert them to float32.
    mask = pred_mask.astype(np.float32)

    # Expand dimensions from (H, W) to (H, W, 1) for broadcasting.
    mask = mask[..., None]

    # Convert the original image to float32.
    orig_img_f = orig_img.astype(np.float32)

    # Apply vectorized blending over the entire image:
    #   new_pixel = (1 - alpha_val * mask) * original_pixel + (alpha_val * mask) * overlay_color
    blended_img = (1 - alpha_val * mask) * orig_img_f + (alpha_val * mask) * overlay_color

    # Clip the values and convert back to uint8.
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)

    # Save the annotated image.
    base_name = os.path.basename(image_name)
    prefix, _ = os.path.splitext(base_name)
    annotated_path = os.path.join(dest_root, prefix + '.jpg')
    Image.fromarray(blended_img).save(annotated_path)
    Image.fromarray(blended_img).save(annotated_path)
