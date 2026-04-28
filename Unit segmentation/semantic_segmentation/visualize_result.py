import json
import os
from PIL import Image, ImageDraw, ImageFont

# --- Script ---

def visualize_annotations_updated(json_path, image_root, save_path):
    """
    Visualizes bounding box annotations on images with larger, centered text
    and saves them.

    Args:
        json_path (str): The file path for the JSON annotations.
        image_root (str): The root directory of the original images.
        save_path (str): The directory where visualized images will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created output directory: {save_path}")

    # Load the annotations from the JSON file
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    print(f"Loaded {len(annotations)} annotations.")

    # Process each image annotation
    for image_name, data in annotations.items():
        image_path = os.path.join(image_root, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}. Skipping.")
            continue

        try:
            # Open the image
            with Image.open(image_path).convert("RGB") as img:
                draw = ImageDraw.Draw(img)

                # Use a larger font size.
                try:
                    font = ImageFont.truetype("arial.ttf", 300)
                except IOError:
                    font = ImageFont.load_default()

                # Draw each bounding box
                for detection in data.get('bboxes', []):
                    bbox = detection['bbox']
                    class_name = detection['class_name']

                    # Bbox format is [xmin, ymin, xmax, ymax]
                    xmin, ymin, xmax, ymax = bbox

                    # Draw the rectangle
                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=5)

                    # --- MODIFIED SECTION for Centered Text ---
                    text = f"{class_name}"

                    # Get the bounding box of the text itself
                    try:
                        # For newer Pillow versions
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError:
                        # Fallback for older Pillow versions
                        text_width, text_height = draw.textsize(text, font=font)

                    # Calculate the center of the main bounding box
                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    center_x = xmin + box_width / 2
                    center_y = ymin + box_height / 2

                    # Calculate the top-left position for the text to be centered
                    text_x = center_x - (text_width / 2)
                    text_y = center_y - (text_height / 2)

                    # Draw the text
                    draw.text((text_x, text_y), text, fill="white", font=font)
                    # --- END MODIFIED SECTION ---

                # Save the new image
                output_filename = os.path.splitext(image_name)[0] + '.jpg'
                output_image_path = os.path.join(save_path, output_filename)
                img.save(output_image_path, 'JPEG')

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print("\nVisualization complete!")
    print(f"Annotated images are saved in: {save_path}")


if __name__ == '__main__':
    json_file_path = '/data1/beit2/finetune3/fault/single_images_results.json'  # Path to your JSON annotation file.
    image_root_path = '/data1/beit2/running_record/finetune8/data'  # Path to the folder where original images are stored.
    output_save_path = '/data1/beit2/finetune3/fault/vis'  # Path to the folder where visualized images will be saved.


    if not os.path.exists(image_root_path):
        os.makedirs(image_root_path)


    visualize_annotations_updated(json_file_path, image_root_path, output_save_path)

