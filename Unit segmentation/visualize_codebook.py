import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# Import necessary components from your project
from timm.models import create_model
from datasets import build_vqkd_dataset
import modeling_vqkd
import utils
import torch.backends.cudnn as cudnn
from torchvision import transforms # <-- ADD THIS IMPORT
import torch.distributed as dist

# Default ImageNet normalization values (from timm.data.constants)
IMAGENET_DEFAULT_MEAN = (0.3043, 0.3043, 0.3043)
IMAGENET_DEFAULT_STD = (0.2216, 0.2216, 0.2216)


def get_args():
    parser = argparse.ArgumentParser('VQKD Codebook Visualization', add_help=False)
    parser.add_argument('--model', default='vqkd_encoder_base_decoder_3x768x12_clip', type=str, metavar='MODEL',
                        help='Name of model to visualize (e.g., vqkd_encoder_base_decoder_3x768x12_clip)')
    parser.add_argument('--model_path', default='/data1/beit2/vqkd_output1/checkpoint-299.pth', type=str,
                        # /data1/beit2/vqkd_output1/checkpoint-279.pth
                        help='Path to the trained VQKD model checkpoint')
    parser.add_argument('--data_path', default='/data1/beit2/data', type=str,
                        help='Path to your evaluation dataset (general data path, might be used for training too)')
    parser.add_argument('--output_dir', default='/data1/beit2/vqkd_output1/codebook_visualizations_train',
                        help='Path where to save visualization images')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Input image size for the VQKD encoder and dataset transforms')
    parser.add_argument('--train_interpolation', default='bicubic', type=str,
                        help='Interpolation method for training resize transforms (e.g., "bicubic")')
    parser.add_argument('--data_set', default='image_folder', type=str,
                        help='Dataset to use (e.g., "image_folder", "imagenet")')
    parser.add_argument('--eval_data_path', default='/data1/beit2/data', type=str,
                        # /data1/detectron2/normal1.20 /data1/beit2/data
                        help='Path to your evaluation dataset (specific for eval)')

    parser.add_argument('--codebook_n_emd', default=8192, type=int,
                        help='Number of embeddings in the codebook')
    parser.add_argument('--codebook_emd_dim', default=32, type=int,
                        help='Dimension of each codebook embedding')
    parser.add_argument('--batch_size', default=64, type=int,  # Increased default for efficiency
                        help='Batch size for evaluation dataset loading')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers (dataloader)')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='remove this flag if you use pytorch ddp')
    parser.add_argument('--dist_eval', action='store_true',
                        help='Enabling distributed evaluation (recommended for large datasets)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


# Helper function to load VQKD model (adapted from run_vqkd_training.py)
def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_weight=args.model_path,
        as_tokenzer=True,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        teacher_model_type=args.model.split('_')[-1] if 'clip' in args.model or 'dino' in args.model else 'None',
    ).eval()
    return model


# Function to denormalize image for visualization
def denormalize_image(tensor_image, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    # Ensure tensor_image is 4D (B, C, H, W) for broadcasting
    if tensor_image.dim() == 3:
        tensor_image = tensor_image.unsqueeze(0)

    mean_tensor = torch.tensor(mean).view(1, len(mean), 1, 1).to(tensor_image.device)
    std_tensor = torch.tensor(std).view(1, len(std), 1, 1).to(tensor_image.device)
    img_unnormalized = tensor_image * std_tensor + mean_tensor
    return img_unnormalized.clamp(0, 1).squeeze(0)  # Squeeze back to 3D


def visualize_codebook(vqkd_model, data_loader_val, output_dir, max_images_per_code=20, input_size=224):
    """
    Efficiently visualizes codebook. In a distributed setting, all processes gather data,
    then rank 0 saves the images. Stops synchronously across all processes.
    """
    # 1. Initialization (runs on all processes)
    device = next(vqkd_model.parameters()).device
    patch_size = vqkd_model.encoder.patch_embed.patch_size[0]

    if utils.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        print(f"Detected patch size: {patch_size}. Visualizations will be saved to: {output_dir}")

    num_codes = vqkd_model.get_number_of_tokens_for_visualization()
    found_data_for_codes_candidates = {i: [] for i in range(num_codes)}
    used_image_paths = set()

    # 2. Single Pass over the Dataset (runs on all processes in parallel)
    print(f"Rank {utils.get_rank()}: Starting data collection...")
    pbar = tqdm(data_loader_val, desc=f"Rank {utils.get_rank()} Processing", disable=not utils.is_main_process())

    # The main data collection loop
    for images, paths in pbar:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            tokens = vqkd_model.get_codebook_indices(images)

        num_patches_per_dim = int(tokens.shape[1] ** 0.5)
        tokens_grid = tokens.view(images.shape[0], num_patches_per_dim, num_patches_per_dim)

        # This part is the same: find candidates in the current batch
        for i in range(images.shape[0]):
            img_path = paths[i]
            if img_path in used_image_paths:
                continue
            img_tokens = tokens_grid[i]
            unique_codes_in_image = torch.unique(img_tokens)
            potential_codes_for_this_image = []
            for code_tensor in unique_codes_in_image:
                code_idx = code_tensor.item()
                if len(found_data_for_codes_candidates[code_idx]) < max_images_per_code:
                    potential_codes_for_this_image.append(code_idx)
            if potential_codes_for_this_image:
                chosen_code_for_this_image = random.choice(potential_codes_for_this_image)
                locations = (img_tokens == chosen_code_for_this_image).nonzero(as_tuple=False)
                found_data_for_codes_candidates[chosen_code_for_this_image].append({
                    "locations": locations.cpu(), "path": img_path
                })
                used_image_paths.add(img_path)

        # ===== NEW SYNCHRONIZED STOPPING LOGIC =====
        # 1. Count how many codes are satisfied on the CURRENT process
        local_codes_satisfied = sum(
            1 for v in found_data_for_codes_candidates.values() if len(v) >= max_images_per_code)

        # 2. Create a tensor on the current device to hold this count
        stop_tensor = torch.tensor(local_codes_satisfied, dtype=torch.float32, device=device)

        # 3. Use all_reduce to sum the counts from all processes. The result will be on every GPU.
        dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)

        # 4. Check the global condition. We stop if the total number of satisfied codes across all GPUs
        # is at least the number of codes times the number of GPUs. This is a robust heuristic.
        # It means on average, every process has found all codes.
        if stop_tensor.item() >= num_codes * utils.get_world_size():
            if utils.is_main_process():
                print(
                    f"\nGlobal stopping condition met. Total satisfied codes: {stop_tensor.item()}. Stopping all processes.")
            break  # All processes will break here at the same time

        if utils.is_main_process():
            # Update the progress bar with the global satisfaction count
            pbar.set_postfix_str(f"Globally Satisfied Codes: {stop_tensor.item()}/{num_codes * utils.get_world_size()}")

    pbar.close()

    # 3. Synchronize and Gather Data from All Processes (This part remains the same)
    print(f"Rank {utils.get_rank()} finished data collection. Synchronizing all processes...")
    dist.barrier()

    all_candidates_list = [None] * utils.get_world_size()
    dist.gather_object(
        found_data_for_codes_candidates,
        all_candidates_list if utils.is_main_process() else None,
        dst=0
    )

    # 4. Generate and Save Visualization Images (This part remains the same)
    if utils.is_main_process():
        # ... (The merging and saving logic is unchanged)
        print("\nAll data gathered on main process. Merging results...")

        final_candidates = {i: [] for i in range(num_codes)}
        for process_candidates in all_candidates_list:
            if process_candidates is None: continue  # Skip empty lists
            for code_idx, items in process_candidates.items():
                if len(final_candidates[code_idx]) < max_images_per_code:
                    existing_paths = {d['path'] for d in final_candidates[code_idx]}
                    for item in items:
                        if len(final_candidates[code_idx]) < max_images_per_code and item['path'] not in existing_paths:
                            final_candidates[code_idx].append(item)
                            existing_paths.add(item['path'])

        print("Finished merging. Now generating and saving visualization images...")
        # ... (rest of the saving code)
        reload_transform = transforms.Compose([
            transforms.Resize(int(input_size / (224 / 256)), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

        for code_idx, data_list in tqdm(final_candidates.items(), desc="Saving Visualizations"):
            if not data_list:
                continue
            for item in data_list:
                original_path = item["path"]
                locations_yx = item["locations"]
                try:
                    with open(original_path, 'rb') as f:
                        img = Image.open(f).convert('RGB')
                    image_tensor = reload_transform(img)
                except Exception as e:
                    print(f"Warning: Could not load image {original_path}. Skipping. Error: {e}")
                    continue

                original_name_base = os.path.splitext(os.path.basename(original_path))[0]
                output_filename = f"{code_idx:04d}_{original_name_base}.png"
                img_tensor_denorm = denormalize_image(image_tensor)
                img_np = (img_tensor_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                fig, ax = plt.subplots(1, figsize=(6, 6))
                ax.imshow(img_pil)
                ax.set_title(f"Code {code_idx}")
                ax.axis('off')
                for loc in locations_yx:
                    y, x = loc[0].item(), loc[1].item()
                    rect = patches.Rectangle(
                        (x * patch_size, y * patch_size), patch_size, patch_size,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        print("\nVisualization complete. Check the output directory.")

    dist.barrier()

def main():
    args = get_args()

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  # Seed the random module as well
    cudnn.benchmark = True

    dataset_val = build_vqkd_dataset(is_train=False, args=args)

    class DatasetWithPath(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.samples = getattr(dataset, 'samples', None)
            if self.samples is None:
                raise TypeError("The provided dataset must have a 'samples' attribute (like ImageFolder).")

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, _ = self.dataset[index]
            path, _ = self.samples[index]
            return image, path

    dataset_val_with_path = DatasetWithPath(dataset_val)

    if args.dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        # **MODIFICATION HERE**: Set shuffle=True for DistributedSampler to increase diversity
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val_with_path, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        # **MODIFICATION HERE**: Use RandomSampler for non-distributed evaluation to shuffle data
        sampler_val = torch.utils.data.RandomSampler(dataset_val_with_path)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val_with_path, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    vqkd_model = get_visual_tokenizer(args)
    vqkd_model.to(device)

    # UPDATED: Pass the input_size to the function
    visualize_codebook(vqkd_model, data_loader_val, args.output_dir, input_size=args.input_size)


if __name__ == '__main__':
    main()