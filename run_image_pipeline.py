#!/usr/bin/env python3
import os
import torch
import argparse
from PIL import Image
import numpy as np
import cv2
import time
import datetime
from rembg import remove
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, ControlNetModel
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector, HEDdetector, MLSDdetector, NormalBaeDetector
from segment_anything import sam_model_registry, SamPredictor

# Set onnxruntime thread count to avoid thread affinity errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"

# Global settings
DEVICE = "cuda:0"
RES = 1024  # Maximum resolution for preprocessing

def sam_init(sam_checkpoint_path):
    """Initialize the SAM model for segmentation."""
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=DEVICE)
    predictor = SamPredictor(sam)
    return predictor

def sam_segment(predictor, input_image, *bbox_coords):
    """Segment the image using SAM."""
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)

    start_time = time.time()
    predictor.set_image(image)

    masks_bbox, _, _ = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')

def expand2square(pil_img, background_color):
    """Expand the image to a square with padding."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def preprocess(predictor, input_image, segment=True, rescale=False):
    """Preprocess the input image for the pipeline."""
    # Ensure Image.Resampling is available
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image

    # Resize image to fit within RES x RES
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    
    # Background removal if requested
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:,:,-1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        
        if len(x_nonzero[0]) > 0 and len(y_nonzero[0]) > 0:
            x_min = int(x_nonzero[0].min())
            y_min = int(y_nonzero[0].min())
            x_max = int(x_nonzero[0].max())
            y_max = int(y_nonzero[0].max())
            input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    
    # Rescale and recenter
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len//2
        padded_image[center-h//2:center-h//2+h, center-w//2:center-w//2+w] = image_arr[y:y+h, x:x+w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)

        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[...,:3] * rgba_arr[...,-1:] + (1 - rgba_arr[...,-1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    
    # Return both the high-res preprocessed image and a 320x320 version for the model
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)

def generate_control_image(control_type, input_image, resolution=512):
    """Generate control image based on the specified type."""
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image
        
    # Resize input image for control net
    input_np = np.array(input_image.resize((resolution, resolution), Image.Resampling.LANCZOS))
    
    if control_type == "canny":
        detector = CannyDetector()
        control_image = detector(input_np, low_threshold=100, high_threshold=200)
        return Image.fromarray(control_image)
    elif control_type == "depth":
        detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        control_image = detector(input_np)
        return control_image
    elif control_type == "openpose":
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        control_image = detector(input_np)
        return control_image
    elif control_type == "mlsd":
        detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        control_image = detector(input_np)
        return control_image
    elif control_type == "normal":
        detector = NormalBaeDetector.from_pretrained("lllyasviel/ControlNet")
        control_image = detector(input_np)
        return control_image
    elif control_type == "hed":
        detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")
        control_image = detector(input_np)
        return control_image
    else:
        raise ValueError(f"Unsupported control type: {control_type}")

def run_pipeline(input_path, output_dir, guidance_scale=4, steps=75, seed=42, 
               segment=True, rescale=False, model_path=None, control_type=None, 
               control_image_path=None, control_scale=1.0):
    """Run the zero123plus pipeline on an input image and save the output views."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    seed = int(seed)
    torch.manual_seed(seed)
    
    # Load the image
    input_image = Image.open(input_path).convert("RGB")
    print(f"Loaded image: {input_path}")
    
    # Initialize SAM for segmentation
    sam_checkpoint = os.path.join("tmp", "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_checkpoint):
        print(f"SAM checkpoint not found at {sam_checkpoint}. Please download it first.")
        print("You can download it from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return
    
    predictor = sam_init(sam_checkpoint)
    
    # Preprocess the image
    print("Preprocessing image...")
    processed_image, _ = preprocess(predictor, input_image, segment=segment, rescale=rescale)
    processed_image_path = os.path.join(output_dir, "processed_input.png")
    processed_image.save(processed_image_path)
    print(f"Saved preprocessed image to {processed_image_path}")
    
    # Handle ControlNet if specified
    controlnet = None
    control_image = None
    
    if control_type or control_image_path:
        print("Preparing ControlNet...")
        # Load the controlnet model
        controlnet = ControlNetModel.from_pretrained(
            f"lllyasviel/sd-controlnet-{control_type}", torch_dtype=torch.float16
        ).to(DEVICE)
        
        # Generate or load control image
        if control_image_path:
            print(f"Loading control image from {control_image_path}")
            control_image = Image.open(control_image_path).convert("RGB")
        elif control_type:
            print(f"Generating {control_type} control image")
            control_image = generate_control_image(control_type, input_image)
            # Save the control image
            control_image_path = os.path.join(output_dir, f"control_{control_type}.png")
            control_image.save(control_image_path)
            print(f"Saved control image to {control_image_path}")
    
    # Load the pipeline
    print("Loading pipeline...")
    if model_path:
        if controlnet:
            print("Loading pipeline with ControlNet")
            pipeline = DiffusionPipeline.from_pretrained(
                model_path, 
                controlnet=controlnet,
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                model_path, 
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16
            )
    else:
        if controlnet:
            print("Loading pipeline with ControlNet")
            pipeline = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2",
                controlnet=controlnet,
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16
            )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2", 
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16
            )
    
    # Configure scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to(DEVICE)
    
    # Run the pipeline
    print(f"Running pipeline with {steps} steps and guidance scale {guidance_scale}...")
    
    # Create pipeline arguments
    pipeline_args = {
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": torch.Generator(pipeline.device).manual_seed(seed)
    }
    
    # Add controlnet parameters if applicable
    if controlnet and control_image:
        pipeline_args["control_image"] = control_image
        pipeline_args["controlnet_conditioning_scale"] = control_scale
    
    # Run the pipeline
    output = pipeline(processed_image, **pipeline_args).images[0]
    
    # Extract and save the 6 views
    side_len = output.width // 2
    views = [output.crop((x, y, x + side_len, y + side_len)) 
             for y in range(0, output.height, side_len) 
             for x in range(0, output.width, side_len)]
    
    # Save each view
    for i, view in enumerate(views):
        view_path = os.path.join(output_dir, f"view_{i+1}.png")
        view.save(view_path)
        print(f"Saved view {i+1} to {view_path}")
    
    # Also save the full output grid
    output_path = os.path.join(output_dir, "output_grid.png")
    output.save(output_path)
    print(f"Saved output grid to {output_path}")
    
    print("Done! All views have been saved to the output directory.")

if __name__ == "__main__":
    # Get current date and time for default output directory
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output = os.path.join("output", current_time)
    
    parser = argparse.ArgumentParser(description="Run Zero123Plus pipeline on an input image")
    parser.add_argument("--input", "-i", required=True, help="Path to the input image")
    parser.add_argument("--output", "-o", default=default_output, help="Directory to save output images (default: output/date,time)")
    parser.add_argument("--guidance_scale", "-g", type=float, default=4.0, help="Classifier free guidance scale")
    parser.add_argument("--steps", "-s", type=int, default=75, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_segment", action="store_true", help="Skip background removal")
    parser.add_argument("--rescale", action="store_true", help="Rescale and recenter the image")
    parser.add_argument("--model_path", help="Path to local model weights (optional)")
    # ControlNet arguments
    parser.add_argument("--control_type", choices=["canny", "depth", "openpose", "mlsd", "normal", "hed"], 
                         help="Type of ControlNet to use")
    parser.add_argument("--control_image", help="Path to a custom control image (if not generating)")
    parser.add_argument("--control_scale", type=float, default=1.0, help="ControlNet conditioning scale")
    
    args = parser.parse_args()
    
    run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        segment=not args.no_segment,
        rescale=args.rescale,
        model_path=args.model_path,
        control_type=args.control_type,
        control_image_path=args.control_image,
        control_scale=args.control_scale
    ) 