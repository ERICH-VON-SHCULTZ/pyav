#!/usr/bin/env python3
"""
video_dinov2_embeddings.py

Create DINOv2 embeddings frame-by-frame for each input video,
and save one .npy array per video (shape: num_frames x embedding_dim).
Prints filename, frame rate, number of frames processed, and processing time.
"""

import os
import cv2
import torch
import numpy as np
import argparse
import time
from torchvision import transforms

# ------------------------------
# Map short aliases -> full model names
# ------------------------------
MODEL_ALIASES = {
    "small": "dinov2_vits14",
    "base": "dinov2_vitb14",
    "medium": "dinov2_vitb14",   # alias for base
    "large": "dinov2_vitl14",
    "giant": "dinov2_vitg14",
}

# ------------------------------
# Preprocessing transform
# ------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # required input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------
# Function: load model
# ------------------------------
def load_model(alias: str):
    if alias in MODEL_ALIASES:
        model_name = MODEL_ALIASES[alias]
    else:
        model_name = alias  # allow full torch.hub name too

    print(f"Loading DINOv2 model: {model_name}")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# ------------------------------
# Function: video -> embeddings
# ------------------------------
def video_to_embeddings(video_path, model, transform, device):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    embeddings = []
    frame_count = 0

    start_time = time.time()  # start timing

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(frame_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tensor)

        embeddings.append(emb.cpu().numpy().squeeze())

    cap.release()
    end_time = time.time()
    elapsed = end_time - start_time

    if not embeddings:
        raise ValueError(f"No frames read from {video_path}")

    print(f"Processed video: {video_path}")
    print(f"  Frame rate: {fps:.2f} fps")
    print(f"  Frames processed: {frame_count}/{total_frames}")
    print(f"  Processing time: {elapsed:.2f} seconds")

    return np.vstack(embeddings)

# ------------------------------
# Process multiple videos
# ------------------------------
def process_videos(video_files, model, transform, device, output_dir="embeddings"):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for video in video_files:
        arr = video_to_embeddings(video, model, transform, device)
        out_name = os.path.splitext(os.path.basename(video))[0] + "_embeddings.npy"
        out_path = os.path.join(output_dir, out_name)

        np.save(out_path, arr)
        results[video] = out_path
        print(f" -> Saved embeddings {arr.shape} to {out_path}\n")

    return results

# ------------------------------
# Inspect saved embeddings
# ------------------------------
def inspect_embeddings(path, head=5):
    arr = np.load(path)
    print(f"✅ Loaded: {path}")
    print(f"   Shape: {arr.shape}")
    print(f"   Dtype: {arr.dtype}")
    print(f"   Type:  {type(arr)}")
    if head > 0:
        print(f"   First {head} rows:\n{arr[:head]}")

# ------------------------------
# Test harness
# ------------------------------
def run_test(model, transform, device):
    test_video = "test_video.mp4"

    print("🧪 Creating synthetic test video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(test_video, fourcc, 5.0, (64, 64))

    for i in range(10):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i*2 % 64, i*2 % 64),
                      (i*2 % 64 + 20, i*2 % 64 + 20),
                      (0, 255, 0), -1)
        out.write(frame)
    out.release()

    print("🧪 Running embedding extraction...")
    arr = video_to_embeddings(test_video, model, transform, device)
    print(f"✅ Test video embeddings shape: {arr.shape}")

    os.remove(test_video)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings from videos")
    parser.add_argument("videos", help="List of video files")
    parser.add_argument("--outdir", default="embeddings", help="Output directory")
    parser.add_argument("--model", default="small",
                        help="Model size: small | base | large | giant "
                             "(or full torch.hub name like dinov2_vitl14)")
    parser.add_argument("--inspect", metavar="FILE", help="Inspect a saved .npy embeddings file")
    parser.add_argument("--head", type=int, default=5, help="How many rows to show when inspecting")
    parser.add_argument("--test", action="store_true", help="Run test harness")
    args = parser.parse_args()
    
    # Handle inspect mode first
    if args.inspect:
        inspect_embeddings(args.inspect, head=args.head)
    else:
        model, device = load_model(args.model)

        if args.test:
            run_test(model, transform, device)
        elif args.videos:
            with open(args.videos, 'r') as f:
                video_list = [line.strip() for line in f]
            process_videos(video_list, model, transform, device, output_dir=args.outdir)
        else:
            parser.print_help()
