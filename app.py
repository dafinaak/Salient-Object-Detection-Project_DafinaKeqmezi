"""
app.py
======
Gradio demo for the trained Salient Object Detection model.

The user uploads any image; the app:
    1. Resizes to 128x128.
    2. Runs the improved model.
    3. Displays the predicted saliency mask, an overlay, and the inference time.

Usage:
    python app.py --ckpt path/to/best_improved.pth
"""

import argparse
import time

import cv2
import gradio as gr
import numpy as np
import torch

from sod_model import SODNetImproved


def build_predict_fn(ckpt_path):
    """Load the model once and return a closure suitable for Gradio."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SODNetImproved().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded model from {ckpt_path} (epoch {state['epoch']})")
    print(f"Device: {device}")

    def predict(input_image):
        if input_image is None:
            return None, None, "Please upload an image."

        # ----- preprocess -----
        img_resized = cv2.resize(input_image, (128, 128))
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # ----- inference (timed) -----
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(img_tensor)
        inference_ms = (time.perf_counter() - t0) * 1000

        # ----- postprocess -----
        mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

        # red overlay
        overlay = img_resized.copy()
        red = np.zeros_like(img_resized)
        red[..., 0] = 255
        overlay = np.where(
            mask[..., None] > 0,
            (0.5 * overlay + 0.5 * red).astype(np.uint8),
            overlay,
        )

        info = (f"Inference: {inference_ms:.1f} ms  |  "
                f"Image size: 128x128  |  Device: {device}")
        return mask, overlay, info

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to best_improved.pth")
    parser.add_argument("--share", action="store_true",
                        help="Generate a public Gradio share link")
    args = parser.parse_args()

    predict_fn = build_predict_fn(args.ckpt)

    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="numpy", label="Input Image"),
        outputs=[
            gr.Image(label="Predicted Saliency Mask"),
            gr.Image(label="Overlay (red = salient region)"),
            gr.Textbox(label="Info"),
        ],
        title="Salient Object Detection - ECSSD",
        description=("Upload an image. The CNN predicts which pixels belong "
                     "to the most salient object."),
        allow_flagging="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
