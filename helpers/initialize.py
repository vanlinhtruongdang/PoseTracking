from collections import OrderedDict
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import onnx

from models.LLFormer import LLFormer


def init_video_writer(cap, output_path: str):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def init_pose_model(model_path: str) -> YOLO:
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except Exception:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def init_llie_onnx_model(onnx_path: str) -> onnx.ModelProto:
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        return onnx_model

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
