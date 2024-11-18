import torch
from convert import MODEL_LLIE_ONNX_PATH
from helpers.process import process_video


if __name__ == "__main__":
    INPUT_VIDEO = "stock.mp4"
    OUTPUT_VIDEO = "stock_tracked.mp4"
    MODEL_POSE_PATH = "weights/YOLO/yolo11n-pose.pt"
    CONFIDENCE_THRESHOLD = 0.25
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        MODEL_POSE_PATH,
        MODEL_LLIE_ONNX_PATH,
        CONFIDENCE_THRESHOLD,
        BATCH_SIZE,
        DEVICE,
    )
