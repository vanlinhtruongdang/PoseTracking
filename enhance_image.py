import onnx
import torch
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def prepare_batch_input(image_paths: List[str], size: Tuple[int, int] = (480, 240)) -> np.ndarray:
    """
    Chuẩn bị tensor đầu vào cho một batch ảnh

    Args:
        image_paths (List[str]): Danh sách đường dẫn đến các file ảnh
        size (tuple): Kích thước resize ảnh (width, height)

    Returns:
        np.ndarray: Tensor đầu vào cho cả batch
    """
    batch_tensor = []
    for image_path in image_paths:
        # Đọc và resize ảnh
        image = Image.open(image_path)
        image = image.resize(size)

        # Chuyển thành numpy array và normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Chuyển thành tensor CHW
        img_tensor = img_array.transpose(2, 0, 1)
        batch_tensor.append(img_tensor)

    # Stack thành batch NCHW
    return np.stack(batch_tensor, axis=0)


def save_enhanced_image(args: Tuple[np.ndarray, str, str]) -> str:
    """
    Lưu ảnh đã được enhance

    Args:
        args (tuple): Tuple chứa (image_array, output_path, original_image_path)

    Returns:
        str: Đường dẫn file đã lưu
    """
    image_array, output_path, original_image_path = args

    # Chuyển đổi về dạng ảnh
    output_image = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    enhanced_image = Image.fromarray(output_image)

    # Lưu ảnh với kích thước gốc nếu cần
    if os.path.exists(original_image_path):
        original_size = Image.open(original_image_path).size
        enhanced_image = enhanced_image.resize(original_size)

    enhanced_image.save(output_path)
    return output_path


def enhance_images_with_onnx(model_path: str, image_dir: str, output_dir: str, batch_size: int = 32, max_workers: int = 4) -> None:
    """
    Xử lý tất cả ảnh trong thư mục với model ONNX theo batch

    Args:
        model_path (str): Đường dẫn đến file model ONNX
        image_dir (str): Thư mục chứa ảnh đầu vào
        output_dir (str): Thư mục để lưu ảnh kết quả
        batch_size (int): Kích thước batch cho inference
        max_workers (int): Số lượng thread tối đa cho việc xử lý song song
    """
    # Kiểm tra và tạo thư mục output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load và kiểm tra model ONNX
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # Tạo ONNX Runtime session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(model_path, providers=providers)

    # Lấy tên của input và output nodes
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Xử lý theo batch với progress bar
    with tqdm(total=len(image_files), desc="Enhancing images") as pbar:
        for i in range(0, len(image_files), batch_size):
            # Lấy batch hiện tại
            batch_files = image_files[i : i + batch_size]
            batch_paths = [os.path.join(image_dir, f) for f in batch_files]

            # Chuẩn bị input cho batch
            batch_input = prepare_batch_input(batch_paths)

            # Chạy inference cho batch
            outputs = ort_session.run([output_name], {input_name: batch_input})[0]

            # Chuẩn bị tham số cho việc lưu ảnh song song
            save_args = []
            for idx, output in enumerate(outputs):
                output_filename = batch_files[idx]
                output_path = os.path.join(output_dir, output_filename)
                output_image = output.transpose(1, 2, 0)  # CHW -> HWC
                save_args.append((output_image, output_path, batch_paths[idx]))

            # Lưu các ảnh trong batch song song
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Chạy xử lý song song việc lưu ảnh
                for output_path in executor.map(save_enhanced_image, save_args):
                    pbar.update(1)


if __name__ == "__main__":
    MODEL_LLIE_ONNX_PATH = "ONNX/LYT/LYT.onnx"
    IMAGE_DIR = "dataset/image_downscale"
    OUTPUT_DIR = "dataset/image_enhanced"
    BATCH_SIZE = 32
    MAX_WORKERS = 16

    enhance_images_with_onnx(
        model_path=MODEL_LLIE_ONNX_PATH, image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS
    )
