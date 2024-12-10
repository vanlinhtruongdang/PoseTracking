from ultralytics import YOLO
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def plot_bbox(args: Tuple[object, str, str]) -> None:
    """
    Xử lý một ảnh đơn lẻ và lưu kết quả

    Args:
        args (tuple): Tuple chứa (result, img_name, output_dir)
    """
    result, img_name, output_dir = args
    num_bbox = len(result.boxes)

    # Vẽ bounding box
    bbox_img = result.plot()

    # Chuyển đổi sang định dạng PIL Image
    im = Image.fromarray(bbox_img)

    # Tạo tên file đầu ra
    output_name = os.path.splitext(img_name)[0] + f"_({num_bbox}).png"
    output_path = os.path.join(output_dir, output_name)

    # Lưu ảnh
    im.save(output_path)
    return output_path


def process_images_in_batches(model_path: str, image_dir: str, output_dir: str, batch_size: int = 4, max_workers: int = 4) -> None:
    """
    Xử lý ảnh theo batch từ thư mục đầu vào và lưu kết quả vào thư mục đầu ra

    Args:
        model_path (str): Đường dẫn đến file model
        image_dir (str): Thư mục chứa ảnh đầu vào
        output_dir (str): Thư mục để lưu ảnh kết quả
        batch_size (int): Kích thước batch
        max_workers (int): Số lượng thread tối đa cho việc xử lý song song
    """
    # Khởi tạo model
    model = YOLO(model_path)

    # Tạo thư mục output nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Lấy danh sách tất cả các file ảnh
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Tạo thanh progress bar cho các batch
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        # Xử lý theo batch
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i : i + batch_size]

            # Tạo danh sách đường dẫn đầy đủ cho batch hiện tại
            batch_paths = [os.path.join(image_dir, img) for img in batch]

            # Dự đoán cho cả batch
            results = model.predict(
                batch_paths,
                save=False,
                save_txt=False,
                verbose=False,
            )

            # Chuẩn bị tham số cho việc xử lý song song
            process_args = [(result, img_name, output_dir) for result, img_name in zip(results, batch)]

            # Xử lý song song việc lưu ảnh trong batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Chạy các task và thu thập kết quả
                saved_paths = list(executor.map(plot_bbox, process_args))

                # Cập nhật progress bar
                pbar.update(len(saved_paths))


# Sử dụng hàm
if __name__ == "__main__":
    MODEL_POSE_PATH = "weights/YOLO/yolov11n-pose.pt"
    IMAGE_DIR = "dataset/image_enhanced"
    OUTPUT_DIR = "result/enhanced"
    BATCH_SIZE = 32
    MAX_WORKERS = 16

    process_images_in_batches(model_path=MODEL_POSE_PATH, image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
