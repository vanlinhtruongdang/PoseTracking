import os
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np


def process_single_image(args):
    """
    Xử lý một ảnh đơn lẻ

    Args:
        args (tuple): (input_path, output_path, target_size)
    """
    input_path, output_path, target_size = args
    try:
        img = cv2.imread(input_path)
        if img is not None:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, resized_img)
            return True
        return False
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_batch(batch, output_dir, target_size, max_workers=None):
    """
    Xử lý một batch ảnh sử dụng thread pool

    Args:
        batch (list): Danh sách các đường dẫn ảnh cần xử lý
        output_dir (str): Thư mục đầu ra
        target_size (tuple): Kích thước ảnh đích
        max_workers (int): Số lượng thread tối đa
    """
    # Chuẩn bị tham số cho mỗi ảnh
    process_args = []
    for input_path in batch:
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        process_args.append((input_path, output_path, target_size))

    # Xử lý song song batch với thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_image, process_args))

    return sum(results)  # Trả về số lượng ảnh xử lý thành công


def downscale_images_batch(input_dir, output_dir, target_size, batch_size=32, max_workers=None):
    """
    Xử lý ảnh theo batch với thread pool

    Args:
        input_dir (str): Thư mục đầu vào
        output_dir (str): Thư mục đầu ra
        target_size (tuple): Kích thước ảnh đích (width, height)
        batch_size (int): Số lượng ảnh xử lý trong một batch
        max_workers (int): Số lượng thread tối đa (None để tự động)
    """
    # Tạo thư mục output nếu chưa tồn tại
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Lấy danh sách các file ảnh
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Chia thành các batch
    total_images = len(image_files)
    batches = [image_files[i : i + batch_size] for i in range(0, total_images, batch_size)]

    # Xử lý từng batch với progress bar
    successful_count = 0
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for batch in batches:
            batch_success = process_batch(batch, output_dir, target_size, max_workers)
            successful_count += batch_success
            pbar.update(len(batch))

    print(f"\nProcessing completed:")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {successful_count}")
    print(f"Failed: {total_images - successful_count}")


# Thực thi script
if __name__ == "__main__":
    input_directory = "dataset/image"
    output_directory = "dataset/image_downscale"
    new_size = (480, 240)  # (width, height)

    # Bạn có thể điều chỉnh các tham số này
    BATCH_SIZE = 32  # Số ảnh xử lý trong một batch
    MAX_WORKERS = None  # Số thread tối đa (None = tự động theo CPU)

    downscale_images_batch(input_directory, output_directory, new_size, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS)
