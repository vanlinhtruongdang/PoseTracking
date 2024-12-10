import os
import cv2
import numpy as np
from pathlib import Path
import re


def extract_bbox_count(filename):
    # Lấy số lượng bbox từ tên file
    try:
        return int(filename.split("_(")[1].split(").")[0])
    except:
        return 0


def combine_images(img1_path, img2_path, output_path):
    # Đọc hai ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Đảm bảo cả hai ảnh có cùng kích thước
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    # Resize ảnh nếu cần
    img1 = cv2.resize(img1, (width, height))
    img2 = cv2.resize(img2, (width, height))

    # Ghép ngang hai ảnh
    combined = np.hstack((img1, img2))

    # Lưu ảnh kết quả
    cv2.imwrite(str(output_path), combined)


def main():
    # Tạo đường dẫn
    result_dir = Path("result")
    original_dir = result_dir / "original"
    enhanced_dir = result_dir / "enhanced"
    compare_dir = result_dir / "compare"

    # Tạo thư mục compare nếu chưa tồn tại
    compare_dir.mkdir(exist_ok=True)

    # Lấy danh sách các file trong thư mục original
    original_files = sorted(original_dir.glob("*"))
    enhanced_files = sorted(enhanced_dir.glob("*"))

    for idx in range(0, len(original_files)):
        original_file_path = str(original_files[idx])
        enhanced_file_path = str(enhanced_files[idx])

        original_file_name = original_file_path.split("_")[0].split("/")[-1]
        enhanced_file_name = original_file_path.split("_")[0].split("/")[-1]

        original_num_bboxes = re.search(r"(?<=\().*?(?=\))", original_file_path).group()
        enhanced_num_bboxes = re.search(r"(?<=\().*?(?=\))", enhanced_file_path).group()

        if (original_file_name == enhanced_file_name) and (original_num_bboxes != enhanced_num_bboxes):
            # Tạo tên file output
            output_name = f"compare_{original_file_name}_{original_num_bboxes}vs{enhanced_num_bboxes}.jpg"
            output_path = compare_dir / output_name

            # Ghép và lưu ảnh
            combine_images(original_file_path, enhanced_file_path, output_path)
            print(f"Đã lưu file so sánh: {output_name}")


if __name__ == "__main__":
    main()
