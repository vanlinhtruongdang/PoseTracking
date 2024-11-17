import gc
import time
import torch
import cv2
from tqdm import tqdm

from helpers.initialize import init_llie_onnx_model, init_pose_model, init_video_writer
from helpers.llie import enhance_images


def process_frame_batch(
    pose_model,
    frames,
    llie_model=None,
    device=None,
    conf_thres=0.25,
):
    enhanced_frames = enhance_images(llie_model, frames) if llie_model else None

    results = pose_model.predict(
        enhanced_frames if enhanced_frames else frames,
        device=device,
        conf=conf_thres,
        save=False,
        save_txt=False,
        verbose=False,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def process_video(
    input_path: str,
    output_path: str,
    model_pose_path: str,
    model_llie_onnx_path: str,
    conf_thres: float = 0.25,
    batch_size: int = 1,
    device: str = "cpu",
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_path}")
    out = init_video_writer(cap, output_path)

    # LLIE_onnx_model = init_llie_onnx_model(model_llie_onnx_path)
    LLIE_onnx_model = None
    YOLO_model = init_pose_model(model_pose_path)

    try:
        frame_times = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        with tqdm(total=total_frames) as pbar:
            frame_count = 0
            while True:
                frame_buffer = []
                start_time = time.time()

                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_buffer.append(frame)

                if not frame_buffer:
                    break

                results = process_frame_batch(
                    YOLO_model,
                    frame_buffer,
                    LLIE_onnx_model,
                    device,
                    conf_thres,
                )

                for result in results:
                    annotated_frame = result.plot()
                    out.write(annotated_frame)
                    del annotated_frame

                    # cv2.imshow("Pose Estimation", annotated_frame)
                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     return

                batch_time = time.time() - start_time
                frame_times.append(batch_time / len(frame_buffer))
                processed_frames += len(frame_buffer)
                pbar.update(len(frame_buffer))

                frame_buffer.clear()

                frame_count += 1
                if frame_count % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during processing: {str(e)}")

    finally:
        cap.release()
        out.release()
        # cv2.destroyAllWindows()

        if frame_times:
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            print("\nProcessing completed:")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Total frames processed: {processed_frames}")
