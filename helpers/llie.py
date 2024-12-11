import torch
import torchvision.transforms.functional as TF
import onnxruntime as ort
import numpy as np
import cv2

ort.set_default_logger_severity(3)


def to_numpy(tensor):
    """Chuyển tensor sang numpy array"""
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def enhance_images(model, images):
    images_numpy = np.stack([np.array(img).astype(np.float32) / 255.0 for img in images])

    inputs = torch.from_numpy(images_numpy).permute(0, 3, 1, 2).cuda()

    ort_session = ort.InferenceSession(
        model,
        sess_options=ort.SessionOptions(),
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": torch.cuda.current_device(),
                    "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                },
            ),
        ],
    )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    ort_outs = np.transpose(ort_outs, (0, 2, 3, 1))

    enhanced_images = []
    for img in ort_outs:
        img = np.ascontiguousarray(img)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        enhanced_images.append(img)

    return enhanced_images
