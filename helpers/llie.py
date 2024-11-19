import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def enhance_images(model, images, denoise_strength=3, contrast_alpha=1.3, contrast_beta=3):
    enhanced_images = []

    inputs = torch.stack([TF.to_tensor(img) for img in images]).cuda()

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

    for img in ort_outs:
        img = np.ascontiguousarray(img)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        img = cv2.bilateralFilter(img, d=denoise_strength, sigmaColor=35, sigmaSpace=35)
        if len(img.shape) == 3:
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.convertScaleAbs(img, alpha=contrast_alpha, beta=contrast_beta)
        img = gaussian_filter(img, sigma=1.0)
        img = np.clip(img, 0, 255).astype(np.uint8)

        enhanced_images.append(img)

    return enhanced_images
