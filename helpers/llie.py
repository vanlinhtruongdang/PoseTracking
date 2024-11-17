import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

torch.backends.cudnn.benchmark = True


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def enhance_images(model, images):
    enhanced_images = []
    mul = 16

    inputs = torch.stack([TF.to_tensor(img) for img in images]).cuda()

    h, w = inputs.shape[2], inputs.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul

    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0

    inputs = F.pad(inputs, (0, padw, 0, padh), "reflect")

    with torch.no_grad():
        restored_batch = model(inputs)

    restored_batch = restored_batch[:, :, :h, :w]
    restored_batch = restored_batch.permute(0, 2, 3, 1).cpu().detach().numpy()

    for img in restored_batch:
        img = np.ascontiguousarray(img)
        enhanced_images.append(np.clip(img * 255, 0, 255).astype(np.uint8))

    return enhanced_images
