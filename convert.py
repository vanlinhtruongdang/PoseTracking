import torch
from models.LIT.LITNet import LYT
from models.CID.CIDNet import CIDNet

WIDTH = int(1080 / 3)
HEIGHT = int(1920 / 3)
BATCH_SIZE = 8
DEVICE = torch.device("cpu")

MODEL_LLIE_PATH = "weights/LYT/LOLv1.pth"
MODEL_LLIE_ONNX_PATH = "ONNX/LYT/LYT.onnx"
# MODEL_LLIE_PATH = "weights/CID/CID.pth"
# MODEL_LLIE_ONNX_PATH = "ONNX/CID/CID.onnx"


def pytorch_to_onnx(model_path: str, onnx_path: str):
    try:
        if "CID" in model_path:
            model = CIDNet(
                weight=model_path,
                device=DEVICE,
            )
        elif "LYT" in model_path:
            model = LYT(
                weight=model_path,
                device=DEVICE,
            )

        model.to(DEVICE)
        model.eval()

        input = torch.randn(BATCH_SIZE, 3, WIDTH, HEIGHT, requires_grad=True).to(DEVICE)
        torch.onnx.export(
            model,
            input,
            onnx_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        print(f"Successfully export model to ONNX format: {onnx_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    pytorch_to_onnx(MODEL_LLIE_PATH, MODEL_LLIE_ONNX_PATH)
