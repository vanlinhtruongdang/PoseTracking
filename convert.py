from collections import OrderedDict

import torch

from models.LLFormer import LLFormer

WIDTH = 640
HEIGHT = 360
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_LLIE_PATH = "weights/LLFormer/model_bestPSNR.pth"
MODEL_LLIE_ONNX_PATH = "ONNX/LLFormer/LLFormer.onnx"


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except Exception:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def convert_to_onnx(model_path: str, onnx_path: str):
    try:
        model = LLFormer(
            inp_channels=3,
            out_channels=3,
            dim=16,
            num_blocks=[2, 4, 8, 16],
            num_refinement_blocks=2,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            attention=True,
            skip=False,
        )
        model.to(DEVICE)
        load_checkpoint(model, model_path)
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

        print(f"Successfully export LLFormer to ONNX format: {onnx_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    convert_to_onnx(MODEL_LLIE_PATH, MODEL_LLIE_ONNX_PATH)
