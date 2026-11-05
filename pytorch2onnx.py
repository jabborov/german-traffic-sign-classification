import argparse
import pathlib

import numpy as np
import torch

from model.model import TrafficSignModel


def conversion(model, opt):
    """Export the model to ONNX."""
    print("Starting conversion from PyTorch to ONNX ...")
    pathlib.Path(opt.save_onnx).parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, opt.image_channel, opt.image_size, opt.image_size)
    torch.onnx.export(model,                                     # model being run
                      dummy_input,                               # example model input
                      opt.save_onnx,                             # where to save the model
                      export_params=True,                        # store the trained weights in the file
                      opset_version=16,                          # ONNX opset version
                      do_constant_folding=True,                  # constant folding for optimization
                      input_names=["input"],                     # model input name (used by inference.cc)
                      output_names=["output"],                   # model output name (used by inference.cc)
                      dynamic_axes={"input": {0: "batch_size"},  # variable batch size
                                    "output": {0: "batch_size"}})
    print(f"Saved {opt.save_onnx}")


def verify(model, opt, batch_size: int = 4):
    """Check the ONNX graph and compare ONNX Runtime outputs with PyTorch."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(opt.save_onnx))

    x = torch.randn(batch_size, opt.image_channel, opt.image_size, opt.image_size)
    with torch.no_grad():
        expected = model(x).numpy()

    session = ort.InferenceSession(opt.save_onnx, providers=["CPUExecutionProvider"])
    actual = session.run(["output"], {"input": x.numpy()})[0]

    max_diff = float(np.abs(expected - actual).max())
    np.testing.assert_allclose(expected, actual, rtol=1e-3, atol=1e-4)
    print(f"Verification passed (batch={batch_size}, max abs diff={max_diff:.2e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=32, help="Image width and height")
    parser.add_argument("--image-channel", type=int, default=3, help="Image channels: RGB = 3, Grayscale = 1")
    parser.add_argument("--save-onnx", type=str, default="./weights/best.onnx", help="Path to save .onnx file")
    parser.add_argument("--weights", type=str, default="./weights/best.pt", help="Pretrained PyTorch weight file")
    parser.add_argument("--classes", type=int, default=43, help="Number of classes")
    parser.add_argument("--no-verify", action="store_true", help="Skip the ONNX Runtime output check")
    opt = parser.parse_args()

    model = TrafficSignModel(opt.classes)
    model.load_state_dict(torch.load(opt.weights, map_location="cpu", weights_only=True))
    model.eval()

    conversion(model, opt)
    if not opt.no_verify:
        verify(model, opt)
