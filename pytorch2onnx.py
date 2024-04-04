import argparse
import torch

from model.model import TrafficSignModel

def conversion(opt):
    """Export the model to ONNX"""

    print("Starting conversion from PyTorch to ONNX ...")
    tensor_shape = torch.randn(1, opt.image_channel, opt.image_size, opt.image_size)
    torch.onnx.export(model,                                             # model being run
                    tensor_shape,                                        # model input (or a tuple for multiple inputs)
                    opt.save_onnx,                                       # where to save the model (can be a file or file-like object)
                    export_params=True,                                  # store the trained parameter weights inside the model file
                    opset_version=16,                                    # the ONNX version to export the model to
                    do_constant_folding=True,                            # whether to execute constant folding for optimization
                    input_names = ['input'],                             # the model's input names
                    output_names = ['output'],                           # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},          # variable length axes
                                'output' : {0 : 'batch_size'}})
    print("Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-size', nargs='+', type=int, default=32, help='Image widh and height size')
    parser.add_argument('--image-channel', nargs='+', type=int, default=3, help='Image channel size: RGB = 3, Grayscale = 1')
    parser.add_argument("--save-onnx", type=str, default="./weights/best.onnx", help="Path to save .onnx file")
    parser.add_argument("--weights", type=str, default='./weights/best.pt', help="Load pretrained pytorch weight file")
    parser.add_argument("--classes", type=int, default=43, help="Number of classes")

    opt = parser.parse_args()

    model = TrafficSignModel(opt.classes)

    # Load the pre-trained PyTorch model
    weight_file = torch.load(opt.weights)
    model.load_state_dict(weight_file)

    # Set the model to evaluation mode
    model.eval()    

    conversion(opt)

