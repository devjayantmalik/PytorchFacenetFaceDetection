import onnx
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

resnet = InceptionResnetV1(pretrained='vggface2').eval()

torch_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(resnet, torch_input,
                  'facenet.onnx',
                  opset_version=17,  # The ONNX version to export the models to
                  do_constant_folding=True,  # Whether to execute constant folding for optimization
                  input_names=['input0'],  # The models's input names
                  output_names=['output0']  # The models's output names
                  )

# Load the ONNX models
onnx_model = onnx.load('facenet.onnx')

# Check that the models is well-formed
onnx.checker.check_model(onnx_model)
print("ONNX models is well-formed and saved successfully.")


# convert models to fp16
# This didn't work for us.
# from onnxconverter_common import float16

# model_fp16 = float16.convert_float_to_float16(onnx_model)
# onnx.save(model_fp16, "facenet_fp16.onnx")