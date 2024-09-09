import time

import onnxruntime as ort
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

img = Image.open("data/source.jpg")

# Get cropped and prewhitened image tensor
mtcnn = MTCNN(image_size=512, margin=10)
img_cropped = mtcnn(img)

# =====================
# INFERENCE VIA ONNX
# =====================
start = time.time()
ort_sess = ort.InferenceSession('facenet.onnx')
outputs = ort_sess.run(None, {'input0': img_cropped.unsqueeze(0).numpy()})
print(f"ONNX Inference took {time.time() - start} seconds") # 0.3332226276397705 seconds
print(f"Onnx Prediction: {sum(outputs[0][0])}") # -0.07012744320309139

# =====================
# INFERENCE VIA PYTORCH
# =====================

start = time.time()
resnet = InceptionResnetV1(pretrained='vggface2').eval()
img_embedding = resnet(img_cropped.unsqueeze(0))
print(f"Pytorch Inference took {time.time() - start} seconds") # 0.33654356002807617 seconds
print(f"Pytorch Prediction: {sum(img_embedding.detach().numpy()[0])}") # -0.07012762896010827


# =====================
# 🚀 FASTEST MODE 🚀
# INFERENCE VIA ONNX WITH BASIC AND EXTENDED OPTIMISATIONS
# =====================
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

start = time.time()
ort_sess = ort.InferenceSession('facenet.onnx', sess_options=sess_options)
outputs = ort_sess.run(None, {'input0': img_cropped.unsqueeze(0).numpy()})
print(f"ONNX B+E Inference took {time.time() - start} seconds") # 0.20347380638122559 seconds
print(f"Onnx Prediction: {sum(outputs[0][0])}") # -0.07012725650929497

# =====================
# 🚀 2ND FASTEST MODE 🚀
# INFERENCE VIA ONNX WITH ALL OPTIMISATIONS
# =====================
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

start = time.time()
ort_sess = ort.InferenceSession('facenet.onnx', sess_options=sess_options)
outputs = ort_sess.run(None, {'input0': img_cropped.unsqueeze(0).numpy()})
print(f"ONNX A+O Inference took {time.time() - start} seconds") # 0.22117352485656738 seconds
print(f"Onnx Prediction: {sum(outputs[0][0])}") # -0.07012744320309139

