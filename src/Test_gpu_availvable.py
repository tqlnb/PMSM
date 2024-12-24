import torch

print(torch.cuda.is_available())   # false

import tensorflow as tf
print("GPU 可用性：", tf.test.is_gpu_available())

print("GPU 可用性：", tf.config.list_physical_devices('GPU'))