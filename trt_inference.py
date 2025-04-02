import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from utils import get_data_contiguous

# TensorRT logger oluştur
logger = trt.Logger(trt.Logger.WARNING)

# TensorRT engine yükle
with open('YOLO12-RTDETR_ensemble_model.trt', 'rb') as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

# Context oluştur (inference için gerekli)
context = engine.create_execution_context()

# Input ve output buffer hazırlıkları
input_shape = (1, 3, 640, 640)
input_data = np.random.random(input_shape).astype(np.float32)
input_data = get_data_contiguous("273271-1b86f000bc5b77bf_jpg.rf.83c36538b0d981f8786884c725a25f1f")


# GPU bellek ayır
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(8700 * 4 * 4)  # float32 için 4 byte, bbox için 4 sayı (x,y,w,h)

# CPU→GPU veri aktarımı
cuda.memcpy_htod(d_input, input_data)

# Inference işlemi
context.execute_v2(bindings=[int(d_input), int(d_output)])

# Sonuçları GPU→CPU aktarma

output = np.empty((5, 5), dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)

print(output)
