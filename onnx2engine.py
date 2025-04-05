# """
# trtexec --onnx=YOLO12-RTDETR_ensemble_model.onnx \
#         --shapes=image:1x3x640x640 \
#         --fp16 \ 
#         --saveEngine=YOLO12-RTDETR_ensemble_model.trt
#         # 1650TI desteklemiyor hatası aldım.
# """
# # shell ile trt dosyası oluşturuldu.

# ############################################################################################################

import tensorrt
import os
import numpy as np
import pycuda.driver as cuda
#import pycuda.autoinit
from utils import get_names, get_data_contiguous, plot_image, get_image

onnx_path = "YOLO12-RTDETR_ensemble_model.onnx"  # ONNX dosyanın adı

LOGGER = tensorrt.Logger(tensorrt.Logger.INFO)
BUILDER = tensorrt.Builder(LOGGER)

def onnx2network(logger: tensorrt.ILogger, 
                 builder: tensorrt.Builder, 
                 onnx_path: str, 
                 explicit_batch: bool)-> tensorrt.INetworkDefinition:

    assert os.path.exists(onnx_path) and os.path.isfile(onnx_path), f"ONNX file '{onnx_path}' not found"

    flag = (1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) if explicit_batch else 0
    network = builder.create_network(flag)
    parser = tensorrt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as model:
        success = parser.parse(model.read())

    if not success:
        print(f"Failed to parse the ONNX file: {onnx_path}")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError(f"Failed to parse the ONNX file: {onnx_path}")
    
    return network

def network2engine(network: tensorrt.INetworkDefinition,
                   builder: tensorrt.Builder,
                   max_workspace_size_bytes: int = None,
                   max_workspace_size_kb: int = None,
                   max_workspace_size_mb: int = None,
                   max_workspace_size_gb: int = None,
                   engine_file_path: str = None):
    sizes = [max_workspace_size_bytes, max_workspace_size_kb, max_workspace_size_mb, max_workspace_size_gb]
    assert len([item for item in sizes if item is not None]) == 1,\
        "Please provide only one of the following arguments: max_workspace_size_bytes, max_workspace_size_kb, max_workspace_size_mb, max_workspacae_size_gb"

    def get_max_workspace_size():
        sizes = [max_workspace_size_bytes, max_workspace_size_kb, max_workspace_size_mb, max_workspace_size_gb]

        not_none_idx = None         
        for idx, size in enumerate(sizes):
            if size is not None:
                not_none_idx = idx

        max_workspace_size = sizes[not_none_idx]*(1024**not_none_idx)

        return max_workspace_size

    config = builder.create_builder_config()
    config.set_memory_pool_limit(tensorrt.MemoryPoolType.WORKSPACE, get_max_workspace_size())
    
    if builder.platform_has_fast_fp16:
        config.set_flag(tensorrt.BuilderFlag.FP16)
        print("✅ FP16 aktif edildi.")

    profile = builder.create_optimization_profile()

    profile.set_shape(network.get_input(0).name,
                    min=(1,3,640,640),
                    opt=(1,3,640,640),
                    max=(1,3,640,640))

    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Engine serialization has an error")

    if engine_file_path is not None:
        with open(engine_file_path, "wb") as f:
            f.write(engine)
        print(f"Engine diske kaydedildi: {engine_file_path}")
        
    return engine

def load_engine(engine_path: str, logger: tensorrt.ILogger) -> tensorrt.ICudaEngine:
    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError("Engine deserialization failed!")

    print("Engine başarıyla deserialize edildi.")
    return engine

def __iter_engine(engine: tensorrt.ICudaEngine):

    for binding in engine:
        print(f"binding: {binding}")
        #print(f"{idx}: {name}({'input' if engine.binding_is_input(idx) else 'output'})")

def __get_engine_input_output_names_and_shapes(engine: tensorrt.ICudaEngine):

    for binding in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(binding)
        print(f"Tensor: {tensor_name}({engine.get_tensor_shape(tensor_name)})")

class HostDeviceMem:
    def __init__(self, host, device):
        self.host = host          # CPU (host) tarafındaki NumPy array
        self.device = device      # GPU (device) tarafındaki bellek adresi (pycuda.DeviceAllocation)

    def __str__(self):
        return f"Host:\t{self.host}\nDevice:\t{self.device}"

def allocate_input_output_buffers(engine: tensorrt.ICudaEngine) -> tuple[list[HostDeviceMem], list[HostDeviceMem], list[int], cuda.Stream]:

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        print(f"Tensor: {tensor_name}({tensor_shape}) Dtype: {engine.get_tensor_dtype(tensor_name)}")
        size = tensorrt.volume(engine.get_tensor_shape(tensor_name)) if tensor_shape[0] != -1 else tensorrt.volume((15,5))
        dtype = tensorrt.nptype(engine.get_tensor_dtype(tensor_name))        

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype) # page-locked memory buffer (won't swap to disk)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer address to device bindings.
        # When cast to int, it's a linear index into the context's memory (like memory address).
        bindings.append(int(device_mem))

        # Append to the appropriate input/output list.
        if engine.get_tensor_mode(tensor_name) == tensorrt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def run_inference(context, bindings, inputs, outputs, stream, input_data: np.array):
    # 1. Input verisini host belleğe yaz
    np.copyto(inputs[0].host, input_data.ravel())

    # 2. CPU → GPU kopyası
    cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)

    # 3. Inference
    context.execute_v2(bindings)

    # 4. GPU → CPU kopyası
    cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)

    # 5. Tüm işlemler bitsin
    stream.synchronize()

    # 6. Output’u döndür
    return outputs[0].host.reshape(outputs[0].host.shape)


# network = onnx2network(LOGGER, BUILDER, onnx_path, explicit_batch=True)
# engine = network2engine(network, BUILDER, max_workspace_size_gb=2, engine_file_path="demo.engine")
# cuda_engine = load_engine("demo.engine", LOGGER)
#print(cuda_engine.get_tensor_dtype(cuda_engine.get_tensor_name(0)))
#context = cuda_engine.create_execution_context()
#inputs, outputs, bindings, stream = allocate_input_output_buffers(cuda_engine)
# names = get_names()
# input_data = get_data_contiguous(name=names[0], imgs=640)
# input_data = np.random.random((1,3,640,640)).astype(np.float32)
#output = run_inference(context, bindings, inputs, outputs, stream, input_data)





if __name__ == "__main__":
    import time
    import pycuda.autoinit
    cuda_engine = load_engine("demo.engine", LOGGER)
    print(cuda_engine.get_tensor_dtype(cuda_engine.get_tensor_name(0)))
    context = cuda_engine.create_execution_context()
    context.set_input_shape(cuda_engine.get_tensor_name(0), (1, 3, 640, 640))
    
    inputs, outputs, bindings, stream = allocate_input_output_buffers(cuda_engine)

    names = get_names()
    latencies = []

    for i,name in enumerate(names):
        print(f"{i}")
        if i == 120:
            break
        input_data = get_data_contiguous(name=name, imgs=640)
        start = time.time()
        output = run_inference(context, bindings, inputs, outputs, stream, input_data).reshape(-1, 5)
        latencies.append(time.time() - start)


    print(f"fps: {1/np.mean(latencies)}")
    # input_data = get_data_contiguous(name=names[0], imgs=640)

    # output = run_inference(context, bindings, inputs, outputs, stream, input_data).reshape(-1, 5)
    # print(output)
    # plot_image(get_image(names[0], 640), output)