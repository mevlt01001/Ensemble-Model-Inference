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
from utils import get_names, get_data_contiguous

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
        print(f"💾 Engine diske kaydedildi: {engine_file_path}")
        
    return engine

def load_engine(engine_path: str, logger: tensorrt.ILogger) -> tensorrt.ICudaEngine:
    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = tensorrt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError("❌ Engine deserialization failed!")

    print("✅ Engine başarıyla deserialize edildi.")
    return engine


# network = onnx2network(LOGGER, BUILDER, onnx_path, explicit_batch=True)
# engine = network2engine(network, BUILDER, max_workspace_size_gb=2, engine_file_path="demo.engine")
cuda_engine = load_engine("demo.engine", LOGGER)
context = cuda_engine.create_execution_context()





