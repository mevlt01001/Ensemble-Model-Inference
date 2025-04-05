import tensorrt
from utils import get_names, get_image, get_data, plot_image, onnx2network, network2engine, allocate_input_output_buffers, run_inference

names = get_names()

onnx_path = "YOLO12-RTDETR_ensemble_model.onnx"
engine_path = "YOLO12-RTDETR_ensemble_model.engine"

logger = tensorrt.Logger(tensorrt.Logger.VERBOSE)
builder = tensorrt.Builder(logger)

network = onnx2network(logger, builder, onnx_path, True)
engine_cuda = network2engine(network, builder, logger, engine_file_path=engine_path, max_workspace_size_gb=3)

context = engine_cuda.create_execution_context()
inputs, outputs, bindings, stream = allocate_input_output_buffers(engine_cuda)
input_data = get_data(names[0])

output = run_inference(context, bindings, inputs, outputs, stream, input_data)
print(output)
plot_image(get_image(names[0]), output)