from utils import get_names, get_image, get_data, plot_image, create_onnx_session, onnxruntime_inference

names = get_names()
onnx_path = "YOLO12-RTDETR_ensemble_model.onnx"
input_data = get_data(names[0])
providers = ['CUDAExecutionProvider']
session = create_onnx_session(onnx_path, providers)
output = onnxruntime_inference(input_data, session)

print(output)
plot_image(get_image(names[0]), output)