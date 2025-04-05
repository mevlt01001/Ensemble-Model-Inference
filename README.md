# ONNX Model in TensorRT Inference
This repository is covering to inference with TensorRT Engine.

---

İnference works with an [`tensorrt.ICudaEngine`](https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-tensorrt/api-reference/docs/python/infer/Core/Engine.html#tensorrt.ICudaEngine) object. [`tensorrt.INetworkDefinition`](https://developer.nvidia.com/docs/drive/drive-os/6.0.5/public/drive-os-tensorrt/api-reference/docs/python/infer/Graph/Network.html#tensorrt.INetworkDefinition) objeCt is necessery to create `tensorrt.ICudaEngine` object.

---

[This](utils/tensroRT/utils.py) file is for creating `tensorrt.ICudaEngine`, `tensorrt.ICudaEngine` and some other dependecies.

And [this](utils/utils.py) is for other processing steps.

##### CHECK [TENSORRT INFERENCE](trt_inference.py)