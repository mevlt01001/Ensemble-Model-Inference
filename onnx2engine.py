# """
# trtexec --onnx=YOLO12-RTDETR_ensemble_model.onnx \
#         --shapes=image:1x3x640x640 \
#         --fp16 \ 
#         --saveEngine=YOLO12-RTDETR_ensemble_model.trt
#         # 1650TI desteklemiyor hatası aldım.
# """
# # shell ile trt dosyası oluşturuldu.

# ############################################################################################################

# import tensorrt as trt
# import os

# onnx_path = "YOLO12-RTDETR_ensemble_model.onnx"  # ONNX dosyanın adı

# def onnx_parser(network, onnx_path):
#     parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
#     with open(onnx_path, "rb") as model:
#         parser.parse(model.read())

# def create_network(builder, explicit_batch=True, imlp_explicit_batch=False):
#     flag = int(explicit_batch) | int(imlp_explicit_batch)
#     if flag != 2 and flag != 1:
#         raise Exception("JUST SELECT EXPLICIT_BATCH OR IMPLICIT_BATCH. NOT BOTH!")
        
#     network = builder.create_network(flag)

# def onnx2engine(onnx_path):
#     # 1. Logger oluştur (INFO seviyesi önerilir)
#     # 2. Builder oluştur
#     # 3. Network tanımı (explicit batch ile)
#     # 4. ONNX parser oluştur
#     # 5. ONNX dosyasını oku ve parse et
#     # 6. Model oluştur
#     # 7. Engine oluştur
#     # 8. Engine dosyasını kaydet
#     LOGGER = trt.Logger(trt.Logger.INFO)
#     BUILDER = trt.Builder(LOGGER)
#     NETWORK = create_network()