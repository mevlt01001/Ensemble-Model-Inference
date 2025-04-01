"""
trtexec --onnx=YOLO12-RTDETR_ensemble_model.onnx \
        --shapes=image:1x3x640x640 \
        --fp16 \ 
        --saveEngine=YOLO12-RTDETR_ensemble_model.trt
        # 1650TI desteklemiyor hatası aldım.
"""
# shell ile trt dosyası oluşturuldu.