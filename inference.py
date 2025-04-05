from utils import get_names, get_boxes, get_data
import onnxruntime
import numpy as np
import pandas as pd

names = get_names()
lenght = len(names)

sesion = onnxruntime.InferenceSession("YOLO12-RTDETR_ensemble_model.onnx", providers=['CUDAExecutionProvider'])

pred_boxes = []
truth_boxes = []

for i,name in enumerate(names):
    if i == 1200:
        break
    print(f"{i}/{lenght}")
    data = get_data(name)
    result = sesion.run(None, {"image": data})[0] 
    pred_boxes.extend(result.tolist())    
    truth_boxes.extend(get_boxes(name).tolist())


pred_boxes = np.array(pred_boxes)
truth_boxes = np.array(truth_boxes)
pd.DataFrame(pred_boxes).reset_index(drop=True).to_csv("pred_boxes.csv")
pd.DataFrame(truth_boxes).reset_index(drop=True).to_csv("truth_boxes.csv")