import time
import pynvml
import tensorrt
import threading
import numpy as np
import matplotlib.pyplot as plt
from utils import *


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

power_log = []
memory_log = []
timestamp_log = []
stop_signal = threading.Event()

def measure_gpu_loop(interval=0.5):

    while not stop_signal.is_set():
        watt = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        timestamp = time.time()

        power_log.append(watt)
        memory_log.append(mem)
        timestamp_log.append(timestamp)

        print(f"Power: {watt:.2f} W, Memory: {mem:.2f} MB, Timestamp: {(timestamp-timestamp_log[0]):.2f} s")
        stop_signal.wait(interval)

def inference_all_names(context, bindings, inputs, outputs, stream, names):

    preds = []
    truths = []
    inference_latencies = []
    lenght = len(names)

    for i,name in enumerate(names):
        print(f"{i+1}/{lenght}: {name}")
        input_data = get_data(name)

        start = time.process_time()
        output = run_inference(context, bindings, inputs, outputs, stream, input_data)
        inference_latencies.append(time.process_time() - start)

        preds.extend(output.tolist())
        truths.extend(get_boxes(name).tolist())

    return preds, truths, inference_latencies

if __name__ == "__main__":

    LOGGER = tensorrt.Logger(tensorrt.Logger.INFO)
    BUILDER = tensorrt.Builder(LOGGER)
    NETWORK = onnx2network(LOGGER, BUILDER, "YOLO12-RTDETR_ensemble_model.onnx", True)
    ENGINE = network2engine(NETWORK, BUILDER, LOGGER, max_workspace_size_gb=3, engine_file_path="YOLO12-RTDETR_ensemble_model.engine")
    CONTEXT = ENGINE.create_execution_context()
    inputs, outputs, bindings, stream = allocate_input_output_buffers(ENGINE)

    names = get_names()

    measure_thread = threading.Thread(target=measure_gpu_loop, daemon=True)
    measure_thread.start()

    preds, truths, latencies = inference_all_names(CONTEXT, bindings, inputs, outputs, stream, names[0:-1:4])

    stop_signal.set()
    measure_thread.join()
    timestamp_log = [_timestamp_log - timestamp_log[0] for _timestamp_log in timestamp_log]

    np.save("inference_data/preds.npy", preds)
    np.save("inference_data/truths.npy", truths)
    np.save("inference_data/latencies.npy", latencies)
    np.save("inference_data/power_log.npy", power_log)
    np.save("inference_data/mem_log.npy", memory_log)
    np.save("inference_data/timestamp_log.npy", timestamp_log)



    

    
