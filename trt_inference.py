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

    onnx_names = [name.rstrip(".onnx") for name in os.listdir("OnnxFolder")]

    measure_thread = threading.Thread(target=measure_gpu_loop, daemon=True)
    measure_thread.start()

    for name in onnx_names:

        onnx_path = os.path.join("OnnxFolder", name+".onnx")
        engine_path = os.path.join("EngineFolder", name+".engine")

        LOGGER = tensorrt.Logger(tensorrt.Logger.INFO)
        BUILDER = tensorrt.Builder(LOGGER)
        NETWORK = onnx2network(LOGGER, BUILDER, onnx_path, True)
        ENGINE = network2engine(NETWORK, BUILDER, LOGGER, max_workspace_size_gb=3, engine_file_path=engine_path)
        CONTEXT = ENGINE.create_execution_context()
        inputs, outputs, bindings, stream = allocate_input_output_buffers(ENGINE)

        names = get_names()

        preds, truths, latencies = inference_all_names(CONTEXT, bindings, inputs, outputs, stream, names[0:-1:8])
        timestamp_log = [_timestamp_log - timestamp_log[0] for _timestamp_log in timestamp_log]
        
        os.makedirs(f"inference_data/{name}", exist_ok=True)
        np.save(f"inference_data/{name}/preds.npy", preds)
        np.save(f"inference_data/{name}/truths.npy", truths)
        np.save(f"inference_data/{name}/latencies.npy", latencies)
        np.save(f"inference_data/{name}/power_log.npy", power_log)
        np.save(f"inference_data/{name}/mem_log.npy", memory_log)
        np.save(f"inference_data/{name}/timestamp_log.npy", timestamp_log)
        

    stop_signal.set()
    measure_thread.join()