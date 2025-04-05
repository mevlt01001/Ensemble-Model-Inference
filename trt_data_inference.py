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
mem_log = []
timestamp_log = []
stop_signal = threading.Event()

def measure_gpu_loop(interval=0.1):

    while not stop_signal.is_set():
        watt = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        timestamp = time.time()

        power_log.append(watt)
        mem_log.append(mem)
        timestamp_log.append(timestamp)

        print(f"Power: {watt} W, Memory: {mem} MB, Timestamp: {timestamp-timestamp_log[0]} s")
        stop_signal.wait(interval)

def inference_all_names(context, bindings, inputs, outputs, stream, names):

    preds = []
    truths = []
    inference_latencies = []

    for name in names:
        input_data = get_data(name)

        start = time.process_time()
        output = run_inference(context, bindings, inputs, outputs, stream, input_data)
        inference_latencies.append(time.process_time() - start)

        preds.extend(output.tolist())
        truths.extend(get_boxes(name).tolist())

    return preds, truths, inference_latencies

if __name__ == "__main__":

    LOGGER = tensorrt.Logger(tensorrt.Logger.VERBOSE)
    BUILDER = tensorrt.Builder(LOGGER)
    NETWORK = onnx2network(LOGGER, BUILDER, "YOLO12-RTDETR_ensemble_model.onnx", True)
    ENGINE = network2engine(NETWORK, BUILDER, LOGGER, max_workspace_size_gb=3, engine_file_path="YOLO12-RTDETR_ensemble_model.engine")
    CONTEXT = ENGINE.create_execution_context()
    inputs, outputs, bindings, stream = allocate_input_output_buffers(ENGINE)

    names = get_names()

    measure_thread = threading.Thread(target=measure_gpu_loop, daemon=True)
    measure_thread.start()

    preds, truths, latencies = inference_all_names(CONTEXT, bindings, inputs, outputs, stream, names[:250])

    stop_signal.set()
    measure_thread.join()

    # Sonuçları yazdır (isteğe bağlı)
    print("Average Inference Latency:", sum(latencies) / len(latencies), "s")
    print("Average FPS:", 1 / (sum(latencies) / len(latencies)), "fps")
    print("Average Power Usage:", sum(power_log) / len(power_log), "W")
    print("Max Memory Usage:", max(mem_log), "MB")

    timestamp_log = [_timestamp_log - timestamp_log[0] for _timestamp_log in timestamp_log]
    # watt plotting
    plt.plot(timestamp_log, power_log)
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Power (W)")
    plt.show()
    # mem plotting
    plt.plot(timestamp_log, mem_log)
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Memory (MB)")
    plt.show()
    # fps plotting
    plt.plot(np.arange(len(latencies)), [1 / latency for latency in latencies])
    plt.xlabel("Time (s)")
    plt.ylabel("FPS")
    plt.show()

    

    
