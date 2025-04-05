import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import onnxruntime

images_paht = "test-image-folder/images"
labels_path = "test-image-folder/labels"
images = os.listdir(images_paht)
labels = os.listdir(labels_path)

names = [name.rstrip(".jpg") for name in images]

def get_names():
    return names

def get_boxes(name, imgs=640):
    label_txt_path = os.path.join(labels_path, name + ".txt")
    with open(label_txt_path, "r") as f:
        lines = f.read().splitlines()

    boxes = []
    for line in lines:
        _, cx, cy, w, h = line.split(" ")
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        box = [x1, y1, x2, y2]
        boxes.append(box)
    
    return np.array(boxes)*640

def get_all_boxes(imgs=640):
    return np.array([get_boxes(name, imgs) for name in names])


def get_image(name, imgs=640):
    image_path = os.path.join(images_paht, name + ".jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgs, imgs))
    return image

def get_all_images(imgs=640):
    return np.array([get_image(name, imgs) for name in names])

def get_data(name, imgs=640):
    image = get_image(name, imgs)
    data = np.transpose(image, (2, 0, 1))#hwc2chw
    data = np.expand_dims(data, axis=0).astype(np.float32)/255.0
    return data

def get_data_contiguous(name, imgs=640):
    image = get_image(name, imgs)
    data = np.transpose(image, (2, 0, 1))#hwc2chw
    data = np.expand_dims(data, axis=0).astype(np.float32)/255.0
    return (data)

def preprocess(image, imgs=640):
    data = np.transpose(image, (2, 0, 1))#hwc2chw
    data = np.expand_dims(data, axis=0).astype(np.float32)
    return data/255.0

def _get_all_data(imgs=640):
    return np.array([get_data(name, imgs) for name in names])

def get_all_data(imgs=640):
    images = get_all_images(imgs)
    datas = _get_all_data(imgs)
    boxes = get_all_boxes(imgs)

    return images, datas, boxes

def plot_image(image, boxes):
    image = image.copy()
    for box in boxes:
        x1, y1, x2, y2, *_ = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(image)
    plt.show()

def plot_named_image(name, imgs=640):
    image = get_image(name, imgs)
    boxes = get_boxes(name, imgs)

    plt.imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.show()

def onnxruntime_inference(data: np.ndarray, session: onnxruntime.InferenceSession, providers=["CUDAExecutionProvider", "TensorrtExecutionProvider"]):
    return session.run(None, {"image": data}, providers=providers)[0]