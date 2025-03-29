from utils import get_names, get_boxes, get_image, get_data, plot_image

names = get_names()
boxes = get_boxes(names[0])
image = get_image(names[0])
data = get_data(names[0])

plot_image(image, boxes)
