import cv2
import torch
import base64
import numpy as np
from PIL import Image
import io
import logging
from datetime import datetime

def log_msg(msg):
    logging.info("{}: {}".format(datetime.now(),msg))

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def save_crop(image_object, cords, gain=1.02, pad=10):
    log_msg('Extracting Signature')
    # img = Image.open(image_object, 'rb')
     # Convert the image object to a NumPy array
    img = np.asarray(image_object)
    # img = cv2.imread(filepath)
        # Save an image crop as {file} with crop size multiplied by {gain} and padded by {pad} pixels
    xyxy = torch.tensor(cords).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, img.shape)
    crop = img[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]
    crop_img = Image.fromarray(crop)

    # Save the crop as a base64-encoded string
    with io.BytesIO() as buffer:
        crop_img.save(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image



"""
Another easy method to save crops
//but it trims the crops a little
import PIL.Image as Image

# Create an image object
image = Image.open("image.jpg")
crop = img.crop((cords[0], cords[1], cords[2], cords[3]))

# Save the cropped image
crop.save('crop.png') 

"""