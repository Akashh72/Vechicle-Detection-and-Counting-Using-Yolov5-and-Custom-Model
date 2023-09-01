import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import Profile, check_img_size, non_max_suppression,scale_boxes
from ultralytics.utils.plotting import Annotator, colors
from tracker import Tracker
import numpy as np
import argparse
import os
import glob
import time


# Open video capture
parser = argparse.ArgumentParser(description="Vehicle Detection Script")

# Add an argument for the video file path
parser.add_argument("--path", type=str, required=True, help="Path to the video file")

# Parse the command-line arguments
args = parser.parse_args()

# Use the provided video file path
directory_path  = args.path
directory_path = str(directory_path)

file_list = glob.glob(os.path.join(directory_path, '*'))
num_files = len(file_list)
print(f"Number of files in the directory: {num_files}")
for file_path in file_list:
    # Your code to process each file goes here
    print(f"Processing file: {file_path}")

    start_time = time.time()

    #Read Classes
    my_file = open("classes.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    area_class_dictionary = dict()
    for _class in class_list:
        area_class_dictionary[_class] = set()

    # Initialize the tracker
    tracker = Tracker()

    # Other parameters
    imgsz = (640, 640)
    source = str(file_path)


    # Replace with your video file path
    weights = 'finalmodel.pt'  # Replace with your model weights path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=True, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1  # batch_size
    vid_stride = 1
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    object_id_counter = 0
    object_id_mapping = {}  # Maps object index to its ID

    area_coords = [(0, 210), (0, 230), (640, 230), (640, 210)]


    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            im = torch.nn.functional.interpolate(im, size=imgsz, mode='area')

        # Inference
        with dt[1]:
            pred = model(im, augment=True)

        # NMS
        with dt[2]:
            # pred = non_max_suppression(pred, conf_thres, iou_thres)
            pred = non_max_suppression(pred,conf_thres=0.7, iou_thres=0.6, classes=None, agnostic=True, labels=(),max_det = 1000)


            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                s += '%gx%g ' % im.shape[2:]  # print string
                im0 = cv2.resize(im0, imgsz)  # Resize im0 to imgsz
                
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Prepare the detection results for tracking
                    objects_rect = []
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        x, y, w, h = xyxy
                        objects_rect.append((x, y, w, h, c))

                    # Update the tracker with the detection results
                    tracked_objects = tracker.update(objects_rect)

                    # # Display the tracking results
                    for rect in tracked_objects:
                        x, y, w, h, c, obj_id = rect
                        label = f'{names[c]} ID: {obj_id}'
                        center_x = int((x + w) // 2)
                        center_y = int((y + h) // 2)
                        if cv2.pointPolygonTest(np.array(area_coords,np.int32), (center_x, center_y), False) >= 0:
                            area_class_dictionary[names[c]].add(obj_id)
    for _class in class_list:
        count = len(area_class_dictionary.get(_class, []))
        print(f"{_class} = {count}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
