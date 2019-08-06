import os
# import sys
# import random
# import math
# import re
# import time
import numpy as np
import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn import utils
# from mrcnn import visualize
from mrcnn.visualize import apply_mask
# import mrcnn.model as modellib
# from mrcnn.model import log

import cv2
from tqdm import tqdm


class CarPedrestianModel():
    def __init__(self):
        self.load()

    def load(self):
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        # Path to Shapes trained weights
        SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")


        # Run one of the code blocks

        # Shapes toy dataset
        # import shapes
        # config = shapes.ShapesConfig()

        # MS COCO Dataset
        import coco
        config = coco.CocoConfig()
        COCO_DIR = "path to COCO dataset"  # TODO: enter value here

        # Override the training configurations with a few
        # changes for inferencing.
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()


        # Device to load the neural network on.
        # Useful if you're training a model on the same 
        # machine, in which case use CPU and leave the
        # GPU for training.
        # DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
        DEVICE = "/gpu:0"

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"


        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # Set weights file path
        if config.NAME == "shapes":
            weights_path = SHAPES_MODEL_PATH
        elif config.NAME == "coco":
            weights_path = COCO_MODEL_PATH
        # Or, uncomment to load the last model you trained
        # weights_path = model.find_last()

        # Load weights
        print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)


    def predict(self, image, verbose=0):
        results = model.detect([image], verbose=verbose)
        return results

    def draw_results(self, results):
        # draw bounding boxes and segmentation masks
        r = results[0]
        frame = merge_results_on_image(frame, r['rois'], r['masks'], r['class_ids'], 
                                    labels, r['scores'],
                                    title="Predictions") #dataset.class_names
        return frame

    def predict_and_draw_results(self, image, verbose=0):
        results = self.predict(image, verbose)
        image_with_masks = draw_results(results)
        return image_with_masks


labels = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

allowed_labels = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck"
]


def get_class_color(class_label):
    label_color_dict = {
        "BG": (0, 0, 0),
        "person": (255, 64, 0),
        "bicycle": (255, 255, 0),
        "car": (128, 255, 0),
        "motorcycle": (255, 0, 128),
        "bus": (191, 0, 255),
        "train": (255, 0, 0),
        "truck": (0, 128, 255)
    }

    return label_color_dict[class_label]        


def merge_results_on_image(image, boxes, masks, class_ids, class_names,
                      scores=None, 
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    show_mask, show_bbox: To show masks and bounding boxes or not
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # masked_image = image.astype(np.uint32).copy()
    masked_image = image.copy()
    for i in range(N):
        # color = colors[i]
        class_id = class_ids[i]

        class_label = labels[class_id]
        if not class_label in allowed_labels:
            continue

        # color = get_class_color(class_id)
        # color_int = [int(x * 255) for x in color]
        color_int = get_class_color(class_label)
        color = [x / 255. for x in color_int]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color=color_int, thickness=2)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]

        font = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.4
        thickness = 1
        (w, h), baseline = cv2.getTextSize(caption, font, fontScale, thickness)
        cv2.rectangle(masked_image, (x1, y1), (x1 + w, y1 - h), color=color_int, thickness=-1)
        cv2.putText(masked_image, caption, (x1, y1), font, fontScale, color=(0, 0, 0), thickness=thickness)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

    return masked_image.astype(np.uint8)
