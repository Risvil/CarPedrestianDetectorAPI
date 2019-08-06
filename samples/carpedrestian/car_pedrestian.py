import numpy as np
import cv2
from mrcnn.visualize import apply_mask


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


# def apply_mask(image, mask, color, alpha=0.5):
#     """Apply the given mask to the image.
#     """
#     for c in range(3):
#         image[:, :, c] = np.where(mask == 1,
#                                   image[:, :, c] *
#                                   (1 - alpha) + alpha * color[c] * 255,
#                                   image[:, :, c])
#     return image


def merge_results_on_image(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
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
