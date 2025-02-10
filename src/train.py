#importing required libraries
import tensorflow as tf
import os
import json
import cv2
import numpy as np
import albumentations as A
from tensorflow.keras.utils import to_categorical

#Setting up global variables
IMG_SIZE = (300, 300)
BATCH_SIZE = 8
NUM_CLASSES = 10
CLASS_NAMES = ["traffic light", "traffic sign", "car", "person", "bike", "bus", "truck", "rider", "train", "motor"] 

#Resize the image
AUGMENTATION = A.Compose([A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1])])

#data loader pipeline
def load_labels(label_file):
    """
    This function loads the labels and returns the labels

    Args:
    label_file = File path of the json file.

    Returns:
    It returns the loaded labels
    """
    with open(file=label_file, mode='r') as f:
        return json.load(fp=f)

#Image preprocessing pipeline
def preprocessing_image(image_path):
    """
    This function read the image and converts to RGB format

    Args:
    image_path: Training image file path.

    Returns:
    It returns coloe corrected image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


#Image preprocessing pipeline
def process_sample(sample, image_dir):
    """
    This function returns processed image, bounding box coordinates and label associated with the image.

    Args:
    Sample: Sample image from the training dataset.
    image_dir: Training image path.

    Return:
    It processed image, bounding box coordinates and label associated with the image.
    """
    image_path = os.path.join(image_dir, sample['name'])
    image = preprocessing_image(image_path=image_path)
    
    #place holders for the bounding boxes dimensions and labels
    b_boxes = []
    labels = []

    #extarct bounding box dimensions and labels from sample
    for item in sample['labels']:
        if "box2d" in item:
            x1, y1, x2, y2 = item['box2d']['x1'], item['box2d']['y1'], item['box2d']['x2'], item['box2d']['y2']
            b_boxes.append([x1, y1, x2, y2])
            labels.append(item['category'])

    #resize the image
    augmented = AUGMENTATION(image=image, bboxes=b_boxes)

    return augmented['image'], np.array(b_boxes), np.array(labels)

#data pipeline
def data_generator(image_dir, label_file, batch_size=BATCH_SIZE):
    """
    This function generate batches of images, bounding boxes & labels

    Args:
    image_dir: Image path
    label_file: class labels
    batch_size: Training image batch size

    Return:
    It yields the batches of images, bounding boxes & labels

    """
    labels_data = load_labels(label_file)
    total_images = len(labels_data)

    while True:
        for i in range(0, total_images, batch_size):
            batch_images = labels_data[i:i + batch_size]
            images, boxes, labels = [], [], []
            max_objects = 10

            for sample in batch_images:
                img, box, lbl = process_sample(sample, image_dir)
                images.append(img)
                boxes.append(box[:max_objects])
                labels.append([CLASS_NAMES.index(l) for l in lbl[:max_objects]])
            
            #Convert to TensorFlow Ragged Tensors to handle varying sizes
            images = np.array(images)  # (batch, 300, 300, 3)
            boxes_padded = np.zeros((len(boxes), max_objects, 4))  # Fixed size (batch, 10, 4)
            for j in range(len(boxes)):
                boxes_padded[j, :len(boxes[j])] = boxes[j]
            padded_labels = np.zeros((len(labels), max_objects), dtype=np.int32)  # Pad with `0`
            for j in range(len(labels)):
                padded_labels[j, :len(labels[j])] = labels[j]
                

            labels_onehot = to_categorical(padded_labels, num_classes=NUM_CLASSES)  # One-hot encode

            # Flatten `labels_onehot` to match model shape `(batch, 10 * NUM_CLASSES)`
            labels_onehot = labels_onehot.reshape(len(labels), -1)

            
            yield images, {"bounding_box": boxes_padded.reshape(len(boxes), -1), "class_label": labels_onehot}

# Loading Data and file paths
IMAGE_DIR = "D:/learning_desk/bosch_assignment_bdd_100k/data/bdd100k/images/train"
LABEL_FILE = 'D:/learning_desk/bosch_assignment_bdd_100k/data/bdd100k/labels/bdd100k_labels_images_train.json'

#Traing data

train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(IMAGE_DIR, LABEL_FILE, batch_size=BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(None, 300, 300, 3), dtype=tf.float32),  # Images
        {
            "bounding_box": tf.TensorSpec(shape=(None, 40), dtype=tf.float32),  # (10 objects * 4 coords)
            "class_label": tf.TensorSpec(shape=(None, 10 * NUM_CLASSES), dtype=tf.float32),  # (10 objects * classes)
        }
    )
).prefetch(tf.data.AUTOTUNE)

#model build

def build_model():
    """
    This funtion Builds a multi-task model for object detection + classification and returns the model.
    
    Args:

    Returns:
    It returns a model
    
    """

    # Base Model
    base_model = tf.keras.applications.MobileNetV2(input_shape=(300, 300, 3), include_top=False)
    base_model.trainable = False

    # Object Detection Branch
    x1 = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x1 = tf.keras.layers.Dense(128, activation="relu")(x1)
    bbox_output = tf.keras.layers.Dense(40, activation="linear", name="bounding_box")(x1)  # Predicts (10 objects * 4)

    # Classification Branch
    x2 = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x2 = tf.keras.layers.Dense(128, activation="relu")(x2)
    class_output = tf.keras.layers.Dense(10 * NUM_CLASSES, activation="softmax", name="class_label")(x2)  # Predicts (10 objects * classes)

    # Define Multi Output Model
    model = tf.keras.Model(inputs=base_model.input, outputs=[bbox_output, class_output])

    model.compile(
        optimizer="adam",
        loss={"bounding_box": "mse", "class_label": "categorical_crossentropy"},
        metrics={"bounding_box": "mae", "class_label": "accuracy"},
    )

    return model

#model build and training
model = build_model()

model.fit(train_data, epochs=1, steps_per_epoch=100)
model.save()