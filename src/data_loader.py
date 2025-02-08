#Import Important Libraries

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

#Setting up the directories for data and labels

DATA_DIR = "D:/learning_desk/bosch_assignment_bdd_100k/data/bdd100k/"
LABELS_JSON_FILE = os.path.join(DATA_DIR, 'labels/bdd100k_labels_images_train.json')


#Load JSON Labels

def load_labels(path):
    """"
    This function Just loads the JSON labels and  returns them.

    Args:
    path: The file path of the labels.

    Return:
    Loaded JSON data.
    """
    with open(file=path, mode='r') as f:
        data = json.load(f)
    return data

#list the labels and count them

def label_list(data):
    """
    This function takes the loaded labels and returns the number of classes.
    
    Args:
    data = loaded data from the 'load_labels function'

    Returns:
    It returns the number of classes from the loaded label data
    
    """
    class_count = defaultdict(int)
    for item in data:
        for label in item['labels']:
            class_count[label['category']] += 1
    return class_count

#Ploting the distribution of the class labels

def class_distribution_plot(class_count):
    """
    This function plots the class counts Distribution.

    Args:
    class_count = Number of classes and counts.

    Return:
    It returns class count plot.
    
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(class_count.keys()), y=list(class_count.values()))
    plt.xticks(rotation=45)
    plt.title("Class Distribution Plot")
    plt.show()

if __name__ == "__main__":
    labels = load_labels(path=LABELS_JSON_FILE)
    class_count = label_list(data=labels)
    class_distribution_plot(class_count=class_count)

