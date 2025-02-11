# Bosch CV Assignment
This project trains (Only one Epoch) and evaluates an object detection model using TensorFlow & OpenCV within a Dockerized environment. It uses BDD100k datatset for training. Only 10 labels are used in the training process, ignored 'lane' and 'drivable area' on trainig. But they are included in data analysis.

## Project folder structure
bosch_cv_assignment_bdd_100k/
│── data/
│   ├── bdd100k/
│── notebooks/
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│   ├── evaluation.ipynb
│── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│── Dockerfile
│── requirements.txt
│── README.md

## Setup
Here is the github repo for the project.

git clone https://github.com/vijivenkadesh/bosch_assignment_bdd_100k.git
cd bosch_cv_assignment
pip install -r requirements.txt