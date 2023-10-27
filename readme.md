# Brand(KPIs) Exposure using Video Analysis in Tensorflow

## Business Objective

In the world of marketing and advertising, brand exposure plays a pivotal role in assessing the effectiveness of promotional strategies. This project aims to measure brand exposure during an IPL match video. The primary objective is to calculate key performance metrics (KPIs) for each brand logo, including the number of logo appearances, area coverage, frame frequency, and area percentage variations.

## Data Description

For this case study, we utilize a video clip from an IPL match between CSK and RCB. This 2-minute 35-second dataset, downloaded from YouTube, undergoes further processing, including frame extraction and object detection. Annotators convert the data into XML files, which are later transformed into CSV files for KPI computation.

---

## Aims and Objectives

The primary objectives of this project are as follows:

1. Calculate KPI metrics for each brand logo.
2. Determine the number of logo appearances.
3. Compute the total area occupied by each logo.
4. Analyze the frequency of logos across frames.
5. Measure the shortest and longest area percentage relative to logo size.

---

## Tech Stack

- Language: `Python`
- Libraries: `TensorFlow`, `Pillow`, `OpenCV`, `Matplotlib`, `NumPy`, `uuid`

---

## Approach

### Data Preparation and Annotation

1. Download the input video:
   - Obtain the video from YouTube.
   - Utilize a Python script (Youtube_downloader.py) for downloading.

2. Use the annotation tool (LabelImg) to create XML files:
   - Annotate images with logo positions and details.

### Model Training and Object Detection

3. Set up TensorFlow for object detection:
   - Clone the necessary repositories and set up the environment.

4. Convert XML files to CSV files:
   - Transform annotation data into CSV format for training.

5. Convert CSV files to TFRecords:
   - Prepare data for TensorFlow model training.

6. Configure base model:
   - Download and customize the base model (e.g., ssd resnet 50fpn coco).

7. Train the model:
   - Initiate model training and monitor results using TensorBoard.

8. Freeze the model:
   - Combine checkpoint files (data, index, meta) into a single frozen model.

9. Export inference graph:
   - Generate the frozen_inference_graph.pb for making predictions.

### Logo Detection and KPI Computation

10. Predict brand logos in test images:
    - Utilize the trained model to detect logos in video frames.

11. Tweak visualization utilities:
    - Modify utilities to return bounding boxes and associated scores.

12. Video frame processing:
    - Process the video into frames for logo detection.

13. Compute KPI metrics:
    - Calculate KPIs such as logo appearance count, total area, frame frequency, and area percentage variations.

---

## Modular Code Overview

1. `input`: Contains input data, including test_video.mp4, labels.txt, and frozen_inference_graph.pb.
2. `src`: Contains modularized code for various project steps.
3. `output`: Stores the output data, which includes a dictionary with key performance metrics for each logo.
4. `lib`: Reference folder containing the original iPython notebook and a reference presentation (ppt).

---

## Key Concepts Explored

1. Understanding the business problem and the significance of brand exposure.
2. Leveraging Python for video data acquisition.
3. Annotating image data using LabelImg.
4. Cloning and customizing a model for object detection with TensorFlow.
5. Training and monitoring a deep learning model for logo detection.
6. Visualizing training results using TensorBoard.
7. Implementing logo detection in video frames.
8. Calculating KPI metrics to measure brand exposure.

---

# Project Workflow

## Assumption: 
The modular code of the project assumes that we already have a trained object detection model and it does not focus on data collection, labelling and preparation. Although the step by step process has been explained in the video lectures to be replicated by the learners.

## Approach:

1. **Data Collection**: Collect the video of Pepsi brand exposure on YouTube with the youtube_downloader.py script. Video Link - https://www.youtube.com/watch?v=xWOoBJUqlbI

2. **Data Labelling**: Label each frame of the video with the object detection tool LabelImg to identify the appearance of Pepsi and other brands in the video.

3. **Data Preparation**: Prepare the labelled data for training by splitting it into training and testing sets and converting it into a format (XML to CSV to TFrecords) suitable for training an object detection model.

4. **Setting up Tensorflow for Object Detection**: 
   * Clone the repository 
      ``` git clone https://github.com/tensorflow/models.git```
   * Run ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` inside /models/research directory.

5. **Download the Base Model**: Download the base model- ssd_resnet_50_fpn_coco from the repository https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md and extract.

6. **Model Training**: Get the training files from the previously cloned Tensorflow repository models/research/object_detection. Copy the train.py file from legacy and paste into the base folder and Train an object detection model on the labelled data as explained in the video lectures.

7. **Model Freezing**: Freeze the ckpt files into a frozen graph file for inference.

8. **KPI Calculation:** Calculate Key Performance Indicators (KPIs) such as the total appearance count, largest area percentage, smallest area percentage, and exposure over time comparison to understand the impact of the video on brand exposure.

---

### **Setting up the Modular Code**
The modular code is specifically for inference purposes and not training.

   * Install requirements.txt using following command
   ```pip install -r requirements.txt```

   * Clone the repository 
      ``` git clone https://github.com/tensorflow/models.git```
   * Run ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` inside /models/research directory.

   * Run Engine.py for all the KPI's metrics

---
