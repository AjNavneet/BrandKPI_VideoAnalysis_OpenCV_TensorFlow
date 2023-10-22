import tensorflow as tf  
from PIL import Image  # Import the Python Imaging Library (PIL)
import cv2  # Import OpenCV library
import numpy as np  
import uuid  # Import UUID library for generating unique identifiers
import os  

from .admin import model_path, label_path  
from .utility import load_image_into_numpy_array, calculate_area, delete_and_create_folder, shortest_longest_area  # Import utility functions

import sys
sys.path.append("../models/research")  # Add a path to sys for TensorFlow research models
from object_detection.utils import label_map_util  # Import utility functions from TensorFlow object detection
from object_detection.utils import visualization_utils as vis_util  # Import visualization utility functions

# Define a class (BrandObjectService) to process video frames and calculate KPI metrics
class BrandObjectService:
    def __init__(self, video_path):
        self.video_path = video_path  # Initialize video path
        self.save_path = "./save_path"  # Set the save path for frames
        self.predicted_path = './predicted_frames'  # Set the path for predicted frames

        delete_and_create_folder(self.save_path)  # Create or clear the save folder
        delete_and_create_folder(self.predicted_path)  # Create or clear the predicted frames folder

    def predict(self):
        NUM_CLASSES = 7  # Set the number of classes for object detection
        KPIs_dict = dict()  # Initialize a dictionary to store KPI metrics

         # Load a (frozen) TensorFlow model into memory.
        detection_graph = tf.Graph()  # Create a TensorFlow graph for object detection
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()  # Read the serialized model
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')  # Import the graph definition

        # Loading label map for object detection
        label_map = label_map_util.load_labelmap(label_path)  # Load the label map
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)  # Create a category index

        # Size, in inches, of the output images
        IMAGE_SIZE = (500, 500)  # Set the image size
        count = 0  # Initialize frame count
        frame_number = 0  # Initialize frame number

        cap = cv2.VideoCapture(self.video_path)  # Open the video file for reading
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while cap.isOpened():
                    frame_number += 1  # Increment frame number

                    ret, frame = cap.read()  # Read a frame from the video

                    filename = str(uuid.uuid4()) + ".jpg"  # Generate a unique filename
                    fullpath = os.path.join(self.save_path, filename)  # Create the full path for saving the frame
                    cv2.imwrite(fullpath, frame)  # Save the frame as an image
                    count += 1

                    ### For testing script, break after 50 frames
                    if count == 50:
                        break

                    image = Image.open(fullpath)  # Open the saved image
                    image_np = load_image_into_numpy_array(image)  # Convert the image to a NumPy array
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')  # Get image tensor
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  # Get bounding boxes
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')  # Get detection scores
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')  # Get detected classes
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')  # Get the number of detections

                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded}
                    )

                    # Visualization of the results of a detection
                    image, box_to_display_str_map = vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8
                    )

                    image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
                    im_width, im_height = image_pil.size
                    area_whole = im_width * im_height
                    for key, value in box_to_display_str_map.items():
                        ymin, xmin, ymax, xmax = key
                        (left, right, top, bottom) = (
                            xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                        area = calculate_area(top, left, bottom, right)

                        percent_area = round(area / area_whole, 2)
                        rindex = value[0].rfind(':')
                        brand_name = value[0][:rindex]

                        if brand_name in KPIs_dict.keys():
                            KPIs_dict[brand_name]['count'] += 1
                            KPIs_dict[brand_name]['area'].append(percent_area)
                            KPIs_dict[brand_name]['frames'].append(frame_number)
                        else:
                            KPIs_dict[brand_name] = {"count": 1}
                            KPIs_dict[brand_name].update({"area": [percent_area]})
                            KPIs_dict[brand_name].update({"frames": [frame_number]})

                    full_predicted_path = os.path.join(self.predicted_path, str(uuid.uuid4()) + ".jpg")
                    cv2.imwrite(full_predicted_path, image)  # Save the predicted frame

        KPIs_dict = self.process_kpi(KPIs_dict)  # Process the KPI metrics
        return KPIs_dict  # Return the KPI metrics

    # Define a function that will return the dictionary with KPI metrics per logo
    def process_kpi(self, KPIs_dict):
        for each_brand, analytics_dict in KPIs_dict.items():
            area = analytics_dict['area']
            response = shortest_longest_area(area)  # Calculate the shortest and longest areas
            KPIs_dict[each_brand].update(response)
        return KPIs_dict  # Return the updated KPI metrics
