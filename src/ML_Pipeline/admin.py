import os

# Define the paths for the model and label files using os.path.join
model_path = os.path.join("../", 'input', 'frozen_inference_graph.pb')
label_path = os.path.join("../", 'input', 'labels.txt')