# Import the BrandObjectService class from the ML_Pipeline.predict module
from ML_Pipeline.predict import BrandObjectService

# Define the path to the test video
video_path = "../input/test_video.mp4"

# Create an instance of the BrandObjectService class using the video_path
brand_expo_obj = BrandObjectService(video_path)

# Make predictions on the video
kpi_s = brand_expo_obj.predict()

# Print the dictionary containing KPI metrics as output per logo
print(kpi_s)
