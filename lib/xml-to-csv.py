import os  
import glob  # Import glob to search for XML files in a directory
import pandas as pd  
import xml.etree.ElementTree as ET  # Import ElementTree for parsing XML

# Define a function to convert XML annotations to CSV
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):  # Iterate through XML files in the specified directory
        tree = ET.parse(xml_file)  # Parse the XML file
        root = tree.getroot()
        for member in root.findall('object'):
            # Extract relevant information from the XML file
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)  # Append the extracted information to the list
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)  # Create a DataFrame from the list
    return xml_df

# Define a function to convert XML annotations to CSV and save as a CSV file
def train():
    annotation_path = os.path.join('/BrandExposure/data/train', 'annotation')  # Specify the directory with XML annotations
    xml_df = xml_to_csv(annotation_path)  # Convert XML annotations to a DataFrame
    
    labels_path = os.path.join('/BrandExposure/data', 'train.csv')  # Specify the path for the CSV output
    xml_df.to_csv(labels_path, index=None)  # Save the DataFrame as a CSV file

train()  # Execute the conversion and save the CSV
