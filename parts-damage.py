#pip install roboflow supervision opencv-python
#
from PIL import Image
from roboflow import Roboflow
import supervision as sv
import cv2
import tempfile
import os

# Load the Roboflow API and authenticate with your API key
rf = Roboflow(api_key="l3VppnppccFDHOcbFqRQ")

# Load the project for identifying parts of the car
project_parts = rf.workspace().project("car-parts-segmentation")
model_parts = project_parts.version(2).model

# Load the project for detecting damaged areas of the car
project_damage = rf.workspace().project("car-damage-detection-ha5mm")
model_damage = project_damage.version(1).model

# Path to the input image
img_path = "/home/thegador/Data/technical/Openshift-AI-Roadshow/images/Buick-Encore.jpg"

# Display the image
Image.open(img_path)

# Run the models on the input image
result_damage = model_damage.predict(img_path, confidence=40).json()

# Extract labels and detections from the results
labels_damage = [item["class"] for item in result_damage["predictions"]]
detections_damage = sv.Detections.from_inference(result_damage)

# Extract coordinates of the damaged area
coordinates = []
for List_Coordinates in detections_damage.xyxy:
    for item in List_Coordinates:
        item = int(item)  # Convert to integer
        coordinates.append(item)

print(coordinates)

# Unpack coordinates
x1, y1, x2, y2, x3, y3, x4, y4 = coordinates

# Initialize label and mask annotators
label_annotator = sv.LabelAnnotator(text_scale=0.15)
mask_annotator = sv.MaskAnnotator()

# Read the input image
image = cv2.imread(img_path)

# Annotate damaged areas of the car
annotated_image_damage = mask_annotator.annotate(
    scene=image, detections=detections_damage)

# Display the annotated damaged areas image
sv.plot_image(image=annotated_image_damage, size=(10, 10))

# Crop the damaged area from the original image
annotated_image_damage = annotated_image_damage[y1:y2, x1:x2]

# Create a temporary directory and save the cropped damaged area
temp_dir = tempfile.mkdtemp()
damage_detect_img = os.path.join(temp_dir, "damage_image.png")
cv2.imwrite(damage_detect_img, annotated_image_damage)

# Run the parts detection model on the cropped damaged area
result_parts = model_parts.predict(damage_detect_img, confidence=15).json()
labels_parts = [item["class"] for item in result_parts["predictions"]]
detections_parts = sv.Detections.from_inference(result_parts)

# Print the parts of the car with probable damages
print("The parts of the car with probable damages are:")
for label in labels_parts:
    print(label)

# Remove the temporary files
os.remove(damage_detect_img)
os.rmdir(temp_dir)


