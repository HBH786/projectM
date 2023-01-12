import os
import cv2

def preprocess_batch(image_filenames):
    # List to store the image data
    image_data = []

    # Pre-process each image
    for filename in image_filenames:
        img = cv2.imread(filename)
        if img is None:
            print(f"{filename} could not be read.")
            continue
        
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img / 255.0
        image_data.append(img)
        
    return image_data

# path to the directory containing the images
path = "e:\projects\ptest"

# get the filenames of all files in the directory
image_filenames = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

# Pre-process the images
image_data = preprocess_batch(image_filenames)

# Use the pre-processed images as input for your model
# Your model code here


for i, img in enumerate(image_data):
    cv2.imwrite(f"preprocessed_{i}.jpg", img)
