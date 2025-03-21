# Python script to generate YOLO standard annotations from a .cvs file containing GeoJSON labels. We will assume that the .cvs file contains columns as id | image_id | geometry (in GeoJSON) | class (str). Moreover the images will have same width and height. 

#This script can be improved (or generalized). Instead of fixing width and height we could iterate over each image and found their corresponding dimensions using ().size from Image.open(). In this case one should create a method to find and save the image sizes of each picture. This is in case the images have different sizes, which is a common scenario. 

import pandas as pd 
import os 
import ast
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

class YOLOConverter: 
    def __init__(self, image_width, image_height, output_dir = "data"):
        '''
        Initialize YOLOConverter with image dimensions and output directory. 
        '''
        self.image_width = image_width
        self.image_height = image_height
        self.output_dir = output_dir
        self.class_mapping = {}  # Dictionary to store class mappings

        #Create the directory structure
        self.train_dir = os.path.join(output_dir, "train")
        self.train_images_dir = os.path.join(self.train_dir, "images")
        self.train_labels_dir = os.path.join(self.train_dir, "labels")
        self.val_dir = os.path.join(output_dir, "validation")
        self.val_images_dir = os.path.join(self.val_dir, "images")
        self.val_labels_dir = os.path.join(self.val_dir, "labels")
        
        #Create the directories
        for dir_path in [self.output_dir, self.train_dir, self.train_images_dir, self.train_labels_dir, self.val_dir, self.val_images_dir, self.val_labels_dir]:
        
            os.makedirs(dir_path, exist_ok=True)
        # os.makedirs(self.annotations_dir, exist_ok=True)
  
    def polygon_to_bbox(self, polygon_str):
        '''
        Convert the GeoJSON polygon to a YOLO-format bounding box. Returns a string in "x_center y_center width height" format.
        '''    
        try: 
            polygon = ast.literal_eval(polygon_str)  # Convert string to list of tuples

            # Get min and max values for x and y
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Compute YOLO format (normalized values)
            x_center = (x_min + x_max) / 2 / self.image_width
            y_center = (y_min + y_max) / 2 / self.image_height
            width = (x_max - x_min) / self.image_width
            height = (y_max - y_min) / self.image_height
            
            return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        except Exception as e:
            print(f"Error processing geometry: {e}")
            return None
      
    def convert_csv_to_yolo(self, csv_file, images_dir, train_split=0.8):
        '''
        Read a CSV file and convert its geometries to YOLO format.
        Saves output .txt files per image_id in the specified directory.
        '''
        # Load CSV file
        df = pd.read_csv(csv_file)

        # Create a mapping of class names to integers
        unique_classes = df["class"].unique()
        self.class_mapping = {name: idx for idx, name in enumerate(unique_classes)}

        # Save class mapping to a text file in the output directory
        class_file_path = os.path.join(self.output_dir, "classes.txt")
        with open(class_file_path, "w") as class_file:
            for class_name, class_id in self.class_mapping.items():
                class_file.write(f"{class_id} {class_name}\n")

        # Store the DataFrame for use in organize_dataset
        self.annotations_df = df

        print(f"Class mapping complete! Starting dataset organization...")
        
        # Organize the dataset and create label files
        self.organize_dataset(images_dir, train_split)
        
        print(f"Dataset organization complete! Check '{self.output_dir}' directory for the organized dataset.")

    def organize_dataset(self, images_dir, train_split=0.8):
        '''
        Organize dataset into train and validation sets and create corresponding label files.
        '''
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        
        # Split the data
        train_files, val_files = train_test_split(
            image_files,
            train_size=train_split,
            random_state=42,
            shuffle=True
        )

        # Process training files
        for img_file in train_files:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(self.train_images_dir, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Create label file
            image_id_without_ext = os.path.splitext(img_file)[0]
            label_file = os.path.join(self.train_labels_dir, f"{image_id_without_ext}.txt")
            self._create_label_file(img_file, label_file)

        # Process validation files
        for img_file in val_files:
            # Copy image
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(self.val_images_dir, img_file)
            shutil.copy2(src_img, dst_img)

            # Create label file
            image_id_without_ext = os.path.splitext(img_file)[0]
            label_file = os.path.join(self.val_labels_dir, f"{image_id_without_ext}.txt")
            self._create_label_file(img_file, label_file)

    def _create_label_file(self, image_id, label_file_path):
        '''
        Helper method to create a label file for a specific image.
        '''
        # Get annotations for this image
        image_annotations = self.annotations_df[self.annotations_df['image_id'] == image_id]
        
        with open(label_file_path, "w") as f:
            for _, row in image_annotations.iterrows():
                class_id = self.class_mapping[row["class"]]
                bbox = self.polygon_to_bbox(row["geometry"])
                if bbox:  # Only write valid entries
                    f.write(f"{class_id} {bbox}\n")      
      