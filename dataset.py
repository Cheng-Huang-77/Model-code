import os
from torch.utils.data import Dataset
from skimage import io

class PNGDataset(Dataset):
    def __init__(self, root_dir, class_labels, transform=None):  # root_dir: cwt folder
        self.root_dir = root_dir
        self.class_labels = class_labels
        self.transform = transform
        # Create a list of image paths and labels
        self.image_paths = []
        self.image_labels = []
        self.class_counts = {label: 0 for label in class_labels}

        # Walk through numbered folders and class folders
        for numbered_folder in os.listdir(root_dir):
            numbered_folder_path = os.path.join(root_dir, numbered_folder)
            if os.path.isdir(numbered_folder_path):
                for class_folder in class_labels:
                    class_folder_path = os.path.join(numbered_folder_path, class_folder)
                    if os.path.isdir(class_folder_path):
                        for img_file in os.listdir(class_folder_path):
                            if img_file.lower().endswith(".png"):
                                self.image_paths.append(os.path.join(class_folder_path, img_file))
                                self.image_labels.append(class_labels[class_folder])
                                self.class_counts[class_folder] += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the image
        img_path = self.image_paths[index]
        image = io.imread(img_path)
        label = self.image_labels[index]

        # Handle grayscale images by adding a channel dimension
        if len(image.shape) == 2:  # If grayscale, expand to 3 channels
            image = image[:, :, None]  # Add an extra channel dimension
            image = image.repeat(3, axis=2)  # Repeat along the channel dimension to make it RGB

        if self.transform:  # Apply transformation, if any
            image = self.transform(image)

        return image, label # Return label as is (no torch.tensor)

    def get_class_counts(self):
        return self.class_counts