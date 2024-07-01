import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir , transform = None):
        self.image_paths, self.mask_paths = self.read_data(data_dir) 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def read_data(self, data_dir):
        img_path_list = []
        img_mask_list = []
        
        # Traverse the directory
        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                image_dir = os.path.join(folder_path, "images")
                mask_dir = os.path.join(folder_path, "masks")
                # Assuming there's only one image in the image directory
                if os.path.exists(image_dir) and os.path.exists(mask_dir):
                    image_files = os.listdir(image_dir)
                    if image_files:
                        img_path = os.path.join(image_dir, image_files[0])
                        img_path_list.append(img_path)
                        
                        # Collect all masks
                        masks = []
                        for mask_file in os.listdir(mask_dir):
                            masks.append(os.path.join(mask_dir, mask_file))
                        img_mask_list.append(masks)
        
        return img_path_list, img_mask_list

    def merge_mask(self, m_paths):
        merged_mask = None
        for m_path in m_paths : 
            try :
                mask = np.array(Image.open(m_path).convert("L"), dtype=np.float32)
            except :
                continue
            if merged_mask is None:
                # Initialize the merged mask with the first mask
                merged_mask = mask
            else:
                # Add the current mask to the merged mask
                merged_mask += mask
                
        return merged_mask

    def __getitem__(self, index) :

        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = self.merge_mask(mask_path)
        try :
            mask[mask == 255.0] = 1.0 # the segmentation output is produced by sigmoid thats why. The mask value is either 0 or 255 in original
        except :
            print(mask_path)
        if self.transform is not None :
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
    

    
