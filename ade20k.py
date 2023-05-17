import os
import numpy as np
import json
from base import BaseDataset
from glob import glob
from PIL import Image

class ADE20KDataset(BaseDataset):
    """
    MIT Scene Parsing dataset: subset of ADE20K dataset
    http://sceneparsing.csail.mit.edu/
    150 labels
    train: ~20K
    val: ~2K
    test: ~3.5K
    """

    def __init__(self, root_folder, ade20kpalette="files/ade20kpalette.json", mode="training", num_classes=150):
        self.root_folder = root_folder
        self.num_classes = 150
        self.mode = mode
        self.palette = json.load(open(ade20kpalette, 'r'))
        try:
            if self.mode in ["training", "validation"]:
                self.image_dir = os.path.join(self.root_folder, "images/"+self.mode)
                self.label_dir = os.path.join(self.root_folder, "annotations/"+self.mode)
                self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + "/*.jpg")]
                print('DEBUG:', len(self.files), self.image_dir)
        except ValueError as err:
            print("Cant access root folder:" + str(err))
        super().__init__()

    def _get_files(self):
        return

    def _load_data(self, idx):
        image_id = self.files[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, self.files[idx] + ".jpg")).convert("RGB"))
        label = np.array(Image.open(os.path.join(self.label_dir, self.files[idx] + ".png"))) - 1
        return image, label, image_id
