import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        target_image = image[:, :512, :] #单张图片1024x512,左边为上色后，右边为素描
        input_image = image[:, 512:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image

class GetTest(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        transform_valid = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ]
        )
        img = Image.open(img_path)
        test_img = transform_valid(img)

        return test_img


if __name__ == "__main__":
    #dataset = MapDataset("data/train/")
    testset = GetTest("data/test/")
    #loader = DataLoader(dataset, batch_size=5)
    loader = DataLoader(testset, batch_size=3)
    for x in loader:

        save_image(x, "data/test/x.png")
        import sys

        sys.exit()