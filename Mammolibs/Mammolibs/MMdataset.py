import gcsfs
import imageio
from torch.utils.data import Dataset
import torchvision.transforms as T


def mutations(image):
    image = T.RandomRotation(180)(image)
    image = T.RandomAutocontrast()(image)
    image = T.RandomPerspective()(image)
    return image

class MMImageSet(Dataset):
    def __init__(self, gcs_path, stage='encoder', aug=True):
        super(MMImageSet, self).__init__()
        self.fs = gcsfs.GCSFileSystem()
        if aug: self.filenames = self.fs.ls(gcs_path)
        else: self.filenames = [s for s in  self.fs.ls(gcs_path) if s.count('_') == 1 and s.endswith(('.png', '.jpg', '.jpeg'))]
        self.stage = stage
        if self.stage == 'finetune':
            self.labels = [filename.replace('CombinedBreastMammography', 'CombinedROIMask').replace('CBIS', 'CBIS_ROI') for filename in self.filenames] # noqa
        print(f'The dataset contain {len(self.filenames)} images...')
        self.count = 0

    def __len__(self):
        return len(self.filenames)
    
    def process_image(self, image):
        image = image if image.shape[0] != 3 else image.mean(dim=0, keepdim=True)
        return T.ToTensor()(image)
    
    def read_image(self, path):
        with self.fs.open(path, 'rb') as f:
            try:
                image = imageio.imread(f)
                return self.process_image(image)
            except Exception as e:
                print(f"Error reading image from path {path}: {e}")
                return None

    def __getitem__(self, idx):
        image = self.read_image(self.filenames[idx])
        # if self.stage != 'finetune':
        if image is None:
            print(f"Image at index {idx} is None. Returning a zero tensor instead.")
            # Replace with the correct shape for your images.
            return torch.zeros(1, 256, 256)
        else:
            print(self.count)
            self.count+=1
            return image
        # else:
        #     roi = self.read_image(self.labels[idx])
        #     return image, roi