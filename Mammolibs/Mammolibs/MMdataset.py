import gcsfs
import imageio
from torch.utils.data import Dataset
import torchvision.transforms as T


def mutations(image):
    image = T.RandomRotation(180)(image)
    image = T.RandomAutocontrast()(image)
    image = T.RandomPerspective()(image)
    return image


def mutation(image):
    image1 = T.RandomRotation(180)(image)
    image1 = T.RandomAutocontrast()(image1)
    return image1


def rgb_to_grayscale(img):
    return img.mean(dim=0, keepdim=True)


class MMImageSet(Dataset):
    def __init__(self, gcs_path, stage='encoder', aug=True):
        super(MMImageSet, self).__init__()
        self.fs = gcsfs.GCSFileSystem()
        if aug:
            self.filenames = self.fs.ls(gcs_path)
        else:
            self.filenames = [s for s in  self.fs.ls(gcs_path) if s.count('_') == 1]
        print(self.filenames[:10])
        self.stage = stage
        print(f'The dataset contain {len(self.filenames)} images...')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with self.fs.open(self.filenames[idx], 'rb') as f:
            image = imageio.imread(f)

        image = T.ToTensor()(image)
        if image.shape[0] == 3:
            image = rgb_to_grayscale(image)

        return image, self.filenames[idx]
