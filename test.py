import gcsfs
import imageio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def process_image(self, image):
    image = T.ToTensor()(image)
    image = image if image.shape[0] != 3 else image.mean(dim=0, keepdim=True)
    return image

def read_image(self, path):
    with fs.open(path, 'rb') as f:
        try:
            image = imageio.imread(f)
            return process_image(image)
        except Exception as e:
            print(f"Error reading image from path {path}: {e}")
            return None

gcs_path = 'gs://combined-dataset/labelled-dataset/CombinedBreastMammography/'
fs = gcsfs.GCSFileSystem()
filenames = [s for s in fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))]
print(f'The dataset contain {len(filenames)} images...')
labels = [filename.replace('CombinedBreastMammography', 'CombinedROIMask').replace("_", "_ROI_", 1) for filename in filenames] # noqa

idx = input("Enter your idx: ")

print(f'filename: {filenames[idx]}')
print(f'labels: {labels[idx]}')
image = read_image(filenames[idx])
roi = read_image(labels[idx])

np_roi = np.array(roi)
np_roi = (np_roi * 255).astype(np.uint8)

imageio.imwrite("original.png", np_roi)

np_roi = np.array(tensor)
np_roi = np.where(np_roi >= 0.5, 1, 0)
np_roi = (np_roi * 255).astype(np.uint8)

imageio.imwrite("threshold.png", np_roi)