import os
from PIL import Image, ImageDraw
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FireHydrantDataset(Dataset):
	def __init__(self, root, transforms=None):
		self.root = root
		self.transforms = transforms
		# load all image files, sorting them
		self.imgs = sorted(os.listdir(os.path.join(root, "fire_hydrants")))

	def __getitem__(self, idx):
		# load images and bounding boxes
		img_path = os.path.join(self.root, "fire_hydrants", self.imgs[idx])
		img = Image.open(img_path).convert("RGB")

		if self.transforms is not None:
			img = self.transforms(img)

		return img, self.imgs[idx]

	def __len__(self):
		return len(self.imgs)


def get_transform(train=False):
	# converts the image, a PIL image, into a PyTorch Tensor
	transforms = [T.ToTensor()]
	# during training, randomly flip the training images and ground-truth for data augmentation
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


my_dataset = FireHydrantDataset('.', get_transform())
data_loader = DataLoader(dataset=my_dataset, shuffle=False)

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# move model to the right device
model.to(device)

model.eval()

with torch.no_grad():
	for image, img_name in data_loader:
		images = list(image.to(device))
		predictions = model(images)

		for idx in range(len(predictions)):
			prediction = predictions[idx]

			image = Image.fromarray(images[idx].mul(255).permute(1, 2, 0).byte().numpy())
			draw = ImageDraw.Draw(image)

			for idx in range(len(prediction['boxes'])):
				boxes = prediction['boxes'][idx].cpu().numpy()
				score = np.round(prediction['scores'][idx].cpu().numpy(), decimals=4)
				label = COCO_INSTANCE_CATEGORY_NAMES[prediction['labels'][idx]]

				if score > 0.8:
					draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline='red', width=3)
					draw.text((boxes[0], boxes[1]), text=f'{label} : {score}')

			image.save('./fire_hydrant_predictions/' + img_name[0])
