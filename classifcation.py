import os
from PIL import Image
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torchvision import transforms, models, datasets
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Classification(nn.Module):
    classes = ["cane", "cavallo", "elefante", "farfalla", "gallina",
               "gatto", "mucca", "pecora", "ragno", "scoiattolo"]
    translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
                 "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
                 "ragno": "spider", "scoiattolo": "squirrel"}

    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.classifier = nn.Sequential(nn.Flatten(),
                                              nn.Linear(512, 256),
                                              nn.ReLU(),
                                              nn.Dropout(0.2),
                                              nn.Linear(256, 10)).to(device)
        self.model.load_state_dict(torch.load(
            'vgg16_pretrained.pt', map_location=device))

    def forward(self, x):
        x = x.to(device)
        pred = self.model(x)
        pred = torch.softmax(pred, dim=1)
        conf, clss = torch.max(pred, 1)
        clss = self.classes[(np.squeeze(clss.numpy()) if not torch.cuda.is_available(
        ) else np.squeeze(clss.cpu().numpy()))]
        return conf.item(), clss

    def predict_from_path(self, path):
        image = Image.open(path)
        image = image.resize((224, 224), Image.ANTIALIAS)
        return self.predict_from_image(image)

    def predict_from_image(self, image):
        im_tensor = torch.from_numpy(np.array(image))
        im_tensor = im_tensor.permute(2, 0, 1)
        im_tensor = im_tensor.unsqueeze(0)
        im_tensor = im_tensor.to(device, dtype=torch.float32)
        conf, predicted_class = self(im_tensor)
        translated_class = self.translate[predicted_class]
        return {'class': translated_class, 'confidence': f'{conf:.4f}'}
