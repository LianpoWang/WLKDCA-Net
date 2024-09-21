import os
from torch import nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from WLKDCANet import WLKDCANet


image_path = "1.bmp"
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
image = transform(image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WLKDCANet()
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


with torch.no_grad():
    output = model(image)


output = output.squeeze().cpu().numpy()
df = pd.DataFrame(output)
df.to_csv('output_depth_map.csv', index=False, header=False)
print(output)  # 输出结果


