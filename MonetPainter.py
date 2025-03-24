"""
This is the main file for the MonetPainter program. This program will take an image and apply a Monet style to it.
The model is based on the CycleGAN architecture and the model is trained on the Monet2Photo dataset on Kaggle.
"""

# Importing the necessary libraries
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the Model Architecture
class CycleGenerator(nn.Module):
    """
    Define the architecture of the generator network.
    Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=256, input_channels=3, init_zero_weight=False):
        super(CycleGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.InstanceNorm2d(64),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 64, 3, stride=1, padding=0),
                nn.InstanceNorm2d(64)
            ) for _ in range(6)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x) + x
        return self.decoder(x)


class Discriminator(nn.Module):
    """
    The Discriminator is based on the structure of PatchGAN,
    """
    
    def __init__(self, conv_dim=256, input_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.classifier = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator_M = CycleGenerator().to(device)
generator_M.load_state_dict(torch.load("./models/generator_M.pth", map_location=device))
generator_M.eval()

def process_image(image_path):
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    print(img_tensor.shape)
    
    # Generate
    with torch.no_grad():
        output = generator_M(img_tensor)
    
    # Back to image
    output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    output = (output * 0.5 + 0.5).clip(0, 1)
    return output

# Use the model to generate a Monet style image
image_path = r"微信图片_20250324195327.jpg"
output = process_image(image_path)
plt.imshow(output)
plt.axis("off")
plt.savefig("monet_tran.jpg", bbox_inches="tight")
plt.show()