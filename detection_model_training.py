from imutils import paths
import os
import cv2
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import splitfolders
from PIL import ImageFilter

seed = 12
img_width =  224
img_height = 224
batch_size = 256
epochs = 50

data_dir = './flickr_logos_27_dataset/flickr_logos_27_dataset_images/'
dest = 'logos'
plot_folder = 'plots'

if not os.path.exists(dest):
    os.makedirs(dest)
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Augmentation functions
train_augmentations = transforms.Compose([
    transforms.Resize(size=(img_width, img_height)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_augmentations = transforms.Compose([
    transforms.Resize(size=(img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Split in to train and validate set
splitfolders.ratio(dest, output="output", seed=42, ratio=(0.8,0.2))

# Load dataset
train_dataset = ImageFolder(root='./output/train',transform=train_augmentations)
validate_dataset = ImageFolder(root='./output/val', transform=val_augmentations)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validate_dataset, batch_size=batch_size,shuffle=True)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Do not update existing params
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 256),
                 nn.Dropout(0.5),
                 nn.ReLU(inplace=True), 
                 nn.Linear(256, len(train_dataset.classes)),                   
                 nn.LogSoftmax(dim=1)]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss()

def train():
    model.train()
    net_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        (data, target) = (data.to(device), target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        output = torch.exp(output) 
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss.backward()
        optimizer.step()
        
        net_loss = net_loss + loss.item()
    acc = correct.float() / len(train_loader.dataset)
    return net_loss,acc

def test():
    model.eval()  
    test_loss = 0
    correct = 0

    with torch.no_grad():  
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)  
            pred = output.max(1, keepdim=True)[1]  
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(val_loader.dataset)  
    acc = correct.float() / len(val_loader.dataset)  
    return test_loss, acc

TRAINING_LOSS = []
TRAINING_ACC = []
TESTING_LOSS = []
TESTING_ACC = []

for epoch in range(1, epochs + 1):
    start = time.time()
    print("--- Epoch {} ---".format(epoch))
    epoch_loss,tracc = train()
    TRAINING_LOSS.append(epoch_loss)
    TRAINING_ACC.append(tracc)
    print("\tTrain Accuracy = {} || Train Loss  = {} ".format(tracc,epoch_loss))
    tloss,tacc =  test()
    print("\tTest Accuracy =  {} || Test Loss = {} ".format(tacc,tloss))
    TESTING_LOSS.append(tloss)
    TESTING_ACC.append(tacc)
    stop = time.time()
    print("\tTraining time = ", (stop - start))

torch.save(model.state_dict(), 'detect_model_weights.pth')
print("Model weights saved")

# Accuracy-Loss Plot
epoch_axis = np.arange(epochs)

training_acc = torch.Tensor(TRAINING_ACC).detach().cpu().numpy()
training_loss = torch.Tensor(TRAINING_LOSS).detach().cpu().numpy()
testing_acc = torch.Tensor(TESTING_ACC).detach().cpu().numpy()
testing_loss = torch.Tensor(TESTING_LOSS).detach().cpu().numpy()

# Loss Plot
plt.figure(figsize=(10, 10))
plt.plot(epoch_axis, training_loss, 'b-', label='Training Loss')
plt.plot(epoch_axis, testing_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss.png', dpi=300) 
plt.close()

# Accuracy Plot
plt.figure(figsize=(10, 10))
plt.plot(epoch_axis, training_acc, 'b-', label='Training Accuracy')
plt.plot(epoch_axis, testing_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy.png', dpi=300) 
plt.close()

print("Plots complete")