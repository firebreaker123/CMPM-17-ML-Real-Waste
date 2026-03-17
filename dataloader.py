from PIL import Image
#import matplotlib.pyplot as plt
import os
#import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
import math
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
#import wandb

run = wandb.init(project="Loss Graphs", name="Waste-model-run")

imageIndex = 0

#Directories for each folder of images
cardboardDirectory = "dataset/Cardboard/"
foodOrganicsDirectory = "dataset/Food Organics/"
glassDirectory = "dataset/Glass/"
metalDirectory = "dataset/Metal/"
miscTrashDirectory = "dataset/Miscellaneous Trash/"
paperDirectory = "dataset/Paper/"
plasticDirectory = "dataset/Plastic/"
textTrashDirectory = "dataset/Textile Trash/"
vegetationDirectory = "dataset/Vegetation/"

"""
images = []
labels = []

#Image samples to view from each folder
for file in os.listdir(cardboardDirectory):
    img = Image.open(cardboardDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(foodOrganicsDirectory):
    img = Image.open(foodOrganicsDirectory + file)
    images.append(img)
    labels.append(file)

    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(glassDirectory):
    img = Image.open(glassDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(metalDirectory):
    img = Image.open(metalDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(miscTrashDirectory):
    img = Image.open(miscTrashDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(paperDirectory):
    img = Image.open(paperDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(plasticDirectory):
    img = Image.open(plasticDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(textTrashDirectory):
    img = Image.open(textTrashDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1

imageIndex = 0

for file in os.listdir(vegetationDirectory):
    img = Image.open(vegetationDirectory + file)
    images.append(img)
    labels.append(file)
    
    if(imageIndex >= 14):
        break
    imageIndex = imageIndex + 1


nextBatch = 0

#Loop to view at all image samples, total of 135 images to view
for batch in range(9):
    for idx, image in enumerate(images[nextBatch:]):
        plt1 = plt.subplot(3, 5, idx+1)
        plt1.imshow(image)
        plt1.set_title(labels[idx+nextBatch])
        plt1.axis('off')

        if idx == 14:
            break 

    plt.tight_layout()
    plt.show()

    nextBatch = nextBatch + 15
"""

#image transformations
transformTrain = v2.Compose([v2.ToTensor(), v2.Resize((224, 224)), v2.RandomHorizontalFlip(0.3), v2.ColorJitter(0.5, 0.3, 0.3), v2.RandomGrayscale()])
transform = v2.Compose([v2.ToTensor(),v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), v2.Resize((224, 224))])

#path dataset folders
train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
test_dir = "dataset_split/test"

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA is available. Using GPU.')
else:
    device = 'cpu'

#new dataset folders
train_dataset = datasets.ImageFolder(root=train_dir, transform=transformTrain)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

"""
#metrics
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))


#numeric labels
print(train_dataset.class_to_idx)



#check
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Label index:", label)              
print("Label name:", train_dataset.classes[label])
"""

#dataLoaders for each dataset
if torch.cuda.is_available():
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

"""
#loops
print("train")
for inputs, labels in train_loader:
    print(inputs)
    print(labels)


print("val")
for inputs, labels in val_loader:
    print(inputs)
    print(labels)


print("test")
for inputs, labels in test_loader:
    print(inputs)
    print(labels)
"""

class Convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.3)

        self.linear1 = nn.Linear(128 * 7 * 7, 256) #initially 126 * 14 * 14, 1028
        self.linear2 = nn.Linear(256, 9)

        
    def forward(self, X):
        X = self.conv1(X)
        X= self.batchnorm1(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X= self.batchnorm2(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv3(X)
        X= self.batchnorm3(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv4(X)
        X = self.batchnorm4(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.pool2(X)
        X = X.flatten(start_dim=1)
        X = self.relu(self.linear1(X))
        output = self.linear2(X)
        return output

if __name__ == '__main__':

    model = Convnet()
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    NUM_EPOCHS = 32

    train_loss = 0
    val_loss = 0
    test_loss = 0

    for epoch in range(NUM_EPOCHS):

        train_loss = 0
        val_loss = 0
        num_correct = 0
    
        # Training Loop
        for train_x, train_y in train_loader:
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            train_preds = model(train_x)
            loss = criterion(train_preds, train_y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
        train_loss = train_loss/len(train_loader)

        print("\n------------------------Training Phase-----------------------------\n")
        print(f"Epoch {epoch} | Loss: {train_loss}")
    
        print("\n------------------------Validation Phase-----------------------------\n")

        # Validation Loop
        for val_x, val_y in val_loader:
            val_x = val_x.to(device)
            val_y = val_y.to(device)
        
            val_preds = model(val_x)
            loss = criterion(val_preds, val_y)
            val_loss += loss.item()

            _, class_preds = torch.max(val_preds, dim=1)
            num_correct = num_correct + (class_preds == val_y).sum()
    
        accuracy = num_correct/len(val_dataset)
        val_loss = val_loss/len(val_loader)

        print(f"Epoch {epoch} | Loss: {val_loss} Accuracy {accuracy * 100}")

        #run.log({"Train Loss" : train_loss, "Validation Loss" : val_loss})

    print("\n------------------------Testing Phase-----------------------------\n")

    # Testing Loop
    model.eval()

    num_correct = 0
    test_loss = 0

    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            test_preds = model(test_x)
            loss = criterion(test_preds, test_y)
            test_loss += loss.item()

            _, class_preds = torch.max(test_preds, dim=1)
            num_correct += (class_preds == test_y).sum()

    accuracy = num_correct / len(test_dataset)
    test_loss = test_loss / len(test_loader)

    print(f"Loss: {test_loss} Accuracy {accuracy * 100}")

    print("finished training, saving model")

    torch.save(model.state_dict(), "real_waste_model.pt")