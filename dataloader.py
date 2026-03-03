<<<<<<< HEAD
from PIL import Image
import matplotlib.pyplot as plt
import os

from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

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

#image transformations
transform = v2.Compose([v2.ToTensor(), v2.Resize((224, 224)), v2.RandomHorizontalFlip(0.3), v2.ColorJitter(0.5, 0.3, 0.3), v2.RandomGrayscale()])
=======
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#image resize
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

>>>>>>> refs/remotes/origin/main

#path dataset folders
train_dir = "dataset_split/train"
val_dir = "dataset_split/val"
test_dir = "dataset_split/test"


#new dataset folders
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
 


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


#dataLoaders for each dataset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)


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
