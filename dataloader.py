from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#image resize
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])


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
