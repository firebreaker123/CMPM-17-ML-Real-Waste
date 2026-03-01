import torchvision
import splitfolders
from torchvision import datasets
from torchvision import transforms


input_folder = "dataset"
output_folder = "dataset_split"


splitfolders.ratio(input_folder, output=output_folder, seed = 11, ratio=(0.6, 0.3, 0.1))