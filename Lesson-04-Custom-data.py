import torch
from torch import nn
import os
import pathlib
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, List
import torchvision
from torchvision import datasets, transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer 
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

"""
Working with Pytorch has a consistent set of steps:
    Find a dataset, turn the dataset into numbers, 
    build a model (or find an existing model) to 
    find patterns in those numbers that can be 
    used for prediction.

PyTorch has many built-in datasets used for 
    a wide number of machine learning benchmarks. 
    However, oftentimes you will want to use 
    your own custom dataset - data related to a 
    problem specific to you

In this lession the workflow we used in Lesson 1 
and 2 will be applied but we will be using our
own dataset of pizza, steak, and sushi images so our
model can train and predict on them

Using a subset of the Food101 dataset, which has 1000 images 
of 101 different foods (101,000 total). We will start with
3 food classes and a random 10% of the 1000 images per class
"""

print("\n\n\nGetting the data\n\n\n")
# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Setup custom image path
custom_image_path = data_path / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# If the image folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory exists")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steakm sushi data...\n")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...\n")
        zip_ref.extractall(image_path)

"""
we have images of pizza, steak and sushi in standard image classification format.
    Image classification format contains separate classes of images in separate 
    directories titled with a particular class name.
        For example, all images of pizza are contained in the pizza/ directory.
The goal will be to take this data storage structure and turn it into a dataset 
usable with PyTorch
"""
# Walk through the data directory and count files present
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str or pathlib.Path): tatger directory

    Returns:
        A print out of:
            number of subdirectories in dir_path
            number of images (files) in each subdirectory
            name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

"""
Now we wull visualize the data. We will:

    Get all of the image paths using pathlib.Path.glob() 
        to find all of the files ending in .jpg.
    Pick a random image path using Python's random.choice().
    Get the image class name using pathlib.Path.parent.stem.
    And since we're working with images, we'll open the random 
        image path using PIL.Image.open() (PIL stands for Python Image Library).
    We'll then show the image and print some metadata.
"""


# Set seed
random.seed(42)

# 1. Get all image paths (* means "any combination")
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the 
                                    #  name of the directory where the image is stored)
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}") 
print(f"Image width: {img.width}")

# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10,7))
plt.imshow(img_as_array)
plt.title(f"\nImage class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)
plt.show(block=True)

print("\n\n\nLoading image data into ImageFolder dataset\n\n\n")

"""
Now to load the image data into pytorch we need to:
Turn it into tensors (numerical representations of our images).
Turn it into a torch.utils.data.Dataset and subsequently a 
    torch.utils.data.DataLoader, we'll call these Dataset 
    and DataLoader for short.

There are several kinds of pre-built datasets and dataset loaders 
    for pytorch depending on the problem at hand

Structure:
Problem space
    Pre-built datasets and Functions

Vision 	
    torchvision.datasets (We will use this and torchvision.transforms 
                          to prepare the data)
Audio 	
    torchaudio.datasets
Text 	
    torchtext.datasets
Recommendation system 	
    torchrec.datasets

torchvision.transforms contains many pre-built methods for formatting images,
    turning them into tensors and even manipulating them for data augmentation 
    (the practice of altering data to make it harder for a model to learn, 
    we'll see this later on) purposes 
To get experience with torchvision.transforms, let's write a series of transform steps that:
    Resize the images using transforms.Resize() (from about 512x512 to 64x64, 
                                                the same shape as the images on the CNN Explainer website).
        Generally, the larger the shape of the image, the more information a model can recover.
            For example, an image of size [256, 256, 3] will have 16x more pixels than an 
            image of size [64, 64, 3] ((256*256*3)/(64*64*3)=16).
        However, the tradeoff is that more pixels requires more computations

    Flip our images randomly on the horizontal using transforms.RandomHorizontalFlip() 
        (this could be considered a form of data augmentation because it will artificially change our image data).

    Turn our images from a PIL image to a PyTorch tensor using transforms.ToTensor().
"""

# Transform for image
data_transform = transforms.Compose([
    # Resize images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show(block=True)

plot_transformed_images(image_path_list, transform=data_transform, n=3)

"""
Now that we've transformed the images into tensors, we can turn our image data 
    into a dataset capable of being used with pytorch. Since the data is in standard
    image classification format, we can use the class torchvision.datasets.ImageFolder
        Where we can pass it the file path of a target image directory as well as 
        a series of transforms we'd like to perform on our images.
"""

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on images
                                  target_transform=None) # transforms to perform on labels (if needed)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(f"\nTrain data:\n{train_data}\nTest data:\n{test_data}\n")

"""
It looks like PyTorch has registered our Dataset's.

Let's inspect them by checking out the classes and 
    class_to_idx attributes as well as the lengths 
    of our training and test sets.
"""

# Get class names as a list
class_names = train_data.classes
print(f"\nClass Names: {class_names}\n")

# Can also get class names as a dict
class_dict = train_data.class_to_idx
print(f"\nClass Dict: {class_dict}\n")

print(f"\nTrain Data length: {train_data}, Test data length: {test_data} \n")

img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")

"""
Our images are now in the form of a tensor (with shape [3, 64, 64]) 
    and the labels are in the form of an integer relating to a specific class 
    (as referenced by the class_to_idx attribute).
How about we plot a single image tensor using matplotlib?
    We'll first have to to permute (rearrange the order of its dimensions) 
    so it's compatible.
Right now our image dimensions are in the format CHW 
    (color channels, height, width) but matplotlib prefers 
    HWC (height, width, color channels).
"""

# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14)
plt.show(block=True)

"""
Notice the image is now more pixelated (less quality).
    This is due to it being resized from 512x512 to 64x64 pixels.
The intuition here is that if you think the image is harder to 
    recognize what's going on, chances are a model will find it 
    harder to understand too.

We've got our images as PyTorch Dataset's but now let's turn 
    them into DataLoader's.
        We'll do so using torch.utils.data.DataLoader.
Turning our Dataset's into DataLoader's makes 
    them iterable so a model can go through and 
    learn the relationships between samples and 
    targets (features and labels).
To keep things simple, we'll use a batch_size=1 and num_workers=1.

num_workers defines how many subprocesses will be created to load your data.
    Think of it like this, the higher value num_workers is set to, 
    the more compute power PyTorch will use to load your data.
    This ensures the DataLoader recruits as many cores as possible to load data.  
"""

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1, # how many samples per batch
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=1, # how many samples per batch
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=False) # don't need to shuffle it

img, label = next(iter(train_dataloader))

# Batch size will now be 1
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

print("\n\n\nLoad Image data with a custom dataset\n\n\n")

"""
What if a pre-built Dataset creator like 
    torchvision.datasets.ImageFolder() didn't exist?
    Or one for your specific problem didn't exist?
Well, you could build your own.
    But wait, what are the pros and cons of creating 
    your own custom way to load Dataset's?

Structure
Pros of creating a custom Dataset
    Cons of creating a custom Dataset

Can create a Dataset out of almost anything. 	
    Even though you could create a Dataset out of 
    almost anything, it doesn't mean it will work.
Not limited to PyTorch pre-built Dataset functions. 	
    Using a custom Dataset often results in writing more code, 
    which could be prone to errors or performance issues.

To see this in action, let's work towards replicating 
    torchvision.datasets.ImageFolder() by subclassing 
    torch.utils.data.Dataset (the base class for 
                             all Dataset's in PyTorch).

We'll start by importing the modules we need:
    Python's os for dealing with directories 
        (our data is stored in directories).
    Python's pathlib for dealing with filepaths 
        (each of our images has a unique filepath).
    torch for all things PyTorch.
    PIL's Image class for loading images.
    torch.utils.data.Dataset to subclass and create 
        our own custom Dataset.
    torchvision.transforms to turn our images into tensors.
    Various types from Python's typing module 
        to add type hints to our code.

"""

"""
Let's write a helper function capable of creating a list of 
    class names and a dictionary of class names and their 
    indexes given a directory path.

To do so, we'll:
    Get the class names using os.scandir() to traverse a target 
        directory (ideally the directory is in standard image classification format).
    Raise an error if the class names aren't found 
        (if this happens, there might be something wrong with the directory structure).
    Turn the class names into a dictionary of numerical labels, 
        one for each class.

Let's see a small example of step 1 before we write the full function.
"""

# Setup path for target directory
target_directory = train_dir
print(f"\nTarget directory: {target_directory}\n")

# Get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / "train"))])
print(f"\nClass names found: {class_names_found}\n")

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    
    # 3. Create a dictonary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

print(f"\nClasses: {find_classes(train_dir)}\n")

"""
To bulid a custom dataset, we will replicate the 
    functionality of torchvision.datasets.ImageFolder().

Let's break it down:

    Subclass torch.utils.data.Dataset.
    Initialize our subclass with a targ_dir parameter 
        (the target data directory) and transform parameter 
        (so we have the option to transform our data if needed).
    Create several attributes for paths (the paths of our target images), 
        transform (the transforms we might like to use, this can be None), 
        classes and class_to_idx (from our find_classes() function).
    Create a function to load images from file and return them, 
        this could be using PIL or torchvision.io (for input/output of vision data).
    Overwrite the __len__ method of torch.utils.data.Dataset to return 
        the number of samples in the Dataset, this is recommended but not required. 
        This is so you can call len(Dataset).
    Overwrite the __getitem__ method of torch.utils.data.Dataset to return a 
        single sample from the Dataset, this is required.

"""

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir and transform parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        # 3, Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to lead images
    def load_image(self, index: int) -> Image.Image:
        "Opens and image via a path and returns it"
        image_path = self.paths[index]
        return Image.open(image_path)
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples"
        return len(self.paths)
    
    # 6. Overwrite the (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
        
"Before we test the class, we can create some transforms to prepare our images"
# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

"""
Now we can turn the training images in train_dir and our testing images in test_dir
into Dataset's using the ImageFolderCustom class and then test and compare the 
torchvision.datasets.ImageFolder()
"""

train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                      transform=test_transforms)

# Check for equality amongst our custom Dataset and ImageFolder Dataset
print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)

"""
Let's create a helper function called display_random_images() that helps us 
    visualize images in our Dataset's.

Specifically, it'll:

    Take in a Dataset and a number of other parameters such as classes 
        (the names of our target classes), the number of images to 
        display (n) and a random seed.
    To prevent the display getting out of hand, we'll cap n at 10 images.
    Set the random seed for reproducible plots (if seed is set).
    Get a list of random sample indexes 
        (we can use Python's random.sample() for this) to plot.
    Setup a matplotlib plot.
    Loop through the random sample indexes found in step 4 
        and plot them with matplotlib.
    Make sure the sample images are of shape HWC (height, width, color channels) 
        so we can plot them.
"""
# 1. Take in a Dataset as well as a list as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjust samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show(block=True)
    
# Display random images from ImageFolder created Dataset
display_random_images(train_data,
                      n=5,
                      classes=class_names,
                      seed=None)

# Display random images from ImageFolderCustom Dataset
display_random_images(train_data_custom, 
                      n=12, 
                      classes=class_names,
                      seed=None) 

"""
We've turned our raw images into Dataset's (features mapped to labels or X's mapped to y's) 
    through our ImageFolderCustom class.
Now we need to turn it into DataLoader's. We can use torch.utils.data.DataLoader()
"""

train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train dataset
                                     batch_size=1, # how many samples per batch
                                     num_workers=0, # how many subprocesses to use for data loading (higher = more)
                                     shuffle=True) # shuffle the data

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test dataset
                                     batch_size=1, 
                                     num_workers=0, 
                                     shuffle=False) # don't usually need to shuffle testing data

# Get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Batch size will now be 1
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")

"""
We've seen a couple of transforms on our data already but there's plenty more.
    You can see them all in the torchvision.transforms documentation.
The purpose of tranforms is to alter your images in some way.
    That may be turning your images into a tensor (as we've seen before).
    Or cropping it or randomly erasing a portion or randomly rotating them.
Doing these kinds of transforms is often referred to as data augmentation.
    Data augmentation is the process of altering your data in such a way that you 
    artificially increase the diversity of your training set.
        Training a model on this artificially altered dataset hopefully results in 
        a model that is capable of better generalization 
        (the patterns it learns are more robust to future unseen examples).
You can see many different examples of data augmentation performed on images using torchvision.transforms in PyTorch's Illustration of Transforms example.
    Machine learning is all about harnessing the power of randomness and research shows that random transforms 
    (like transforms.RandAugment() and transforms.TrivialAugmentWide()) generally perform better than hand-picked transforms.
The idea behind TrivialAugment is you have a set of transforms and you randomly 
    pick a number of them to perform on an image and at a random magnitude between 
    a given range (a higher magnitude means more instense).

We can try TrivialArgument on our images
    The main parameter to pay attention to in transforms.TrivialAugmentWide() 
    is num_magnitude_bins=31.
        It defines how much of a range an intensity value will be picked to 
        apply a certain transform, 0 being no range and 31 being maximum range 
        (highest chance for highest intensity).
    We can incorporate transforms.TrivialAugmentWide() into transforms.Compose().
"""

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

"""
You usually don't perform data augmentation on the test set. 
    The idea of data augmentation is to to artificially increase 
    the diversity of the training set to better predict on the 
    testing set.
"""

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

"Now we can test the data augmentation"

# Grt all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random images
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)

print("\n\n\nModel 0: TinyVGG w/o data augmentation\n\n\n")

"Now we can create a computer vision classification model using a simple transform"

# Create simple transform, only resizing the images and turning them into tensors
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

"""
With our simple transform at hand, let's now:
    Load the data, turning each training and test folders into a Dataset with
    torchvision.datasets.ImageFolder()
    Then into a DataLoader using torch.utils.data.DataLoader().
        We'll set the batch_size=32 and num_workers to as many CPUs 
        on our machine (this will depend on what machine you're using).

"""

# 1. Load and transform data
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

# 2. Turn data into Dataloaders

# Setup batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"\nCreating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers\n")

# Create Dataloader's
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)
"With the data loaders created, now we can make a model. "
"We will use the same TinyVGG model but using color images instead of grayscale"

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2), # default stride is same as kernal_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
    
torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)

print(f"\nModel 0: {model_0}\n")

"""
Note: One of the ways to speed up deep learning models computing on a GPU is 
    to leverage operator fusion.
        This means in the forward() method in our model above, instead of calling a 
        layer block and reassigning x every time, we call each block in succession 
        (see the final line of the forward() method in the model above for an example).
            This saves the time spent reassigning x (memory heavy) and focuses on only computing on x.
"""

"""
Now we can attempt to do a forward pass on a single image
We will:
    Get a batch of images and labels from the DataLoader
    Get a single image from the batch and unsqeeze() the image 
        so it has a batch size of 1 (so its shape fits the model)
    Perform inference on a single image (ensure image is sent to the target device)
    Print out what's happening and convert the model's raw logits
    to prediction probabilites with torch.softmax() and convert
    the prediction probabilities to prediction labels with torch.argmax()
"""

# 1. Get a batch of images and labels from the DataLoader
img_batch, label_batch = next(iter(train_dataloader_simple))

# 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Simgle image shape: {img_single.shape}\n")

# 3. Perform a forward pass on a single image
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))

# 4. Print out what's happening and convert model logits -> pred probs -> pred label
print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilites:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output logits:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}\n")

"""
It appears that the output is what we expect even though the predictions are often wrong
    This is because the model hasn't been trained yet and is guessing with random weights

We will use torchinfo.summary() to get a summarized version of our model
    It takes a model and an input shape and returns whar happens as a tensor moves
    through the model
"""

summary(model_0, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size

"""
We can now create train and test loop functions and evaulate the model on testing data

We'll make these functions:
train_step() - takes in a model, a DataLoader, a loss function and an optimizer 
    and trains the model on the DataLoader.
test_step() - takes in a model, a DataLoader and a loss function and 
    evaluates the model on the DataLoader.
train() - performs 1. and 2. together for a given number of epochs and 
    returns a results dictionary.
"""

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0,0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    
    # 6. Return the filled results at the end of the epochs
    return results

""" 
Now we will recreate model_0 and train the model for 5 epochs using the torch.nn.CrossEntropyLoss() 
    (since we're working with multi-class classification data) and torch.optim.Adam() with a learning 
    rate of 1e-3 as an optimizer and loss function respectively.
We will also calculate the training time using the timeit.default_timer() method
"""

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 5

# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(train_data.classes)).to(device)

# loss func and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start timer
start_time = timer()

# Train model_0
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# end timer and prinft out how long it took
end_time = timer()
print(f"\n Total training time: {end_time-start_time:.3f} seconds\n")

"""
Looks like the model did poorly. We can evaluate it by plotting its loss curves
    Loss curves show the model's results over time and can show how the model 
    performs on different datasets

We will create a function to plot the values in the model_0_results dictionary
"""

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot and plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show(block=True)

plot_loss_curves(model_0_results)

"""
An overfitting model is one that performs better 
    (often by a considerable margin) on the training set 
    than the validation/test set.
If your training loss is far lower than your test loss, 
    your model is overfitting.
        As in, it's learning the patterns in the training too 
        well and those patterns aren't generalizing to the test data.
The other side is when your training and test loss are not as low as 
    you'd like, this is considered underfitting.

The ideal position for a training and test loss curve is for them 
    to line up closely with each other.

Since the main problem with overfitting is that your model is 
    fitting the training data too well, you'll want to use techniques 
    to "reign it in".
        A common technique of preventing overfitting is known as 
        regularization (capable of fitting more kinds of data).

Here are a few methods to prevent overfitting.        

Structure
Method to prevent overfitting
    What is it?

Get more data 	
    Having more data gives the model more opportunities to learn 
    patterns, patterns which may be more generalizable to new examples.

Simplify your model 	
    If the current model is already overfitting the training data, 
    it may be too complicated of a model. This means it's learning 
    the patterns of the data too well and isn't able to generalize 
    well to unseen data. One way to simplify a model is to reduce 
    the number of layers it uses or to reduce the number of hidden 
    units in each layer.

Use data augmentation 	
    Data augmentation manipulates the training data in a way so 
    that's harder for the model to learn as it artificially adds 
    more variety to the data. If a model is able to learn patterns 
    in augmented data, the model may be able to generalize better 
    to unseen data.

Use transfer learning 	
    Transfer learning involves leveraging the patterns 
    (also called pretrained weights) one model has learned to 
    use as the foundation for your own task. In our case, we 
    could use one computer vision model pretrained on a large 
    variety of images and then tweak it slightly to be more 
    specialized for food images.

Use dropout layers 	
    Dropout layers randomly remove connections between hidden 
    layers in neural networks, effectively simplifying a model 
    but also making the remaining connections better. 
    See torch.nn.Dropout() for more.

Use learning rate decay 
    The idea here is to slowly decrease the learning rate as a 
    model trains. This is akin to reaching for a coin at the 
    back of a couch. The closer you get, the smaller your steps. 
    The same with the learning rate, the closer you get to convergence,
    the smaller you'll want your weight updates to be.

Use early stopping 	
    Early stopping stops model training before it begins to overfit. 
    As in, say the model's loss has stopped decreasing for the past 
    10 epochs (this number is arbitrary), you may want to stop the 
    model training here and go with the model weights that had the 
    lowest loss (10 epochs prior).

When a model is underfitting it is considered to have poor predictive 
    power on the training and test sets.
        In essence, an underfitting model will fail to reduce the loss values 
            to a desired level.

Right now, looking at our current loss curves, I'd considered our 
    TinyVGG model, model_0, to be underfitting the data.

The main idea behind dealing with underfitting is to increase your 
    model's predictive power.

There are several ways to do this:

Structure
Method to prevent overfitting
    What is it?

Add more layers/units to your model 	
    If your model is underfitting, it may not have enough capability 
    to learn the required patterns/weights/representations of the 
    data to be predictive. One way to add more predictive power 
    to your model is to increase the number of hidden layers/units 
    within those layers.
Tweak the learning rate 	
    Perhaps your model's learning rate is too high to begin with. 
    And it's trying to update its weights each epoch too much, 
    in turn not learning anything. In this case, you might lower 
    the learning rate and see what happens.
Use transfer learning 	
    Transfer learning is capable of preventing overfitting and 
    underfitting. It involves using the patterns from a previously 
    working model and adjusting them to your own problem.
Train for longer 	
    Sometimes a model just needs more time to learn representations 
    of data. If you find in your smaller experiments your model 
    isn't learning anything, perhaps leaving it train for a more 
    epochs may result in better performance.
Use less regularization 	
    Perhaps your model is underfitting because you're trying to 
    prevent overfitting too much. Holding back on regularization 
    techniques can help your model fit the data better.

There's a fine line between overfitting and underfitting.
    Because too much of each can cause the other.
"""

print("\n\n\nModel 1: TinyVGG w/ Data Augmentation\n\n\n")

"""
For the next model we will use data augmentation to try and improve
    the results.
First, we'll compose a training transform to include transforms.
    TrivialAugmentWide() as well as resize and turn our images 
    into tensors.
        We'll do the same for a testing transform except without 
        the data augmentation. 
Then, we will turn our images into Dataset's using 
    torchvision.datasets.ImageFolder() and then into 
    DataLoader's with torch.utils.data.DataLoader().
"""

# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
"We'll make sure the train Dataset uses the "
"train_transform_trivial_augment and the test Dataset "
"uses the test_transform."

# Turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

print(f"Augmented Train Data: {train_data_augmented}\nSimple Test Data: {test_data_simple}")

"And we'll make DataLoader's with a batch_size=32 and with num_workers "
"set to the number of CPUs available on our machine "
"(we can get this using Python's os.cpu_count())."

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

"With the data loaded we will build a new model, reusing the TinyVGG class"

# Create model_1 and send it to the target device
torch.manual_seed(42)
model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_augmented.classes)
).to(device)

print(f"\nModel 1: {model_1}\n")

"""
Now we can train. We'll use the same setup as model_0 with only the
 train_dataloader parameter varying:
    Train for 5 epochs.
    Use train_dataloader=train_dataloader_augmented as the 
         data in train().
    Use torch.nn.CrossEntropyLoss() as the loss function 
        (since we're working with multi-class classification).
    Use torch.optim.Adam() with lr=0.001 as the learning rate 
        as the optimizer.
"""

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# Start timer
start_time = timer()

# Train model 1
model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
plot_loss_curves(model_0_results)

"Seems like the model is overfitting the data, we can compare the two models"

print("\n\n\nComparing the models\n\n\n")
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

# Setup a plot 
plt.figure(figsize=(15, 10))

# Get number of epochs
epochs = range(len(model_0_df))

# Plot train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot test loss
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot train accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

# Plot test accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend()

plt.show(block=True)

"Seems like both models performed equally poorly"

print("\n\n\nPredicting using a custom image\n\n\n")

"""
Now since we've trained on a custom dataset we can make a prediction on
    custom data. To do so, we can load an image and then preprocess it 
    in a way that matches the type of data our model was trained on.
In other words, we'll have to convert our own custom image to a tensor 
    and make sure it's in the right datatype before passing it to our 
    model.
We will download a custom image (Picture from the original tutorial 
    'photo of my Dad giving two thumbs up to a big pizza from the 
    Learn PyTorch for Deep Learning GitHub.')
"""

# Read in custom image
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

# Print out image data
print(f"Custom image tensor:\n{custom_image_uint8}\n")
print(f"Custom image shape: {custom_image_uint8.shape}\n")
print(f"Custom image dtype: {custom_image_uint8.dtype}")

"The custom image is of datatype unit8 but our model takes float32"
"We need to convert it to the same format otherwise, the model will error"

# Load in custom image and cinvert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255

# Print out image data
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")


# Plot custom image
plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Image shape: {custom_image.shape}")
plt.axis(False)
plt.show(block=True)

# Create transform pipeline to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

# Print out original shape and new shape
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")

"""
Must add to device to ensure the data and model are on the same device,
and ensure the shape of the tensor is consistent
"""

model_1.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

    # Print out different shapes 
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueeze custom image shape: {custom_image_transformed_with_batch_size.shape}")

   # Make a prediction on image with an extra dimension
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

# Print out prediction logits
print(f"Prediction logits: {custom_image_pred}")

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")

# Find the predicted label
custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
print(f"Prediction: {custom_image_pred_class}\nPrediciton Probablitites: {custom_image_pred_probs}\n")

"""
Having prediction probabilities this similar could mean a couple of things:
    The model is trying to predict all three classes at the same time 
        (there may be an image containing pizza, steak and sushi).
    The model doesn't really know what it wants to predict and is 
        in turn just assigning similar values to each of the classes.
Our case is number 2, since our model is poorly trained, it is 
    basically guessing the prediction.
"""

"""
Doing all of the above steps every time you'd like to make a 
    prediction on a custom image would quickly become tedious.
        So let's put them all together in a function we can 
            easily use over and over again.

Specifically, let's make a function that:
    Takes in a target image path and converts to the right datatype 
        for our model (torch.float32).
    Makes sure the target image pixel values are in the range [0, 1].
    Transforms the target image if necessary.
    Makes sure the model is on the target device.
    Makes a prediction on the target image with a trained model 
        (ensuring the image is the right size and on the same device 
        as the model).
    Converts the model's output logits to prediction probabilities.
    Converts the prediction probabilities to prediction labels.
    Plots the target image alongside the model prediction and 
        prediction probability.
"""

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    """
    Makes a prediction on a target image and plots the image with 
        its prediction
    """

    # 1 . Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the imade pixel values by 255 to get the between [0, 1]
    target_image = target_image / 255

    #3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with extra dimension, send it to target device
        target_image_pred = model(target_image.to(device))
    
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> predicition labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probabilities
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # ensure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show(block=True)

pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)