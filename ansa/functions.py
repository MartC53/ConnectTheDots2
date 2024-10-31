import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io, filters, measure, morphology
from skimage.morphology import disk, binary_dilation, binary_erosion, remove_small_objects
from skimage.feature import canny
from skimage.measure import regionprops_table
from skimage.filters import sobel, rank
from scipy.ndimage import binary_fill_holes, uniform_filter
from PIL import Image
from ipywidgets import interact, IntSlider
from IPython.display import display


def backgroundSub(Data, n_bg=240):
  """Remove the background utilizng the average of 240 frames  
  Extended Summary: 
  -----------------
  This funciton will take the first 240 frames, average the data, then subtract
  it from the entire data set creating a backgorund removal dataset. 
  Parameters:
  -----------
  Data: numpy.ndarray
        [number of images, number of pixel rows, number of pixel columns]
  Returns:
  --------
    Data_adj: numpy.ndarray
        [number of images, number of pixel rows, number of pixel columns] 
  Example:
  --------
  >>>background(Data)
  
  """
  import numpy as np
  [n_images, length, width] = np.shape(Data)

  #Define background array 
  background = np.zeros((length, width))

  #Average the entire background data Frame 1-240
  for i in range (0, n_bg):
    background = background + Data[i,:,:]
  background = background / n_bg

  #Subtract the background from entire dataset 
  Data_adj = np.zeros((n_images, length, width))
  for i in range(n_images):
    Data_adj[i,:,:] = Data[i,:,:] - background
  Data_adj[Data_adj < 0] = 0
  return Data_adj.astype(np.uint16)

# @title Gaussian blur
def gaussianBlur(data):
  """
  For each frame in the image stack data, uses gaussian blur with a 7x7 kernel.
  """
  return np.array([cv2.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT) for img in data])

# @title Circular averaging
def circularBlur(data):
  """
  For each frame in the image stack data, uses circular blur with a 7x7 kernel.
  """
  kernel = skimage.morphology.disk(4).astype(np.float32)
  kernel /= np.sum(kernel).astype(np.float32)
  return np.array([cv2.filter2D(img, -1, kernel) for img in data])

# @title Binary threshold
def binaryThreshold(data):
  """
  For each frame in the image stack data, uses Otsu's thresholding to binarize
  the image.
  """
  # return np.array([cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for img in data])
  return np.array([cv2.threshold(img, 65, 255, cv2.THRESH_BINARY)[1] for img in data])

def preprocessData(data):
  bg_sub = backgroundSub(data)
  blur = gaussianBlur(bg_sub)
  thresh = binaryThreshold(blur)
  # return bg_sub, blur, thresh
  return thresh

def avi_bg(directory_path):
    files = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            data = io.imread(file_path)
            path = file_path[:-4]
            back = backgroundSub(data, n_bg=25)
            io.imsave(path+'_bg.tif', back)

def circle_mask(data: np.ndarray) -> np.ndarray:
    """
    Apply a circular mask to the center of the image.

    Parameters:
        data (np.ndarray): The input grayscale image.

    Returns:
        np.ndarray: Image with the circular mask applied.
    """
    height, width = data.shape
    center = (width // 2, height // 2)
    radius = int(min(center) * 0.90)  # 90% of max possible radius

    y, x = np.ogrid[:height, :width]
    mask = (np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius)
    
    return np.where(mask, data, 0)


def calculate_snr(data: np.ndarray) -> float:
    """
    Calculate the signal-to-noise ratio (SNR) of puncta in an image.

    Parameters:
        data (np.ndarray): The input grayscale image.

    Returns:
        float: The calculated SNR value.
    """
    threshold_value = filters.threshold_otsu(data)
    binary_image = morphology.remove_small_objects(data > threshold_value, min_size=64)
    labeled_image = measure.label(binary_image)

    mask = np.zeros_like(data, dtype=bool)
    for region in measure.regionprops(labeled_image):
        if region.area >= 10:
            mask[region.coords[:, 0], region.coords[:, 1]] = True

    signal = np.mean(data[mask])
    noise = np.std(data[~mask])
    
    return signal / noise if noise else float('inf')


def adaptive_contrast(data: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply adaptive contrast enhancement to an image using CLAHE.

    Parameters:
        data (np.ndarray): Input grayscale image.
        clip_limit (float): Threshold for contrast limiting.

    Returns:
        np.ndarray: Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit)
    return clahe.apply(data)


def remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using Z-score method.

    Parameters:
        df (pd.DataFrame): DataFrame with numerical columns.
        threshold (float): Z-score threshold for defining outliers.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    z_scores = np.abs((df - df.mean()) / df.std())
    return df[(z_scores < threshold).all(axis=1)]


def rolling_ball_filter(image_stack: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a rolling ball filter to an image stack.

    Parameters:
        image_stack (np.ndarray): 3D array of shape (num_images, height, width).
        window_size (int): Number of frames to average for each position.

    Returns:
        np.ndarray: Processed image stack.
    """
    num_images, height, width = image_stack.shape
    processed_stack = np.zeros((num_images, height, width))

    for i in range(num_images):
        start_index = max(0, i - window_size)
        end_index = min(num_images, i + window_size + 1)
        avg_frames = np.mean(image_stack[start_index:end_index], axis=0)
        processed_stack[i] = image_stack[i] - avg_frames

    return processed_stack


def show_images_static(image_stack1: np.ndarray, image_stack2: np.ndarray, index: int = 0) -> None:
    """
    Display two images side by side from two image stacks at a specified index.

    Parameters:
        image_stack1 (np.ndarray): First image stack.
        image_stack2 (np.ndarray): Second image stack.
        index (int): Index of the image to display in each stack.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_stack1[index], cmap='gray')
    axes[0].set_title(f'Stack 1 - Image {index}')
    axes[0].axis('off')
    
    axes[1].imshow(image_stack2[index], cmap='gray')
    axes[1].set_title(f'Stack 2 - Image {index}')
    axes[1].axis('off')
    plt.show()


def calculate_region_properties(data: np.ndarray) -> pd.DataFrame:
    """
    Calculate properties of labeled regions in a binary image.

    Parameters:
        data (np.ndarray): Input grayscale image.

    Returns:
        pd.DataFrame: DataFrame of region properties.
    """
    threshold_value = filters.threshold_otsu(data)
    cleaned_image = morphology.remove_small_objects(data > threshold_value, min_size=20)
    labeled_image = measure.label(cleaned_image)

    properties = []
    for region in measure.regionprops(labeled_image, intensity_image=data):
        properties.append({
            'Label': region.label,
            'Area': region.area,
            'Eccentricity': region.eccentricity,
            'Equivalent Diameter': region.equivalent_diameter,
            'Mean Intensity': region.mean_intensity,
            'Max Intensity': region.max_intensity
        })

    return pd.DataFrame(properties)


def adaptive_histogram_equalization(frame: np.ndarray, clip_limit: float = 0.01, grid_size: int = 8) -> np.ndarray:
    """
    Apply adaptive histogram equalization to a single frame.

    Parameters:
        frame (np.ndarray): Input image.
        clip_limit (float): Contrast clipping limit.
        grid_size (int): Grid size for local contrast equalization.

    Returns:
        np.ndarray: Enhanced frame.
    """
    kernel_size = tuple(max(1, s // grid_size) for s in frame.shape)
    local_mean = uniform_filter(frame, size=kernel_size)
    local_sqr_mean = uniform_filter(frame**2, size=kernel_size)
    local_var = local_sqr_mean - local_mean**2
    local_std = np.maximum(np.sqrt(local_var), clip_limit)

    return np.clip((frame - local_mean) / local_std, -1, 1)


def apply_adaptive_contrast_equalization(dataset: np.ndarray, reference_frame_idx: int = 170) -> np.ndarray:
    """
    Apply adaptive histogram equalization to each frame based on a reference frame.

    Parameters:
        dataset (np.ndarray): 3D image dataset [frames, height, width].
        reference_frame_idx (int): Index of the reference frame.

    Returns:
        np.ndarray: Enhanced dataset.
    """
    reference_frame = adaptive_histogram_equalization(dataset[reference_frame_idx])
    enhanced_dataset = np.array([adaptive_histogram_equalization(frame) for frame in dataset], dtype=np.float32)

    return enhanced_dataset

def show_images(image_stack1, image_stack2):
    """
    Displays two different images side by side from two image stacks based on the selected index.
    
    Parameters:
    image_stack1 (numpy.ndarray): The first input image stack of shape (num_images, height, width)
    image_stack2 (numpy.ndarray): The second input image stack of shape (num_images, height, width)
    """
    num_images1, height1, width1 = image_stack1.shape
    num_images2, height2, width2 = image_stack2.shape
    
    def view_images(index):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show the current image from the first stack on the left
        axes[0].imshow(image_stack1[index], cmap='gray')
        axes[0].set_title(f'Stack 1 - Image {index}')
        axes[0].axis('off')
        
        # Show the current image from the second stack on the right
        axes[1].imshow(image_stack2[index], cmap='gray')
        axes[1].set_title(f'Stack 2 - Image {index}')
        axes[1].axis('off')
        
        plt.show()

    # Create the slider widget
    slider = IntSlider(min=0, max=min(num_images1, num_images2) - 1, step=1, description='Index:')
    
    # Use the interact function to link the slider to the view_images function
    interact(view_images, index=slider)
