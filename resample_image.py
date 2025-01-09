import SimpleITK as sitk
import nibabel
import numpy as np

def resample_image(image, target_spacing, target_size=None):
    """
    Resample a 2D image to the specified resolution and target size.

    :param image: Input image (numpy array or SimpleITK Image).
    :param target_spacing: Target pixel spacing (resolution), format: (spacing_x, spacing_y), unit: mm.
    :param target_size: Target image size (number of pixels), format: (width, height). If None, it is calculated based on the target resolution.
    :return: Resampled image array (Image Array).
    """
    # If the input image is a numpy array, convert it to a SimpleITK image
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)

    # Retrieve the original image information
    original_spacing = image.GetSpacing()  # Original resolution (pixel size)
    original_size = image.GetSize()  # Original size (number of pixels)

    # If target size is not specified, calculate it based on target resolution and original size
    if target_size is None:
        target_size = [
            int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1])))
        ]

    # Configure the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)  # Set the target resolution (pixel spacing)
    resampler.SetSize(target_size)  # Set the target size
    resampler.SetOutputOrigin(image.GetOrigin())  # Keep the origin unchanged
    resampler.SetOutputDirection(image.GetDirection())  # Keep the direction unchanged
    resampler.SetInterpolator(sitk.sitkLinear)  # Set the interpolation method to linear (can be changed to other methods)

    # Perform resampling
    resampled_image = resampler.Execute(image)
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    return resampled_array


import numpy as np
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt


# 将 NumPy 数组转换为 SimpleITK 图像
# image = sitk.GetImageFromArray(image_array)
image = sitk.ReadImage("/Users/zhangzhe/PycharmProjects/data/OpenDataset/Testing/A1K2P5/A1K2P5_sa_gt.nii.gz")
pixel_spacing = image.GetSpacing()  # 返回 (spacing_x, spacing_y, spacing_z)
print("Pixel Spacing:", pixel_spacing)
image_array = sitk.GetArrayFromImage(image)
image_2d = image_array[1, 1, :, :]
image = sitk.GetImageFromArray(image_2d)
pixel_spacing = tuple(round(spacing, 2) for spacing in pixel_spacing[:2])
image.SetSpacing(pixel_spacing[:2])
pixel_spacing = image.GetSpacing()
print("Pixel Spacing:", pixel_spacing)
# 目标分辨率（像素大小）
target_spacing = (1.25, 1.25)
target_size = (256, 256)
# 重采样图像
resampled_array = resample_image(image_2d, target_spacing, target_size)
resampled_image = sitk.GetImageFromArray(resampled_array)
resampled_pixel_spacing = resampled_image.GetSpacing()
print("resampled_pixel_spacing:", resampled_pixel_spacing)

# 显示原始和重采样后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_2d, cmap="gray")
height, width = image_2d.shape
Original_pixel = f"Image size: {width} x {height} pixels pixel_spacing:{pixel_spacing}"
plt.text(0.5, -0.1, Original_pixel, ha="center", va="center", transform=plt.gca().transAxes, fontsize=10)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Resampled Image")
plt.imshow(resampled_array, cmap="gray")
height, width = resampled_array.shape
Resampled_pixel = f"Image size: {width} x {height} pixels resampled_pixel_spacing:{target_spacing}"
plt.text(0.5, -0.1,Resampled_pixel, ha="center", va="center", transform=plt.gca().transAxes, fontsize=10)
plt.axis("off")

plt.show()