from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


im = Image.open("C:/Users/vidal/OneDrive/Images/Camera Roll/WIN_20180401_18_28_11_Pro.jpg")
im2 = Image.open("C:/Users/vidal/OneDrive/Images/Camera Roll/WIN_20180401_18_48_24_Pro.jpg")
im_arr1 = np.array(im) / 255.0
im_arr2 = np.array(im2) / 255.0

im_arr = np.array([im_arr1, im_arr2])

plt.imshow(im_arr[0])
plt.show()

print(im_arr.shape)

np.save("im_array", im_arr)

new_arr = np.load("im_array.npy")

new_arr_flattened = np.reshape(new_arr, (2* 480 * 640 * 3))

print(new_arr_flattened.shape)


plt.imshow(new_arr[0])
plt.show()