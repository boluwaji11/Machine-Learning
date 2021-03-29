from sklearn.datasets import load_digits

digits = load_digits()

# print(digits.DESCR)

# print(digits.data[13])

# print(digits.data.shape)

# print(digits.target[13])
# print(digits.target.shape)

print(digits.images[13])

import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))

for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis tick marks
    axes.set_yticks([])  # remove y-axis tick marks
    axes.set_title(target)  # the target value of the image
plt.tight_layout()
plt.show()