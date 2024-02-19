# !git clone https://github.com/fizyr/keras-retinanet.git

# !pip install keras_retinanet

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
import numpy as np
import glob
import os
import sys
from PIL import Image
import random
sys.path.insert(0, '/content/keras-retinanet')

# import keras_retinanet
# from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
# model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'spine'}

# ## Run detection on example

# In[ ]:


inputs='/content/dataspinal/ori/1001.jpg'
# load image
image1 = cv2.imread(inputs)

# copy to draw on
draw = image1.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image1 = preprocess_image(image1)
image1, scale = resize_image(image1)

# process image
start = time.time()
# boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# correct for image scale
# boxes /= scale



# Crop the image based on the bounding box

# Load the image
image1 = cv2.imread(inputs)

plt.imshow(image1)

# Get image dimensions
height, width, _ = image1.shape

# Specify the width and height of the bounding box
box_width = 250
box_height = 700

# Calculate the top-left coordinates of the bounding box to center it vertically
x = width // 2 - box_width // 2
y = height // 2 - box_height // 2

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

# Create a rectangle patch representing the bounding box
rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')

# Add the rectangle to the axis
ax.add_patch(rect)

# Set axis limits
ax.set_xlim(0, width)
ax.set_ylim(height, 0)


# Show the image with the bounding box
plt.show()

cropped_image = image1[y:y + box_height, x:x + box_width]

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the cropped image
ax.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# Set axis limits
ax.set_xlim(0, box_width)
ax.set_ylim(box_height, 0)

# Show the cropped image
plt.show()

masks = glob.glob("/content/datasetspine/modified_mask/*.png")

masks=sorted(masks)

orgs = glob.glob("/content/datasetspine/modified_ori/*.jpg")

orgs=sorted(orgs)
# Displaying Image Paths-->
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):

    print(image)
    print(mask)
    # imgs_list.append(np.array(Image.open(image).resize((512,512))))
    imgs_list.append([np.array(Image.open(image).convert('L').resize((512,512)))])


    im = Image.open(mask).resize((512,512))

    # bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


    #im_cropped = im.crop((left, top, right, bottom))
    masks_list.append(np.array(im))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)
# Defining custom sizes
image_sizeR=256
image_sizeC=512

imgs_list = []
masks_list = []

for image, mask in zip(orgs, masks):
    # Open, convert to grayscale, and resize the image
    img = np.array(Image.open(image).convert('L').resize((image_sizeR, image_sizeC)))
    imgs_list.append(img)

    # Open and resize the mask
    mask_img = np.array(Image.open(mask).resize((image_sizeR, image_sizeC)))
    masks_list.append(mask_img)

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

print(imgs_np.shape)   # Should print (12, 512, 512)
print(masks_np.shape)  # Should print (12, 512, 512)

print(imgs_np.shape, masks_np.shape)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
transformed = transform(image=imgs_np, mask=masks_np)
transformed_image = transformed['image']
transformed_mask = transformed['mask']

imgs_np1=np.concatenate([imgs_np,transformed_image],axis=0)
masks_np1=np.concatenate([masks_np,transformed_mask],axis=0)
np.array(imgs_np1).shape

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

# Defining Up_Sampling & Down_Sampling blocks(Hyperparameters)
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

# Deifining WNet model
def WNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_sizeR, image_sizeC, 1))

    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

import tensorflow as tf
from tensorflow.keras import layers, models

def unet_block(inputs, filters, kernel_size=(3, 3), padding="same", strides=1, name_prefix=""):
    conv1 = layers.Conv2D(filters, kernel_size, activation='relu', padding=padding, strides=strides, name=f"{name_prefix}_conv1")(inputs)
    conv2 = layers.Conv2D(filters, kernel_size, activation='relu', padding=padding, strides=strides, name=f"{name_prefix}_conv2")(conv1)
    pool = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(conv2)
    return conv2, pool

def upsample_block(inputs, skip, filters, kernel_size=(3, 3), padding="same", strides=1, name_prefix=""):
    upsampled = layers.UpSampling2D((2, 2), name=f"{name_prefix}_upsampled")(inputs)
    concat = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([upsampled, skip])
    conv1 = layers.Conv2D(filters, kernel_size, activation='relu', padding=padding, strides=strides, name=f"{name_prefix}_upconv1")(concat)
    conv2 = layers.Conv2D(filters, kernel_size, activation='relu', padding=padding, strides=strides, name=f"{name_prefix}_upconv2")(conv1)
    return conv2

def build_unet(input_shape=(512, 256, 1), base_filters=64, name_prefix=""):
    inputs = layers.Input(input_shape, name=f"{name_prefix}_input")

    # Encoder
    conv1, pool1 = unet_block(inputs, base_filters, name_prefix=f"{name_prefix}_enc1")
    conv2, pool2 = unet_block(pool1, base_filters * 2, name_prefix=f"{name_prefix}_enc2")
    conv3, pool3 = unet_block(pool2, base_filters * 4, name_prefix=f"{name_prefix}_enc3")

    # Decoder
    up4 = upsample_block(conv3, conv2, base_filters * 2, name_prefix=f"{name_prefix}_dec1")
    up5 = upsample_block(up4, conv1, base_filters, name_prefix=f"{name_prefix}_dec2")

    # Output layer
    output = layers.Conv2D(1, (1, 1), activation='sigmoid', name=f"{name_prefix}_output")(up5)

    model = models.Model(inputs=inputs, outputs=output, name=f"{name_prefix}_U-Net")
    return model

def wnets(image_sizeR=1024, image_sizeC=512, base_filters=64):
    left_input = layers.Input((image_sizeR, image_sizeC, 1), name="left_input")
    right_input = layers.Input((image_sizeR, image_sizeC, 1), name="right_input")

    # Left U-Net
    left_u_net = build_unet(base_filters=base_filters, name_prefix="left")(left_input)

    # Right U-Net
    right_u_net = build_unet(base_filters=base_filters, name_prefix="right")(right_input)

    # Concatenate the outputs of the two U-Nets
    concat_output = layers.Concatenate(axis=-1, name="concatenated_output")([left_u_net, right_u_net])

    # Final Convolutional layer
    final_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name="final_output")(concat_output)

    model = models.Model(inputs=[left_input, right_input], outputs=final_output, name="W-Net")
    return model

# Create and compile the W-Net model
model = wnets()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display the summary of the model
model.summary()

from tensorflow import keras
from tensorflow.keras import layers


# opt = keras.optimizers.Adam(learning_rate=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=opt)

model = WNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])


# Load the image
image1 = cv2.imread(inputs)

plt.imshow(image1)

# Get image dimensions
height, width, _ = image1.shape

# Specify the width and height of the bounding box
box_width = 250
box_height = 700

# Calculate the top-left coordinates of the bounding box to center it vertically
x = width // 2 - box_width // 2
y = height // 2 - box_height // 2

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

# Create a rectangle patch representing the bounding box
rect = patches.Rectangle((x, y), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')

# Add the rectangle to the axis
ax.add_patch(rect)

# Set axis limits
ax.set_xlim(0, width)
ax.set_ylim(height, 0)




# Show the image with the bounding box
plt.show()

cropped_image = image1[y:y + box_height, x:x + box_width]

# Create a figure and axis
fig, ax = plt.subplots(1)

# Display the cropped image
ax.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

# Set axis limits
ax.set_xlim(0, box_width)
ax.set_ylim(box_height, 0)

# Show the cropped image
plt.show()
