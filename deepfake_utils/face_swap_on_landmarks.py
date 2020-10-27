import cv2
import train_utils
import os 
import autoencoder
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import face_alignment
from train_loader_util import find_landmarks


def fillHole(im_in):
	im_floodfill = im_in.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv

	return im_out/im_out.max()



model = autoencoder.Autoencoder_with_landmarks()
inf = train_utils.deepfake_generator_with_landmarks(model_class= model, checkpoint_path = "model.pth")

swaps = []

for x in range(10):
    path = "datasets/large/A/" + os.listdir("datasets/large/A")[x]
    original_img = cv2.resize(cv2.imread(path), (64,64))

    img_b = inf.inference(image_path = path , decoder = "B")  ## baby face

    mask = np.zeros_like(original_img)


    landmarks = find_landmarks(original_img)
    start_point = (landmarks["x"][0], landmarks["y"][0])

    for l in range( len(landmarks["x"][2:18]) -1 ):
        c = (landmarks["x"][l], landmarks["y"][l])
        c_2 = (landmarks["x"][l+1], landmarks["y"][l+1])
        mask = cv2.line(mask, c, c_2, color = (255,255,255), thickness = 1)

    mask = cv2.line(mask, start_point, c_2, color = (255,255,255), thickness = 1)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    mask = fillHole(mask)

    mask_bar = 1- mask

 

    box = np.ones([64,64,3])

    for i in range(3):

        box[:,:,i] = box[:,:,i] * mask_bar

    for i in range(3):

        img_b[:,:,i] = img_b[:,:,i] * mask

    fake = (original_img.astype(np.float32)/original_img.max() * box).astype(np.float32) + img_b
   
    fin = cv2.vconcat([original_img.astype(np.float32)/original_img.max(), img_b, fake])

    swaps.append(fin)

save = cv2.hconcat(swaps)
plt.imshow(cv2.cvtColor(save, cv2.COLOR_BGR2RGB))
plt.show()
