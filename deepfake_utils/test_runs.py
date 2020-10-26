import torch
import autoencoder
import autoencoder_utils
import train_loader_util 

device = ("cuda" if torch.cuda.is_available() else "cpu")

"""
This script is there to help me debug the utils

for now the standard image size would be 64*64
"""

print("sizes form dataloaders: ")



print(batch_a["x"].size())
print(batch_a["y"].size())

print(batch_b["y"].size())
print(batch_b["y"].size())


image  = batch_a["x"]

print("image batch size :", image.size())

dummy_conv = autoencoder_utils._ConvLayer(input_features = 3, output_features  =  10)
pred = dummy_conv(image)
print ("batch through dummy conv size: ", pred.size())


dummy_upscale = autoencoder_utils._UpScale(input_features = 10, output_features  =  4) 
pred  = dummy_upscale(pred)
print ("batch through dummy upsample size: ", pred.size())

dummy_shuffle = autoencoder_utils._PixelShuffler()
shuffled = dummy_shuffle(pred)  ## works on an even number of channels 
print("size after pixel shuffle: ", shuffled.size())

ae = autoencoder.Autoencoder()
pred_a = ae.forward(image, decoder = "A")
pred_b = ae.forward(image, decoder = "B")

print("size after forward pass through  autoencoder (decoder A): ", pred_a.size())
print("size after forward pass through  autoencoder (decoder B): ", pred_b.size())


