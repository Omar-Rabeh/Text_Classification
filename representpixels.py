import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from matplotlib.pyplot import imshow

(train_img,train_labels),(test_img,test_labels)=mnist.load_data()#(img , 0:9) ,(10000, 0:9)

x= type(train_img)
print(f"type: {x} , axis: {train_img.ndim}, shape: {train_img.shape} ,dtype: {train_img.dtype}")

digit = train_img[4,14:,14:]
#digit = train_img[10:100]

#img = imshow(digit,camp = plt.cm.b)
plt.imshow(digit,cmap = plt.cm.binary)
plt.show()

digit = train_img[4,:,:]