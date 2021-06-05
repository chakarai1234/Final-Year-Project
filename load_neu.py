import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical,normalize


def load_neu_data_normalized(test_size=0.3,random_state=42,data_dir = "./Data/NEU surface defect database/",categorize=False,shuffle=False):
    raw_images = []
    for f in os.listdir(data_dir):
        if f.endswith(".bmp"):
            raw_images.append(f)
    img_as_arr = []
    for x in raw_images:
        img_as_arr.append(cv2.imread(data_dir+x,0))
    categories = {'Cr':0, 'In':1, 'PS':2, 'Pa':3, 'RS':4, 'Sc':5}
    targets = [x[:2] for x in raw_images]
    targetsss = np.array([categories[i] for i in targets])
    x_train,x_test,y_train,y_test = train_test_split(img_as_arr,targetsss,test_size=test_size,random_state=random_state,shuffle = shuffle)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape((-1,200,200,1))
    x_train = x_train.astype('float32')
    x_test = x_test.reshape((-1,200,200,1))
    x_test = x_test.astype('float32')
    x_train = normalize(x_train,axis=1)
    x_test = normalize(x_test,axis=1)
    
    if categorize == True:
        y_train = to_categorical(y_train,num_classes=6)
        y_test = to_categorical(y_test,num_classes=6)

    return (x_train,x_test),(y_train,y_test)


def load_neu_data_notnormalized(test_size=0.3,random_state=42,data_dir = "./Data/NEU surface defect database/",categorize=False,shuffle=False):
    raw_images = []
    for f in os.listdir(data_dir):
        if f.endswith(".bmp"):
            raw_images.append(f)
    img_as_arr = []
    for x in raw_images:
        img_as_arr.append(cv2.imread(data_dir+x,0))
    categories = {'Cr':0, 'In':1, 'PS':2, 'Pa':3, 'RS':4, 'Sc':5}
    targets = [x[:2] for x in raw_images]
    targetsss = np.array([categories[i] for i in targets])
    x_train,x_test,y_train,y_test = train_test_split(img_as_arr,targetsss,test_size=test_size,random_state=random_state,shuffle = shuffle)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape((-1,200,200,1))
    x_train = x_train.astype('float32')
    x_test = x_test.reshape((-1,200,200,1))
    x_test = x_test.astype('float32')
    
    if categorize == True:
        y_train = to_categorical(y_train,num_classes=6)
        y_test = to_categorical(y_test,num_classes=6)

    return (x_train,x_test),(y_train,y_test)