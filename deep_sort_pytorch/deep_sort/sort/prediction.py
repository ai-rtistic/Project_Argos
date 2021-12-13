import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import torch.nn as nn 
import torch.nn.functional as F
import torch

from gender_age_estimation.resnet_base import BottleNeck
from gender_age_estimation.resnet_base import ResNet
from gender_age_estimation.resnet_base import resnet50

import cv2
from PIL import Image




def predict_age_gender(source, f_idx, lb_x, lb_y, rt_x, rt_y): 

    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    # video source, frame idx, left bottom x, left bottom y, right top x, right top y
    gender_path = 'gender_age_estimation/model/resnet_gender.pt'
    age_path = 'gender_age_estimation/model/resnet_age.pt'

    # Capture bbox from frame image
    vidcap = cv2.VideoCapture(source)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    _,image = vidcap.read()

    crop_img = image[int(rt_y) : int(lb_y), int(lb_x):int(rt_x)]

    # To PIL image
    transform = transforms.ToPILImage()
    pil_img = transform(crop_img)

    # To tensor
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]) 
    img_t = transform(pil_img)
    batch_t = torch.unsqueeze(img_t, 0)
    batch_t = batch_t.to(DEVICE)

    # Load model
    model_gender = torch.load(gender_path)
    model_age = torch.load(age_path)

    # Output (tensor -> ndarray)
    with torch.no_grad(): 
        result_gender = model_gender(batch_t).cpu().numpy()
        result_age = model_age(batch_t).cpu().numpy()
    
    if result_gender[0][0] > result_gender[0][1]:
        gender = 'male'
    else:
        gender = 'female'
    
    if np.argmax(result_age) == 0:
        age = '0-19'
    elif np.argmax(result_age) == 1:
        age = '20-39'
    elif np.argmax(result_age) == 2:
        age = '40-59'
    else:
        age = '60+'
    

    return gender, age

