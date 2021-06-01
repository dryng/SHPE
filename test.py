import torch
import torch.nn as nn
import data
import annotate
import numpy as np
import model as m
import training
from skimage import io
#import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

myData = data.CleanData("mpii_human_pose_v1_u12_1.mat")
print('got data')