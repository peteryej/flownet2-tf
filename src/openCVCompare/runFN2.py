import time
import cv2
import os
import sys
import numpy as np

#sys.path.append('../../')

#from src.flownet2.test import testInit, fn2Opflow
from ..flownet2.test import testInit, fn2Opflow

filename1 = '/hdd/SMILE/segmentation/flownet2-tf/data/flowTestSet/frame11-0.png'
filename2 = '/hdd/SMILE/segmentation/flownet2-tf/data/flowTestSet/frame11-1.png'

img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)
checkpoint = './checkpoints/FlowNet2/flownet-2.ckpt-0'


net = testInit(checkpoint)

outflow = fn2Opflow(net, img1, img2)
