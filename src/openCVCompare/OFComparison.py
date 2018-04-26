
# coding: utf-8

# In[1]:


import time
import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../../')
from src.flowlib import flow_to_image, evaluate_flow_file, show_flow, write_flow
from src.net import Mode
from src.flownet2.flownet2 import FlowNet2


# In[2]:


pathTosfOutput = '/hdd/SMILE/segmentation/flownet2-tf/data/testOutput/sfOutput/'
pathTofbOutput = '/hdd/SMILE/segmentation/flownet2-tf/data/testOutput/fbOutput/'
pathTofn2Output = '/hdd/SMILE/segmentation/flownet2-tf/data/testOutput/fn2Output/'
pathofTestSet = '/hdd/SMILE/segmentation/flownet2-tf/data/flowTestSet/'
pathToGTFolder = '/hdd/SMILE/segmentation/flownet2-tf/data/flowGT/'


# In[14]:


def showFlow(directory):
    for filename in os.listdir(directory):
        if filename.startswith('.'):
            continue
        show_flow(directory+filename)


# use farneback

# In[18]:



def testFBImages(directory):
	totalTime = 0.0
	for i in range(12):
		i = i+1;
		fileinput = directory+'frame'+str(i)
		filename1 = fileinput+'-0.png' 
		img1 = cv2.imread(filename1)
		print(filename1)
		filename2 = fileinput+'-1.png' 
		print(filename2)
		img2 = cv2.imread(filename2)
		gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
		start = time.time()
		flow = cv2.calcOpticalFlowFarneback(gray1,gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		duration = time.time()-start
		totalTime += duration
		outputFile = pathTofbOutput+'frame'+str(i)+'.flo'
		print('writing to {}'.format(outputFile))
		cv2.optflow.writeOpticalFlow(outputFile, flow)
	print('average runtime over 12 iterations: {}'.format(totalTime/12.0))


# In[19]:


#testFBImages(pathofTestSet)


# use simpleflow

# In[20]:



def testSFImages(directory):
	totalTime = 0.0
	for i in range(12):
		i = i+1;
		fileinput = directory+'frame'+str(i)
		filename1 = fileinput+'-0.png' 
		img1 = cv2.imread(filename1)
		print(filename1)
		filename2 = fileinput+'-1.png' 
		print(filename2)
		img2 = cv2.imread(filename2)
		#gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		#gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

		start = time.time()
		flow = cv2.optflow.calcOpticalFlowSF(img1, img2, 3, 2, 40, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
		duration = time.time()-start
		totalTime += duration

		outputFile = pathTosfOutput+'frame'+str(i)+'.flo'
		print('writing to {}'.format(outputFile))
		cv2.optflow.writeOpticalFlow(outputFile, flow)
	print('average runtime over 12 iterations: {}'.format(totalTime/12.0))


# In[21]:


#testSFImages(pathofTestSet)


# use flownet2

# In[3]:


def resizeImageFile(imgFile):
    BLACK = [0, 0, 0]
    img = cv2.imread(imgFile)
    print img.shape
    height, width = img.shape[0:2]
    top = 0
    left = 0
    if (height%64 != 0):
        top = ((height//64+1)*64-height)//2
    if (width%64 != 0):
        left = ((width//64+1)*64-width)//2
    newImg = cv2.copyMakeBorder(img,top,top,left,left,cv2.BORDER_CONSTANT,value=BLACK)

    return (newImg, top, left)


# In[4]:


def resizeFlow(flow, top, left):
    height, width = flow.shape[0:2]
    return flow[top:height-top, left:width-left]


# In[5]:



def testFN2Images(directory):
    totalTime = 0.0
    
    for i in range(12):
        i = i+1;
        fileinput = directory+'frame'+str(i)
        filename1 = fileinput+'-0.png' 
        print(filename1)
        filename2 = fileinput+'-1.png' 
        print(filename2)

        img1, top1, left1 = resizeImageFile(filename1)
        img2, top2, left2 = resizeImageFile(filename2)

        net = FlowNet2(mode=Mode.TEST)
       
        #start = time.time()       
        flow, duration = net.predictFlow(
                checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                img_a = img1,
                img_b = img2,
            )

        #duration = time.time()-start
        totalTime += duration
        flow = resizeFlow(flow, top1, left1)
        outputFile = pathTofn2Output+'frame'+str(i)+'.flo'
        print('writing to {}'.format(outputFile))
        cv2.optflow.writeOpticalFlow(outputFile, flow)
    print('average runtime over 12 iterations: {}'.format(totalTime/12.0))


# In[6]:


testFN2Images(pathofTestSet)


# showSimpleFlow results

# In[22]:


#showFlow(pathTosfOutput)


# Show Farbeback Results

# In[16]:


#showFlow(pathTofbOutput)

