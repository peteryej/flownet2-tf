import time
import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

'''
if __name__ == '__main__' and __package__ is None:
    	from os import sys, path
    	sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
'''
from ..flowlib import flow_to_image, evaluate_flow_file, show_flow, write_flow
from ..net import Mode
from ..flownet2.flownet2 import FlowNet2

pathTosfOutput = '/hdd/SMILE/segmentation/flownet2-tf/data/testOutput/sfOutput/'
pathTofbOutput = '/hdd/SMILE/segmentation/flownet2-tf/data/testOutput/fbOutput/'
pathTofn2Output = '/hdd/SMILE/segmentation/flownet2-tf/data/testOutput/fn2Output/'
pathofTestSet = '/hdd/SMILE/segmentation/flownet2-tf/data/flowTestSet/'
pathToGTFolder = '/hdd/SMILE/segmentation/flownet2-tf/data/flowGT/'

#def opencvFlowTest(image1, image2):
#	img1 = cv2.imread(image1)
#	img2 = cv2.imread(image2)
#	hsv = np.zeros_like(img1)
#	hsv[...,1] = 255
#	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#	gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#	cv2.imwrite('gray1.png', gray1)
#	cv2.imwrite('gray2.png', gray2)
#	#flow = cv2.calcOpticalFlowFarneback(gray1,gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#	flow = cv2.optflow.calcOpticalFlowSF(img1, img2, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
#	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#    	hsv[...,0] = ang*180/np.pi/2
#    	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#    	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
##	cv2.imshow('flow', bgr)
#	#cv2.imwrite('opticalhsv.png',bgr)
#	cv2.imwrite(pathTosfOutput+'/opticalsf10.png',bgr)
#
#def readFlow(flowFile):
#	output = cv2.optflow.readOpticalFlow(flowFile)
#	cv2.imwrite('opticalsfread.png',output)
#
#def main():
#	image1 = sys.argv[1]
#	image2 = sys.argv[2]
#	opencvFlowTest(image1, image2)

#def resizeImage(img):
#	BLACK = [0, 0, 0]
#	#for filename in os.listdir(directory):
#	#	if filename.startswith('.'):
#	#		continue
#	img = cv2.imread(directory+filename)
#	height, width = img.shape[:2]
#	if (height%64 != 0):
#		top = ((height//64+1)*64-height)//2
#	if (width%64 != 0):
#		left = ((width//64+1)*64-width)//2
#	newImg = cv2.copyMakeBorder(img,top,top,left,left,cv2.BORDER_CONSTANT,value=BLACK)	
#	return newImg
#
#def testImages(directory):
#	totalTime = 0.0
#	for i in range(12):
#		i = i+1;
#		fileinput = directory+'frame'+str(i)
#		filename1 = fileinput+'-0.png' 
#		#img1 = cv2.imread(filename1)
#		print(filename1)
#		filename2 = fileinput+'-1.png' 
#		print(filename2)
#		
#		#img2 = cv2.imread(filename2)
#		#gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#		#gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#		net = FlowNet2(mode=Mode.TEST)
#		
#		start = time.time()
#		net.test(
#        		checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
#        		input_a_path=filename1,
#        		input_b_path=filename2,
#        		out_path=pathTofn2Output,
#    		)
#		#flow = cv2.calcOpticalFlowFarneback(gray1,gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#		#flow = cv2.optflow.calcOpticalFlowSF(img1, img2, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
#		duration = time.time()-start
#		totalTime += duration
#		outputFile = pathTofn2Output+'frame'+str(i)+'.flo'
#		#outputFile = pathTosfOutput+'frame'+str(i)+'.flo'
#		print('writing to {}'.format(outputFile))
#		#cv2.optflow.writeOpticalFlow(outputFile, flow)
#	print('average runtime over 12 iterations: {}'.format(totalTime/12.0))
    
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

def resizeFlow(flow, top, left):
	height, width = flow.shape[0:2]
	return flow[top:height-top, left:width-left]

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
       		flow, duration = net.predictFlow(checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0', img_a = img1,img_b = img2)

        	#duration = time.time()-start
     		totalTime += duration
     		flow = resizeFlow(flow, top1, left1)
    		outputFile = pathTofn2Output+'frame'+str(i)+'.flo'
    		print('writing to {}'.format(outputFile))
     		cv2.optflow.writeOpticalFlow(outputFile, flow)
 	print('average runtime over 12 iterations: {}'.format(totalTime/12.0)
          
#testFN2Images(pathofTestSet)         
#def aveEPE(GTFolder, predictFolder):
#	totalEPE = 0.0
#	for filename in os.listdir(GTFolder):
#		if filename.startswith('.'):
#			continue
#		epe = evaluate_flow_file(GTFolder+filename, predictFolder+filename)
#		totalEPE += epe
#	print("average end point error over 8 tests: {}".format(totalEPE/8))

#opencvFlowTest(pathofTestSet+'frame10-0.png',pathofTestSet+'frame10-1.png')
#testImages(pathofTestSet)
#aveEPE(pathToGTFolder, pathTosfOutput)
#main()
