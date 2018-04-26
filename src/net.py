import abc
import cv2
import time
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE
slim = tf.contrib.slim


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.global_step = slim.get_or_create_global_step()
        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def testInit(self,checkpoint): 

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE

	self.input_1 = tf.placeholder(tf.float32, [512, 1152, 3])
	self.input_2 = tf.placeholder(tf.float32, [512, 1152, 3])

	inputs = {
            'input_a': tf.expand_dims(self.input_1, 0),
            'input_b': tf.expand_dims(self.input_2, 0),
	}
	
        predictions = self.model(inputs, training_schedule)
        self.predicted_flow = predictions['flow']

        saver = tf.train.Saver()

        self.sess = tf.Session() 
        saver.restore(self.sess, checkpoint)
	print('initialization successful')

    def fn2Opflow(self, input_a, input_b ):
	input_a, top_a, bottom_a, left_a, right_a = self.resizeImage(input_a)
        input_b, top_b, bottom_b, left_b, right_b = self.resizeImage(input_b)
        input_a = input_a[..., [2, 1, 0]]
        input_b = input_b[..., [2, 1, 0]]
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0
        
        #print('input_a shape: {}'.format(input_a.shape))
        #print('input_b shape: {}'.format(input_b.shape)) 	
        start = time.time()     
        pred_flow = self.sess.run(self.predicted_flow,feed_dict={self.input_1:input_a, self.input_2:input_b})[0, :, :, :] 
	duration = time.time()-start
	out_flow = self.resizeFlow(pred_flow, top_a, bottom_a, left_a, right_a)
        #print('flow shape: {}'.format(out_flow.shape))
	return out_flow



    def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=False, save_flo=True,batchTestDirectory=None):

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE

	input_1 = tf.placeholder(tf.float32, [512, 640, 3])
	input_2 = tf.placeholder(tf.float32, [512, 640, 3])

	inputs = {
            'input_a': tf.expand_dims(input_1, 0),
            'input_b': tf.expand_dims(input_2, 0),
	}
	
        #inputs = {
        #    'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
        #    'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
        #}
        predictions = self.model(inputs, training_schedule)
        predicted_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
	    print(checkpoint)
            saver.restore(sess, checkpoint)
	    if (batchTestDirectory):
		totalTime = 0.0
		numOfIterations = 0.0
		for fileName in os.listdir(batchTestDirectory):
			if fileName.endswith('-0.png'):
	
				input_a = imread(batchTestDirectory+fileName)
				input_b = imread(batchTestDirectory+fileName.split('-')[0]+'-1.png')
				input_a, top_a, bottom_a, left_a, right_a= self.resizeImage(input_a)
                                input_b, top_b, bottom_b, left_b, right_b= self.resizeImage(input_b)
				input_a = input_a[..., [2, 1, 0]]
				input_b = input_b[..., [2, 1, 0]]
				if input_a.max() > 1.0:
                                    input_a = input_a / 255.0
                                if input_b.max() > 1.0:
                                    input_b = input_b / 255.0
				
				print('input_a shape: {}'.format(input_a.shape))
				print('input_b shape: {}'.format(input_b.shape)) 	
            			start = time.time()     
            			pred_flow = sess.run(predicted_flow,feed_dict={input_1:input_a, input_2:input_b})[0, :, :, :] #
	    			duration = time.time()-start

				if save_image:
				    flow_img = flow_to_image(pred_flow)
				    flow_img = self.resizeFlowImage(flow_img , top_a, bottom_a, left_a, right_a)
				    full_out_path = os.path.join(out_path, fileName.split('-')[0]+ '.png')
                                    imsave(full_out_path, flow_img)
                                                                                                                
                                if save_flo:
				    out_flow = self.resizeFlow(pred_flow, top_a,bottom_a, left_a, right_a)
				    print('flow shape: {}'.format(out_flow.shape))
                                    full_out_path = os.path.join(out_path, fileName.split('-')[0]+ '.flo')
                		    write_flow(out_flow, full_out_path)

				totalTime += duration 
				numOfIterations += 1.0
		averageTime = totalTime/numOfIterations
	    	print('average runtime over {} iterations is {}'.format(numOfIterations, averageTime))
	    else: 
		input_a = imread(input_a_path)                    	
                input_b = imread(input_b_path)
                print('input_a shape: {}'.format(input_a.shape))
                                                                  
                input_a, top_a, left_a = self.resizeImage(input_a)
                input_b, top_b, left_b = self.resizeImage(input_b)
                height, width = input_b.shape[0:2]
                                                                  
                # Convert from RGB -> BGR
                input_a = input_a[..., [2, 1, 0]]
                input_b = input_b[..., [2, 1, 0]]
                                                                  
                # Scale from [0, 255] -> [0.0, 1.0] if needed
                if input_a.max() > 1.0:
                    input_a = input_a / 255.0
		if input_b.max() > 1.0:                           	
                    input_b = input_b / 255.0
                                                                  
		start = time.time()     
		pred_flow = sess.run(predicted_flow,feed_dict={input_1:input_a, input_2:input_b})[0, :, :, :] #
		duration = time.time()-start
		print('one iteration runtime: {}'.format(duration))
		unique_name = 'flow-' + str(uuid.uuid4())
		if save_image:
		    flow_img = flow_to_image(pred_flow)
		    full_out_path = os.path.join(out_path, unique_name + '.png')
		    imsave(full_out_path, flow_img)

		if save_flo:
		    full_out_path = os.path.join(out_path, input_a_path.split('/')[-1]+ '.flo')
		    write_flow(pred_flow, full_out_path)



    def resizeImage(self,img):
	height, width = img.shape[0:2]
	top = 0
	bottom = 0
	left = 0
	right = 0
	BLACK = [0, 0, 0]
        assert (height <= 512), "image height needs to be smaller than or equal to 512"
        assert (width <= 1152), "image width needs to be smaller than or equal to 1152"
        top = (512-height)//2
        left = (1152-width)//2
	if(height%2 == 0):
		bottom = top
	else:
		bottom = top+1
	if(width %2 == 0):
		right = left
	else:
		right = left+1
	#if (height%64 != 0):
	#    top = ((height//64+1)*64-height)//2 
	#if (width%64 != 0):
	#    left = ((width//64+1)*64-width)//2
	newImg = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
	return (newImg, top, bottom, left, right)

    def resizeFlowImage(self, flow, top, bottom, left, right):
        height, width = flow.shape[0:2]
        return flow[bottom:height-top, left:width-right]

    def resizeFlow(self, flow, top, bottom, left, right):
        height, width = flow.shape[0:2]
        return flow[bottom:height-top, left:width-right,:]

               

    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.iteritems():
                variables_to_restore = slim.get_variables(scope=scope)
                renamed_variables = {
                    var.op.name.split(new_scope + '/')[1]: var
                    for var in variables_to_restore
                }
                restorer = tf.train.Saver(renamed_variables)
                with tf.Session() as sess:
                    restorer.restore(sess, checkpoint_path)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            slim.learning.train(
                train_op,
                log_dir,
                # session_config=tf.ConfigProto(allow_soft_placement=True),
                global_step=self.global_step,
                save_summaries_secs=60,
                number_of_steps=training_schedule['max_iter']
            )
