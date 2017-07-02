import vgg

import cv2
import numpy as np
import tensorflow as tf

ITERATIONS = 10000
CONTENT_WEIGHT=5
STYLE_LAYER_WEIGHT_DECAY = 1# sometimes we can try an exponentially weight decay for loss, just ignore it
CONTENT_WEIGHT_BLEND = 0.5
STYLE_WEIGHT_BLEND = (0.2,0.2,0.2,0.2,0.2)

TV_WEIGHT = 0.05
#CONTENT_WEIGHT = 0.1
#STYLE_WEIGHT = 0.8
LEARNING_RATE = 0.1

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def _tensor_size(tensor):
	from operator import mul
	return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def stylize(network_path, content_img, style_img, initial, poolingmethod, outputfile, CONTENT_WEIGHT, STYLE_WEIGHT):
	print content_img.shape

	netweights,netmeanpixel = vgg.LoadMat(network_path)

	g = tf.Graph()
	with g.as_default(), g.device('/gpu:0'),tf.Session() as sess:
		#placeholder means it is the "source" of computational graph, we can pass data to it, and all computation starts here
		img_inp = tf.placeholder('float32', shape=content_img.shape)
		netdict = vgg.LoadGraph(netweights,img_inp,poolingmethod)
		normalized_content = content_img-netmeanpixel
		layer_list = sess.run([netdict[layer] for layer in vgg.VGG19_LAYERS], feed_dict={img_inp:normalized_content})
		content_features = dict(zip(vgg.VGG19_LAYERS,layer_list))#it stores content feature map for every layer
	
	def normed_dot(x):
		assert(len(x.shape)==4)
		print "x.shape:",x.shape,"x.type:",type(x)
		y = np.reshape(x, (-1,x.shape[3]))#keep channel
		return np.matmul(y.T,y)/y.size
	def normed_dot_tf(x):
		_, height, width, channel = map(lambda i: i.value, x.get_shape())
		y = tf.reshape(x,(-1,channel))
		return tf.matmul(tf.transpose(y),y)/(height*width*channel)

	with g.as_default(), g.device('/gpu:0'),tf.Session() as sess:
		#placeholder means it is the "source" of computational graph, we can pass data to it, and all computation starts here
		img_inp = tf.placeholder('float32', shape=style_img.shape)
		netdict = vgg.LoadGraph(netweights,img_inp,poolingmethod)
		normalized_style = style_img-netmeanpixel
		layer_list = sess.run([netdict[layer] for layer in vgg.VGG19_LAYERS], feed_dict={img_inp:normalized_style})
		style_features = dict(zip(vgg.VGG19_LAYERS,layer_list))#it stores content feature map for every layer
		for layer in STYLE_LAYERS:
			style_features[layer] = normed_dot(style_features[layer])


#	for key in content_features.keys():
#		print key,content_features[key].shape
	with tf.Graph().as_default():
		if initial is None:
			initial = tf.random_normal(content_img.shape)*0.256
		assert initial.shape==content_img.shape
		img_inp = tf.Variable(initial)
		netdict = vgg.LoadGraph(netweights,img_inp,poolingmethod)
		
#		print netdict
#		for key in netdict:
#			print key,netdict[key].get_shape(),content_features[key].shape
		#2 content layers, with each weight here
		content_weights = {'relu4_2':CONTENT_WEIGHT_BLEND,'relu5_2':1-CONTENT_WEIGHT_BLEND}
		content_losses = []
		for layer in CONTENT_LAYERS:
			content_losses.append((content_weights[layer])*(tf.nn.l2_loss(netdict[layer]-content_features[layer])/content_features[layer].size))
		content_loss = reduce(tf.add,content_losses)

		style_losses = []
		for i,layer in enumerate(STYLE_LAYERS):
			gram = normed_dot_tf(netdict[layer])
			style_losses.append(STYLE_WEIGHT_BLEND[i]*tf.nn.l2_loss(gram-style_features[layer])/style_features[layer].size)
		style_loss = reduce(tf.add, style_losses)

		tv_y_size = _tensor_size(img_inp[:,1,:,:])
		tv_x_size = _tensor_size(img_inp[:,:,1,:])
		tv_loss = tf.nn.l2_loss(img_inp[:,1:,:,:] - img_inp[:,:-1,:,:])/tv_y_size + tf.nn.l2_loss(img_inp[:,:,1:,:] - img_inp[:,:,:-1,:])/tv_x_size

		#loss = content_loss*CONTENT_WEIGHT
		loss = style_loss*STYLE_WEIGHT + content_loss*CONTENT_WEIGHT + tv_loss*TV_WEIGHT

		train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

		best_loss = float('inf')
		best = None
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(ITERATIONS):
				train_step.run()
				now_loss = loss.eval()
				if now_loss < best_loss:
					best_loss = now_loss
					best = img_inp.eval()
				print "iteration=%d, loss=%f"%(i,now_loss)
		img_out = best.reshape(content_img.shape[1:])+netmeanpixel
		cv2.imwrite(outputfile,img_out)
