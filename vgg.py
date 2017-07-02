import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def ConstConv2D(x, weight, bias, stride=1):
	#stride is the stride of convolution layer
	#It's something like Wx+b
	#4 dimensions are: [batchsize,height,weight,channels]
	#stride is applied for height and weight
	#bias is only 1-D, it should be somehow broadcasted
	conv = tf.nn.conv2d(x,tf.constant(weight), strides=(1,stride,stride,1), padding='SAME')
	return tf.nn.bias_add(conv,bias)
	return tf.nn.bias_add(tf.nn.conv2d(x,tf.constant(weight), strides=(1,stride,stride,1), padding='SAME'),bias)

def Pool2D(x, method='avg', kernel=2, stride=2):
	#kernel=2 means it pools for every 2*2 submatrix for height and weight
	#like stride, it should be [1,kernel,kernel,1]
	#here are 2 types of pooling: average pooling/max pooling
	#average means take the average of 2*2 submatrix
	#max means take the maximum of 2*2 submatrix
	if method=='avg':
		return tf.nn.avg_pool(x,ksize=(1,kernel,kernel,1),strides=(1,stride,stride,1),padding='SAME')
	else:
		return tf.nn.max_pool(x,ksize=(1,kernel,kernel,1),strides=(1,stride,stride,1),padding='SAME')

def LoadGraph(weights, img_inp, poolingmethod):
	#in netdict we save the tensor operator for every layer
	print "load graph: input shape = %s"%img_inp.get_shape()
	netdict = {}
	x = img_inp#input image
	print "type x:",type(x)
	for i,name in enumerate(VGG19_LAYERS):
		layertype = name[:4]
		if layertype=='conv':
			W,b = weights[i][0][0][0][0]
			#the data loaded from .mat by scipy is formed as [width,height,batchsize,channel]
			#but in tensorflow it should be: [batchsize,height,width,channel]
			W = np.transpose(W,(1,0,2,3))
			b = b.reshape(-1) #-1 means automatical shape
			#print W.dtype,b.dtype,x.dtype
			x = ConstConv2D(x, W, b)
		elif layertype=='relu':
			x = tf.nn.relu(x)#relu is non-linear activation function used here
		elif layertype=='pool':
			x = Pool2D(x,poolingmethod)
		netdict[name]=x
		#for example, netdict['conv1_1'] is the op of layer conv1_1
	assert len(netdict) == len(VGG19_LAYERS)
	return netdict

def LoadMat(path):
	data = scipy.io.loadmat(path)
	#data is a dict, include: 'layers', '__header__', '__globals__', 'classes', '__version__', 'normalization'
	#it's somehow complicated, but we only concern about layers and normalization
	#the [0][0][0] comes from some built-in class from the MATLAB matrix type, we don't need to study into it
	mean = data['normalization'][0][0][0]
	mean_pixel = np.mean(mean,axis=(0,1))
	weights = data['layers'][0]
	return weights, mean_pixel
