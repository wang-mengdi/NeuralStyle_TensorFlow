import stylize
import cv2
import sys

if __name__=='__main__':
	contentfile, stylefile, outfile = sys.argv[1:4]
	contentweight = float(sys.argv[4])
	styleweight = 1.0-contentweight
	content_img = cv2.resize(cv2.imread(contentfile),(360,480))[None,:,:,:]
	#content_img = cv2.imread('1-content.jpg')[None,:,:,:]
	style_img = cv2.imread(stylefile)[None,:,:,:]
	stylize.stylize(
		network_path='imagenet-vgg-verydeep-19.mat',
		content_img=content_img,
		style_img=style_img,
		initial=None,
		poolingmethod='max',
		outputfile = outfile,
		CONTENT_WEIGHT=contentweight,
		STYLE_WEIGHT=styleweight
	)
