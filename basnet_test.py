import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet
import sys

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_roi(image_name,pred,dir):
	image = cv2.imread(image_name)
	predict = pred.squeeze()
	predict = predict.cpu().data.numpy()
	predict = cv2.resize(predict,(image.shape[1],image.shape[0]))
	cv2.threshold(predict, 0.001, 1.0, cv2.THRESH_BINARY,predict)
	kernel = np.ones((10, 10), np.uint8)
	cv2.dilate(predict, kernel,predict,iterations=3)
	predict = np.tile(predict[:,:,np.newaxis], (1,1,3))
	image = np.multiply(image,predict)
	# cv2.imshow("m",predict)
	# cv2.imshow("img",image)
	# cv2.waitKey()
	img_name = image_name.split("/")[-1]
	img_name = img_name[0:-4]+"_mask"+img_name[-4:]
	cv2.imwrite(dir+'/'+img_name,predict*255)


def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')


if len(sys.argv) < 3:
	print("Usage: python3 basenet_test input_dir out_dir\n")
	sys.exit(1)
image_dir = sys.argv[1]
prediction_dir = sys.argv[2]

print("image_dir ", image_dir)
print("prediction_dir ", prediction_dir)
# --------- 1. get image path and name ---------
model_dir = '/home/dojing/SFM/BASNet/saved_models/basnet.pth'
img_name_list = []
for ext in ('/*.gif', '/*.png', '/*.jpg', '/*.JPG'):
	img_name_list.extend(glob.glob(image_dir + ext))



# --------- 2. dataloader ---------
#1. dataload
test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)

# --------- 3. model define ---------
print("...load BASNet...")
net = BASNet(3,1)
net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
	net.cuda()
net.eval()

# --------- 4. inference for each image ---------
for i_test, data_test in enumerate(test_salobj_dataloader):

	print("inferencing:",img_name_list[i_test].split("/")[-1])

	inputs_test = data_test['image']
	inputs_test = inputs_test.type(torch.FloatTensor)

	if torch.cuda.is_available():
		inputs_test = Variable(inputs_test.cuda())
	else:
		inputs_test = Variable(inputs_test)

	d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)

	# normalization
	pred = d1[:,0,:,:]
	pred = normPRED(pred)

	# save results to test_results folder
	save_roi(img_name_list[i_test],pred,prediction_dir)

	del d1,d2,d3,d4,d5,d6,d7,d8
