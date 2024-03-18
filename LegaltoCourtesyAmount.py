import torch
from skimage.io import imread, imshow,imsave
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.utils as image
import cv2
import glob
import os
import tensorflow_addons as tfa



color = (255, 0, 0)
def legelcrop(image_file,ou_path):
	model = torch.hub.load('ultralytics/yolov5','custom',path='models/LegalAmounWordDetectionv4')
	model.iou = 0.30
	results = model(image_file)
	results.print() # or .show(), .save(), .crop(), .pandas(), etc.
	img = imread(image_file)
	im_2=img
	rslt=results.pandas().xyxy[0]  # im predictions (pandas)
	rslt.sort_values(by=['xmin'], inplace=True,ascending=False,ignore_index=True)
	if not os.path.exists(ou_path):
		os.makedirs(ou_path)
	for i in range(0,len(rslt)):
		x=img[int(rslt["ymin"][i]):int(rslt["ymax"][i]),int(rslt["xmin"][i]):int(rslt["xmax"][i])]
		start_point = (int(rslt["xmin"][i]),int(rslt["ymin"][i]))
		end_point = ( int(rslt["xmax"][i]),int(rslt["ymax"][i]))

		nm=ou_path+"/"+str(i)+".jpg"
		new_img = Image.fromarray(x)
		new_img.save(nm)
		im_2 = cv2.rectangle(im_2, start_point, end_point, color,2)
	return im_2

def prediction(img_path):
  learning_rate = 1e-4
  optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)
  new_model = tf.keras.models.load_model('models/ArabicWordNetv2.h5', compile=False)
  new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array=img_array/255
  img_batch = np.expand_dims(img_array, axis=0)
  images = np.vstack([img_batch])
  prediction = new_model.predict(images)
  res = np.argmax(prediction)
  dict1={0:8 , 1:800 , 2:80, 3:50, 4:5, 5:500, 6:40,7: 4, 8:400, 9:100,10: 1000000, 11:9,12: 900, 13:90, 14:1, 15:'Only', 16:'Reyal',17: 7,18: 700, 19:70, 20:6, 21:600, 22:60, 23:10, 24:30, 25:1000, 26:3, 27:300, 28:20, 29:2, 30:200, 31:2000000, 32:2000}
  #dict1={0:'Eight' , 1:'Eight Hundred' , 2:'Eighty', 3:'Fifty', 4:'Five', 5:'Five Hundred', 6:'Forty',7: 'Four', 8:'Four Hundred', 9:'Hundred',10: 'Million', 11:'Nine',12: 'Nine Hundred', 13:'Ninety', 14:'One', 15:'Only', 16:'Reyal',17: 'Seven',18: 'Seven Hundred', 19:'Seventy', 20:'Six', 21:'Six Hundred', 22:'Sixty', 23:'Ten', 24:'Thirty', 25:'Thousand', 26:'Three', 27:'Three Hundred', 28:'Twenty', 29:'Two', 30:'Two Hundred', 31:'Two Million', 32:'Two Thousand'}
  #title="The predicted output is :"+str(dict1[res])
  #plt.title(title)
  #plt.imshow(cv2.imread(img_path))
  #plt.show()
  #print(img_path)
  #print("The predicted output is :"+str(dict1[res]))
  return dict1[res]


def find_val(unit_matrix,typ):
  if typ==1:
    if len(unit_matrix)==0:
      val=0
    elif len(unit_matrix)==4:
      val=(unit_matrix[0]+(unit_matrix[1]+unit_matrix[2]))*unit_matrix[3]
    elif len(unit_matrix)==3:
      val=(unit_matrix[0]+unit_matrix[1])*unit_matrix[2]
    elif len(unit_matrix)==2:
      val=unit_matrix[0]*unit_matrix[1]
    else:
      val=unit_matrix[0]
  else:
    if len(unit_matrix)==1:
      val=unit_matrix[0]
    elif len(unit_matrix)==2:
      val=unit_matrix[0]+unit_matrix[1]
    elif len(unit_matrix)==3:
      val=unit_matrix[0]+(unit_matrix[1]+unit_matrix[2])
    else:
      val=0
  return val

def find_Parts_val(unit_matrix,unit):
  part1=[]
  part2=[]
  part3=[]
  part1_val=0
  part2_val=0
  part3_val=0
  for j in range(0,2):
    strt_indx=0
    lst_indx=-1
    if(j==0):
      for i in range(0,len(unit_matrix)):
        if(unit_matrix[i]>=100 and unit_matrix[i]<unit):
          lst_indx=i+1
          part1=unit_matrix[strt_indx:lst_indx]
          del unit_matrix[strt_indx:lst_indx]
          break
    elif(j==1):
      for i in range(0,len(unit_matrix)):
        if(unit_matrix[i]==unit):
          part3_val=unit
          del unit_matrix[i]
          part2=unit_matrix
          break
  if(len(part1)==2):
    part1_val=part1[0]*part1[1]
  elif(len(part1)==1):
     part1_val=part1[0]
  if(len(part2)==2):
    part2_val=part2[0]+part2[1]
  elif(len(part2)==1):
     part2_val=part2[0]
  if(part1_val==0 and part2_val==0):
    val=part3_val
  else:
    val=(part1_val+part2_val)*part3_val
  return val

def get_courtesy_from_legal(path):
  y=[]
  files = glob.glob(path)
  for file in files:
    pred=prediction(file)
    if(pred=="Reyal" or pred=="Only"):
      break
    else:
      y.append(pred)
  mlim_part_vect=[]
  thsnd_part_vect=[]
  hndrd_part_vect=[]
  ons_part_vect=[0]
  ####################################################
  mlim_part_val=0
  thsnd_part_val=0
  hndrd_part_val=0
  ons_part_val=0
  ##################################################
  x=y
  print("orginal x=",x)
  for j in range(0,3):
    find=0
    strt_indx=0
    lst_indx=-1
    if(j==0):
      for i in range(0,len(x)):
         if x[i]>=1000000:
          lst_indx=i+1
          mlim_part_vect=x[strt_indx:lst_indx]
          del x[strt_indx:lst_indx]
          break
    if(j==1):
      for i in range(0,len(x)):
         if x[i]>=1000:
           lst_indx=i+1
           thsnd_part_vect=x[strt_indx:lst_indx]
           del x[strt_indx:lst_indx]
           break
    if(j==2):
      for i in range(0,len(x)):
        if x[i]==100:
          lst_indx=i+1
          hndrd_part_vect=x[strt_indx:lst_indx]
          del x[strt_indx:lst_indx]
          break
    else:
      ons_part_vect=x

  mlim_part_val=find_Parts_val(mlim_part_vect,1000000)
  thsnd_part_val=find_Parts_val(thsnd_part_vect,1000)
  hndrd_part_val=find_val(hndrd_part_vect,1)
  ons_part_val=find_val(ons_part_vect,0)
  total=mlim_part_val+thsnd_part_val+hndrd_part_val+ons_part_val
  return total

