# coding: utf-8
# 피클 짜개

# 추가로 할 수 있는 것 : validation data를 이용한 하이퍼파라미터 값 찾기(교재 p. 221)

# 모듈 임포트

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
import bz2  # 피클 압축용. 압축 안하니 수 GB가 넘어갔다.
import tensorflow as tf
import glob



# 피클 짜기 시작

Arborio = glob.glob('../Rice_Image_Dataset/Arborio/*.*')
Basmati = glob.glob('../Rice_Image_Dataset/Basmati/*.*')
Ipsala = glob.glob('../Rice_Image_Dataset/Ipsala/*.*')
Jasmine = glob.glob('../Rice_Image_Dataset/Jasmine/*.*')
Karacadag = glob.glob('../Rice_Image_Dataset/Karacadag/*.*')

data = []
labels = []

for i in Arborio:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append([1, 0, 0, 0, 0])
for i in Basmati:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append([0, 1, 0, 0, 0])
for i in Ipsala:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append([0, 0, 1, 0, 0])
for i in Jasmine:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append([0, 0, 0, 1, 0])
for i in Karacadag:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append([0, 0, 0, 0, 1])

data = np.array(data)
labels = np.array(labels)

# 확인
print(data.shape)
print(labels.shape)



# 피클로 저장 

ofile = bz2.BZ2File("data.pickle",'wb')
pickle.dump(data, ofile)
ofile.close()

ofile = bz2.BZ2File("labels.pickle",'wb')
pickle.dump(labels, ofile)
ofile.close()
##############################################


# 피클 데이터 읽어오기 - 읽은 후 바로 학습으로 넘어가시오!! 데이터 읽기는 피클 있으니 무시!!
 
ifile = bz2.BZ2File("data.pickle",'rb')
data = pickle.load(ifile)
ifile.close()

ifile = bz2.BZ2File("labels.pickle",'rb')
labels = pickle.load(ifile)
ifile.close()

# 확인

print(data.shape)
print(labels.shape)