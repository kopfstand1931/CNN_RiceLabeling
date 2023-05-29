# 실제 학습 이루어지는 코드

# 모든 코드의 많은 부분은 밑바닥부터 시작하는 딥러닝 교재의 참조 코드 깃허브를 가져와 수정했습니다.
# https://github.com/WegraLee/deep-learning-from-scratch

# 모듈 임포트

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import gc  # 메모리 반환 위해 가비지 콜렉터 호출용으로 가져옴
import numpy as np
import matplotlib.pyplot as plt
import pickle
import bz2  # 피클 압축용. 압축 안하니 수 GB가 넘어갔다.
from sklearn.model_selection import train_test_split    # validation data와 test data, training data를 분리하기 위해 사용했다.
from our_cnn import ConvNet
from trainer import Trainer


# In[ ]:


# 피클 데이터 읽어오기 - 읽은 후 바로 학습으로 넘어가시오!! 데이터 읽기는 피클 있으니 무시!!
# 7,500개 약 21초
# 15,000개 약 30초 ~ 1분
 
ifile = bz2.BZ2File("data.pickle",'rb')
data = pickle.load(ifile)
ifile.close()

ifile = bz2.BZ2File("labels.pickle",'rb')
labels = pickle.load(ifile)
ifile.close()



print(data.shape)
print(labels.shape)

# train, test, valid의 3개 세트를 6:2:2로 나눈다.
x_temp, x_valid, t_temp, t_valid = train_test_split(data, labels, test_size=0.2, random_state=84)   # 무작위 시드값
x_train, x_test, t_train, t_test = train_test_split(x_temp, t_temp, test_size=0.25, random_state=72)   # 무작위 시드값


# In[ ]:


# 임시 변수들 메모리 반환
del data
del labels
del x_temp
del t_temp
gc.collect()


# In[ ]:


# input 데이터 차원 순서 가공
print(x_train.shape)    #채널이 가장 마지막에 오는 (N, H, W, C)
print(x_test.shape)
print(x_valid.shape)

x_train = x_train.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)
x_valid = x_valid.transpose(0, 3, 1, 2)

print(x_train.shape)    #채널이 처음에 오는 (N, C, H, W)
print(x_test.shape)
print(x_valid.shape)

print(t_train.shape)
print(t_test.shape)
print(t_valid.shape)


# In[ ]:


# 하이퍼 파라미터 결정
# validation dataset을 사용한 하이퍼 파라미터 lr 결정
"""
max_epochs = 5  # 작은 에포크
lr = 10 ** np.random.uniform(low=-6, high=-2, size=50)  # 10-6~10-2 까지 무작위 lr 50개
# lr = 10 ** np.random.uniform(low=-2.05, high=-1.5, size=5)  # 10-2.05(0.0089)~10-1.5(0.0316) 까지 무작위 lr 5개
iteration_num = 0

for current_lr in lr :                         
    iteration_num += 1
    network = ConvNet(input_dim=(3,256,256), output_size=5, weight_init_std=0.01)
    trainer = Trainer(network, x_valid, t_valid, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': current_lr},
                  evaluate_sample_num_per_epoch=None, verbose=False)
    print("<<< " + str(iteration_num) + ". Learning Rate : " + str(current_lr) + " >>>")
    trainer.train()

    # 그래프 플로팅
    # train과 test acc 비교 플로팅
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list[0:5], marker='o', label='validation', markevery=1)
    plt.plot(x, trainer.test_acc_list[0:5], marker='s', label='test', markevery=1)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.title("Accuracy with Learning Rate : " + str(current_lr))
    plt.show()


# In[ ]:
"""

# 학습
max_epochs = 2

# stride가 너무 작은 경우 im2col에서 메모리 오류가 나서, 교재 p.234) 출력 크기가 정수인 1보다 큰 스트라이드 값을 선택했다.
network = ConvNet(input_dim=(3,256,256), output_size=5, weight_init_std=0.01)

# 이전 학습된 파라미터 로드해 이어서 학습하는 경우
# network.load_params("params.pkl")
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.0089469825823409},
                  evaluate_sample_num_per_epoch=None, verbose=False)
trainer.train()

# 매개변수 보존
network.save_params("params.pkl")
print("Saved Network Parameters!")



# 그래프 그리기

# train과 test acc 비교 플로팅
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list[0:2], marker='o', label='train', markevery=1)
plt.plot(x, trainer.test_acc_list[0:2], marker='s', label='test', markevery=1)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Accuracy")
plt.show()

# loss 플로팅. trainer.train_loss_list 사용.
x = np.arange(trainer.max_iter)
plt.plot(x, trainer.train_loss_list)
plt.xlabel("Num of iteration")
plt.ylabel("loss")
plt.title("Loss function")
plt.show()


# In[ ]:


# 학습된 모델 파라미터 로드 확인

network = ConvNet(input_dim=(3,256,256), output_size=5, weight_init_std=0.01)

# 파라미터 로드
network.load_params("params.pkl")
# 정확도 확인
print(network.accuracy(x_test, t_test))



