# 밑바닥 딥러닝 교재 깃허브 코드를 가져와 수정했다
# https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch07/simple_convnet.py

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict

from layers import *

# numerical_gradient 함수 <- 역전파로 얻은 결과와 비교용
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad



class ConvNet:
    """우리 프로젝트의 합성곱 신경망
    
    Conv-BatchNorm-ReLU-Pooling-
    Conv-BatchNorm-ReLU-Pooling- # Pool2는 flatten도 겸한다
    Dropout-
    Affine-BatchNorm-ReLU
    Affine-Softmax
    
    Parameters
    ----------
    input_size : 입력 크기
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'는 'He 초깃값'으로 설정
    """

    def __init__(self, input_dim, output_size, weight_init_std):
        
        # 합성곱 계층 레이어 하이퍼 파라미터
        conv1_param = {'filter_num': 8, 'filter_size': 4, 'pad': 0, 'stride': 4}
        conv2_param = {'filter_num': 16, 'filter_size': 4, 'pad': 0, 'stride': 4}

        # 여러가지 사이즈 변화에 따른 변수들 미리 계산
        # 풀링 사이즈는 2x2, stride 2인건 계층에 직접 적는다
        input_size = input_dim[1]
        filter_num1 = conv1_param['filter_num']
        filter_size1 = conv1_param['filter_size']
        filter_pad1 = conv1_param['pad']
        filter_stride1 = conv1_param['stride']
        
        conv1_output_size = int( (input_size - filter_size1 + 2*filter_pad1) / filter_stride1 + 1 )
        pool1_output_size = int(conv1_output_size/2)

        filter_num2 = conv2_param['filter_num']
        filter_size2 = conv2_param['filter_size']
        filter_pad2 = conv2_param['pad']
        filter_stride2 = conv2_param['stride']

        conv2_output_size = int( (pool1_output_size - filter_size2 + 2*filter_pad2) / filter_stride2 + 1 )
        pool2_output_size = int(filter_num2 * (conv2_output_size/2) * (conv2_output_size/2))  # pool2는 flatten도 겸하는 역할로 노드 갯수 계산
        
        hidden_size = int(pool2_output_size/2)

        # 파라미터 분산값 기본치
        # ReLU 가중치 초기화 시 앞 노드 개수 n일때 np.sqrt(2/n) = 'He 초기값'이 좋다
        weight_init_std = 0.01
        weight_std1 = np.sqrt(2 / (input_dim[0] * input_size * input_size))
        weight_std2 = np.sqrt(2 / (filter_num1 * pool1_output_size * pool1_output_size))
        weight_std3 = np.sqrt(2 / pool2_output_size)
        weight_std4 = np.sqrt(2 / hidden_size)

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_std1 * \
                            np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)

        self.params['W2'] = weight_std2 * \
                            np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2)
        self.params['b2'] = np.zeros(filter_num2)

        self.params['W3'] = weight_std3 * \
                            np.random.randn(pool2_output_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)

        self.params['W4'] = weight_std4 * \
                            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)

        # hidden_neuron_num1 = int(filter_num1 * conv1_output_size * conv1_output_size)
        self.params['gamma1'] = np.ones(32768)
        self.params['beta1'] = np.zeros(32768)

        # hidden_neuron_num2 = int(filter_num2 * conv2_output_size * conv2_output_size)
        self.params['gamma2'] = np.ones(1024)
        self.params['beta2'] = np.zeros(1024)

        # hidden_size = int(pool2_output_size/2)
        self.params['gamma3'] = np.ones(128)
        self.params['beta3'] = np.zeros(128)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv1_param['stride'], conv1_param['pad'])
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv2_param['stride'], conv2_param['pad'])
        self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Dropout'] = Dropout(0.3) # 드롭 아웃, Ratio = 0.3

        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):  # 사용하지 않음. 확인용으로만 사용.
        """기울기를 구한다（수치미분）.
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).
        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        grads['gamma1'], grads['beta1'] = self.layers['BatchNorm1'].dgamma, self.layers['BatchNorm1'].dbeta
        grads['gamma2'], grads['beta2'] = self.layers['BatchNorm2'].dgamma, self.layers['BatchNorm2'].dbeta
        grads['gamma3'], grads['beta3'] = self.layers['BatchNorm3'].dgamma, self.layers['BatchNorm3'].dbeta

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
        for i, key in enumerate(['BatchNorm1', 'BatchNorm2', 'BatchNorm3']):
            self.layers[key].gamma = self.params['gamma' + str(i+1)]
            self.layers[key].beta = self.params['beta' + str(i+1)]