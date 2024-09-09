

import torch
import torch.nn as nn
import numpy as np


#构造一个输入
length = 6
input_dim = 12
hidden_size = 7
x = np.random.random((length, input_dim))
# print(x)

#使用pytorch的lstm层
torch_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
print("============ pytorch中的LSTM的权重形状 ==============")
for key, weight in torch_lstm.state_dict().items():
    print(key, weight.shape)



def sigmoid(x):
    return 1/(1 + np.exp(-x))

#将pytorch的lstm网络权重拿出来，用numpy通过矩阵运算实现lstm的计算
def numpy_lstm(x, state_dict):
    # [28, 12] = [7x4, 12] , 4个门的input->hidden的权重矩阵垂直叠加
    # [4xhidden_size, input_size]
    weight_ih = state_dict["weight_ih_l0"].numpy() # ih: 输入到隐藏层的权重
    # [28, 7] = [7x4, 7] 
    # [4xhidden_size, hidden_size]
    weight_hh = state_dict["weight_hh_l0"].numpy()
    # [28, 1]
    bias_ih = state_dict["bias_ih_l0"].numpy()
    # [28, 1]
    bias_hh = state_dict["bias_hh_l0"].numpy()
    
    #pytorch将四个门的权重拼接存储，我们将它拆开
    # i:输入门，f：遗忘门，c:记忆门，o:输出门
    
    #  [hidden_size, input_size]
    w_i_x, w_f_x, w_c_x, w_o_x = weight_ih[0:hidden_size, :], \
                                 weight_ih[hidden_size:hidden_size*2, :],\
                                 weight_ih[hidden_size*2:hidden_size*3, :],\
                                 weight_ih[hidden_size*3:hidden_size*4, :]
    # [hidden_size, hidden_size]
    w_i_h, w_f_h, w_c_h, w_o_h = weight_hh[0:hidden_size, :], \
                                 weight_hh[hidden_size:hidden_size * 2, :], \
                                 weight_hh[hidden_size * 2:hidden_size * 3, :], \
                                 weight_hh[hidden_size * 3:hidden_size * 4, :]
                                 
    # [hidden_size, 1]
    b_i_x, b_f_x, b_c_x, b_o_x = bias_ih[0:hidden_size], \
                                 bias_ih[hidden_size:hidden_size * 2], \
                                 bias_ih[hidden_size * 2:hidden_size * 3], \
                                 bias_ih[hidden_size * 3:hidden_size * 4]
    # [hidden_size, 1]
    b_i_h, b_f_h, b_c_h, b_o_h = bias_hh[0:hidden_size], \
                                 bias_hh[hidden_size:hidden_size * 2], \
                                 bias_hh[hidden_size * 2:hidden_size * 3], \
                                 bias_hh[hidden_size * 3:hidden_size * 4]
    # 沿着列方向（轴为1）进行合并
    w_i = np.concatenate([w_i_h, w_i_x], axis=1) # [hidden_size, hidden_size + input_size]
    w_f = np.concatenate([w_f_h, w_f_x], axis=1) # [hidden_size, hidden_size + input_size]
    w_c = np.concatenate([w_c_h, w_c_x], axis=1) # [hidden_size, hidden_size + input_size]
    w_o = np.concatenate([w_o_h, w_o_x], axis=1) # [hidden_size, hidden_size + input_size]
    b_f = b_f_h + b_f_x # [1, hidden_size]
    b_i = b_i_h + b_i_x
    b_c = b_c_h + b_c_x
    b_o = b_o_h + b_o_x
    # 初始化记忆单元和隐单元
    c_t = np.zeros((1, hidden_size))
    h_t = np.zeros((1, hidden_size))
    sequence_output = []
    for x_t in x: # x_t: [1, input_dim]
        # 为 x_t 增加一个维度
        print("old x_t.shape = ", x_t.shape) # (12,)
        x_t = x_t[np.newaxis, :]
        print("x_t.shape = ", x_t.shape) # (12, 1)
        hx = np.concatenate([h_t, x_t], axis=1) # [1, hidden_size + input_dim]
        
        # f_t = sigmoid(np.dot(x_t, w_f_x.T) + b_f_x + np.dot(h_t, w_f_h.T) + b_f_h)
        f_t = sigmoid(np.dot(hx, w_f.T) + b_f) # [1, hidden_size]
        
        # i_t = sigmoid(np.dot(x_t, w_i_x.T) + b_i_x + np.dot(h_t, w_i_h.T) + b_i_h)
        i_t = sigmoid(np.dot(hx, w_i.T) + b_i) # 
        
        # g = np.tanh(np.dot(x_t, w_c_x.T) + b_c_x + np.dot(h_t, w_c_h.T) + b_c_h)
        g = np.tanh(np.dot(hx, w_c.T) + b_c) # [1, hidden_size]
        c_t = f_t * c_t + i_t * g # [1, hidden_size]
        
        # o_t = sigmoid(np.dot(x_t, w_o_x.T) + b_o_x + np.dot(h_t, w_o_h.T) + b_o_h)
        o_t = sigmoid(np.dot(hx, w_o.T) + b_o)
        h_t = o_t * np.tanh(c_t) # [1, hidden_size]
        sequence_output.append(h_t) # [max_len, hidden_size]
    return np.array(sequence_output), (h_t, c_t)


torch_sequence_output, (torch_h, torch_c) = torch_lstm(torch.Tensor([x]))
numpy_sequence_output, (numpy_h, numpy_c) = numpy_lstm(x, torch_lstm.state_dict())

print("================= sequence output =========================")
print(torch_sequence_output)
print(numpy_sequence_output)
print("--------")
print("================== hidden unit output ====================")
print(torch_h)
print(numpy_h)
print("--------")
print(" ===================== memory cell output ======================")
print(torch_c)
print(numpy_c)
