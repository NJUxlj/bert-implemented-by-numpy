
import torch
import torch.nn as nn
import numpy as np


#构造一个输入
length = 6
input_dim = 12
hidden_size = 7
x = np.random.random((length, input_dim))


def sigmoid(x):
    return 1/(1 + np.exp(-x))



#使用pytorch的GRU层
torch_gru = nn.GRU(input_dim, hidden_size, batch_first=True)
print("================ pytorch中的GRU的权重形状 ================")
for key, weight in torch_gru.state_dict().items():
    print(key, weight.shape)


#将pytorch的GRU网络权重拿出来，用numpy通过矩阵运算实现GRU的计算
def numpy_gru(x, state_dict):
    weight_ih = state_dict["weight_ih_l0"].numpy() # [3xhidden_size, input_size]
    weight_hh = state_dict["weight_hh_l0"].numpy() # [3xhidden_size, hidden_size]
    bias_ih = state_dict["bias_ih_l0"].numpy() # [3xhidden_size, 1]
    bias_hh = state_dict["bias_hh_l0"].numpy()
    #pytorch将3个门的权重拼接存储，我们将它拆开
    # [hidden_size, input_size]
    w_r_x, w_z_x, w_x = weight_ih[0:hidden_size, :], \
                        weight_ih[hidden_size:hidden_size * 2, :],\
                        weight_ih[hidden_size * 2:hidden_size * 3, :]
    # [hidden_size, hidden_size]
    w_r_h, w_z_h, w_h = weight_hh[0:hidden_size, :], \
                        weight_hh[hidden_size:hidden_size * 2, :], \
                        weight_hh[hidden_size * 2:hidden_size * 3, :]
                        
    # [hidden_size, 1]
    b_r_x, b_z_x, b_x = bias_ih[0:hidden_size], \
                        bias_ih[hidden_size:hidden_size * 2], \
                        bias_ih[hidden_size * 2:hidden_size * 3]
    b_r_h, b_z_h, b_h = bias_hh[0:hidden_size], \
                        bias_hh[hidden_size:hidden_size * 2], \
                        bias_hh[hidden_size * 2:hidden_size * 3]
    w_z = np.concatenate([w_z_h, w_z_x], axis=1) # [hidden_size, hidden_size + input_size]
    w_r = np.concatenate([w_r_h, w_r_x], axis=1) 
    b_z = b_z_h + b_z_x # [1, hidden_size]
    b_r = b_r_h + b_r_x
    
    # 初始化隐单元
    h_t = np.zeros((1, hidden_size))
    sequence_output = []
    for x_t in x: # x_t: (input_dim, )
        x_t = x_t[np.newaxis, :] # [1, input_dim]
        hx = np.concatenate([h_t, x_t], axis=1) # [1, hidden_size + input_dim]
        z_t = sigmoid(np.dot(hx, w_z.T) + b_z) # [1, hidden_size]
        r_t = sigmoid(np.dot(hx, w_r.T) + b_r)
        h = np.tanh(r_t * (np.dot(h_t, w_h.T) + b_h) + np.dot(x_t, w_x.T) + b_x) # [1, hidden_size]
        h_t = (1 - z_t) * h + z_t * h_t # [1, hidden_size]
        sequence_output.append(h_t) # [max_len, hidden_size]
    return np.array(sequence_output), h_t

torch_sequence_output, torch_h = torch_gru(torch.Tensor([x]))
numpy_sequence_output, numpy_h = numpy_gru(x, torch_gru.state_dict())

print("============================ GRU sequence output =========================")
print(torch_sequence_output)
print(numpy_sequence_output)
print("--------")
print("============================ GRU hidden unit output ============================")
print(torch_h)
print(numpy_h)
