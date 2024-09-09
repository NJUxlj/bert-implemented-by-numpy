import numpy as np

import torch

import torch.nn as nn

input_dim = 6
hidden_size = 8 # 卷积核个数  C_{out}
kernel_size = 2 

torch_cnn1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_size, kernel_size=kernel_size)


for key, weight in torch_cnn1d.state_dict().items():
    print(key, weight.shape)

# 构造输入
x = torch.randn(6,10) # input_size * max_len


def numpy_cnn1d(x, state_dict):
    '''
    
    '''
    weight = state_dict['weight'].numpy() # hidden_size, input_size, kernel_size
    bias = state_dict['bias'].numpy()

    kernels_result = [] # 
    # 遍历所有卷积步
    for step in range(len(x[1])-kernel_size+1):
        kernels_result_step = []
        # 遍历取kernel
        for kernel in weight:
            window = x[:, step:step+kernel_size]
            dot_product = np.sum(window*kernel)
            
            # kernel输出向量    每个卷积步加入所有kernel输出的一个分量, 一个分量 就是一个点积
            kernels_result_step.append(dot_product)
        kernels_result.append(np.array(kernels_result_step) + bias)
        
    return np.array(kernels_result).T # [num_kernel, num_step]
        
    
    

torch_cnn1d_output = torch_cnn1d(x.unsqueeze(0))   
print("torch_cnn1d_output = \n", torch_cnn1d_output)
print("torch_cnn1d_output.shape = ", torch_cnn1d_output.shape)



numpy_cnn1d_output = numpy_cnn1d(x.numpy(), torch_cnn1d.state_dict())   
print("numpy_cnn1d_output = \n", numpy_cnn1d_output)
print("numpy_cnn1d_output.shape = ", numpy_cnn1d_output.shape)
