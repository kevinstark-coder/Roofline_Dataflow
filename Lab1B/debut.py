import pandas as pd
import csv
from system import *
from analye_model import *
from plot_rooflines import *
from operators import RELU, ADD, GEMM, CONV2D
pd.set_option('display.float_format', lambda x: '%.2f' % x)

## Only run this once you have completed code in operators.py
relu1 = RELU([2, 256])
relu2 = RELU([254, 5])

add1 = ADD([2, 8, 512])
add2 = ADD([8, 128, 1024])

gemm1 = GEMM([32, 8, 16, 32])
gemm2 = GEMM([2, 128, 128, 128])

conv1 = CONV2D([3, 256, 96, 128, 128 , 5, 5])
conv2 = CONV2D([1, 256, 384, 12, 12 , 3, 3])

with open('output_a1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['relu1', list(relu1.get_tensors()), relu1.get_num_ops()])
    writer.writerow(['relu2', list(relu2.get_tensors()), relu2.get_num_ops()])
    writer.writerow(['add1',  list(add1.get_tensors()), add1.get_num_ops()])
    writer.writerow(['add2',  list(add2.get_tensors()), add2.get_num_ops()])
    writer.writerow(['gemm1', list(gemm1.get_tensors()), gemm1.get_num_ops()])
    writer.writerow(['gemm2', list(gemm2.get_tensors()), gemm2.get_num_ops()])
    writer.writerow(['conv1', list(conv1.get_tensors()), conv1.get_num_ops()])
    writer.writerow(['conv2', list(conv2.get_tensors()), conv2.get_num_ops()])

    ## A100 https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
A100_GPU = System( offchip_mem_bw=1935,
                   flops=312, frequency=1095 ,
                   compute_efficiency=0.75, memory_efficiency=0.7)
## https://developer.nvidia.com/embedded/jetson-modules
jetson_nano = System( offchip_mem_bw=34, 
                 flops=20, frequency=625, 
                 compute_efficiency=0.85, memory_efficiency=0.75  )
example_network = [relu1, relu2, add1, add2, gemm1, gemm2, conv1, conv2]
model_df = analysis_model(example_network, A100_GPU)
model_df.to_csv('output_a2.csv', index=False)

def alexnet(batch_size):
    ## Fill in the opertors of alexnet, please refer to the figure in pdf document.
    ## Refer the example_network to follow the network declaration
    model_arch = [         
        #B, K, C, X, Y, R, S
                    CONV2D([batch_size, 96,3,224,224,11,11]),
                    RELU([batch_size, 96*54*54]),
                    CONV2D([batch_size, 256,96,54,54,5,5]),
                    RELU([batch_size, 256*26*26]),
                    CONV2D([batch_size, 384,256,26,26,3,3]),
                    RELU([batch_size, 384*12*12]),
                    CONV2D([batch_size, 384,384,12,12,3,3]),
                    RELU([batch_size, 384*12*12]),
                    CONV2D([batch_size, 256,384,12,12,3,3]),
                    RELU([batch_size, 256*12*12]),
        #B, M, N, K
                    GEMM([batch_size,1,4096,6400]),
                    GEMM([batch_size,1,4096,4096]),
                    GEMM([batch_size,1,1000,4096]),
                 ]
    return model_arch

def bert(batch_size):
    ## Fill in the opertors of bert, please refer to the figure in pdf document.
    ## Refer the example_network to follow the network declaration
    model_arch = [
        #B, M, N, K
        GEMM([batch_size,256,2304,768]),
        GEMM([batch_size,256,256,768]),
        GEMM([batch_size,256,768,256]),
        GEMM([batch_size,256,768,768]),
        #B, X, Y
        ADD([batch_size,256,768]),
        GEMM([batch_size,256,3072,768]),
        RELU([batch_size,256*3072 ]),
        GEMM([batch_size,256,768,3072]),
        ADD([batch_size,256,768]),
    ]
    return model_arch


alexnet_on_a100_df = analysis_model(alexnet(256), A100_GPU)
# display(alexnet_on_a100_df)

dot_roofline(alexnet_on_a100_df, A100_GPU)
print(f'Total Cycles:{sum(alexnet_on_a100_df.loc[:, "Cycles"]):0.2f}, Total data (MB): {sum(alexnet_on_a100_df.loc[:, "Total Data (MB)"]):0.2f}')