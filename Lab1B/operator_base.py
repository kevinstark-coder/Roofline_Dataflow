import numpy as np
from unit import Unit


class Operator(object):
    def __init__(self, dim):
        self.dim = dim
        self.input_a, self.input_w, self.output = self.get_tensors()
        self.num_ops = self.get_num_ops()
    
    def set_tensor(self, input_a=None, input_w=None, output=None):
        if input_a is not None:
            self.input_a = input_a
        if input_w is not None:
            self.input_w = input_w
        if output is not None:
            self.output = output


    def get_op_type(self):
        return self.op_type

    def get_tensors(self):
        pass

    def get_num_ops(self):
        pass

    def get_effective_dim_len(self):
        pass


    ################################################################
    ## TODO A.2.i
    ################################################################
    # Use number of operations for given operator and system parameters to determine the compute time.
    def get_ideal_compute_time(self, system, data_format = None, compute = None):
        number_of_ops = self.get_num_ops()
        if(data_format == 'int8' and (compute) ):
            op_per_sec = system.op_per_sec *2
        elif(data_format == 'fp32' and (compute)):
            op_per_sec = system.op_per_sec/2
        elif(data_format == 'fp64' and (compute)):
            op_per_sec = system.op_per_sec/4
        else:
            op_per_sec = system.op_per_sec

        compute_time = number_of_ops/op_per_sec/system.compute_efficiency
        return compute_time 

    ################################################################
    ## TODO A.2.ii
    ################################################################
    # Use number of elements for given operator and system parameters to determine the memory time.
    def get_ideal_memory_time(self, system, read = True, write = True,  Tm = None, Tk = None, Tn = None,tiling = None, data_format = None):
        ## Number of elements
        input_a, input_b, output = self.get_tensors()
        
        ## Assume data format of BF16 for all both inputs and outputs.
# Size of each tile of A  = 1024 x 2048   
# Size of each tile of B = 2048 x 512
# Size of each Output tile = 1024 x 512
        if(tiling == 'A'):
            B, M, N, K = self.dim[:self.get_effective_dim_len()]
            input_a =  B* M * K                                 #Tm* Tk * M * K / (Tk* Tm)
            input_b =  Tm * N * K                             #Tm * Tk * Tn * N * K / (Tk* Tn)
            output =  B* 2 * Tk * M * N                             # 2*Tm * Tk * Tn * M * N / (Tm* Tn)
        elif(tiling == 'B'):
            B, M, N, K = self.dim[:self.get_effective_dim_len()]
            input_a =  B* Tn * M * K                                 #Tm* Tk *Tn * M * K / (Tk* Tm)
            input_b =  N * K                             #Tk * Tn * N * K / (Tk* Tn)
            output =  B* 2 * Tk * M * N                             # 2*Tm * Tk * Tn * M * N / (Tm* Tn)
        elif(tiling == 'C'):
            B, M, N, K = self.dim[:self.get_effective_dim_len()]
            input_a =  B* Tn * M * K                                 #Tm* Tk *Tn * M * K / (Tk* Tm)
            input_b =  Tm * N * K                             # Tm* Tk * Tn * N * K / (Tk* Tn)
            output =  B* 2 * M * N                             # 2*Tm * Tn * M * N / (Tm* Tn)
        else:
            input_a = input_a
            input_b = input_b
            output = output
        # input_a = B * M * K
        # input_b = K * N
        # output = B * M * N
        # size_A =  M * K / (Tk* Tm)
        # size_B = N * K / (Tk* Tn)
        # size_C =  M * N / (Tm* Tn)
        if(data_format == 'int8'):
            input_a = input_a/2
            input_b = input_b/2
            output = output/2
        elif(data_format == 'fp32'):
            input_a = input_a*2
            input_b = input_b*2
            output = output*2
        elif(data_format == 'fp64'):
            input_a = input_a*4
            input_b = input_b*4
            output = output*4
        else:
            input_a = input_a
            input_b = input_b
            output = output
        input_a_read_time = 0 if (not read) else (input_a * 2 / (system.offchip_mem_bw)/system.memory_efficiency)
        input_b_read_time = input_b * 2 / (system.offchip_mem_bw)/system.memory_efficiency
        output_write_time = 0 if (not write) else (output * 2 / (system.offchip_mem_bw)/system.memory_efficiency)

        memory_total_time = input_a_read_time + input_b_read_time + output_write_time 

        return  memory_total_time 


    def get_roofline(self, system, read = True, write = True,  Tm = None, Tk = None, Tn = None,tiling = None, data_format = None, compute = None):
        unit = Unit() 
        ideal_compute_time = self.get_ideal_compute_time(system, data_format,compute)
        ideal_memory_time = self.get_ideal_memory_time(system, read, write, Tm , Tk, Tn,tiling,data_format) 
        num_ops = self.get_num_ops()
        input_a_size, input_w_size, output_size = self.get_tensors()

        num_data = (input_a_size + input_w_size + output_size)

        op_intensity = float(num_ops/num_data)

    ################################################################
    ## TODO A.2.iii
    ################################################################
    # Assume the computation and memory operation is perfectly synchronized  so they can be executed in parallel.
        exec_time = max(ideal_compute_time,ideal_memory_time)
        thrpt = num_ops/exec_time if exec_time else 0
        com_to_mem_ratio = ideal_compute_time/ideal_memory_time if ideal_memory_time else 0
        boundedness = 'C' if com_to_mem_ratio > 1 else 'M'


        ret = {
            'Op Type': self.get_op_type(),
            'Dimension': self.dim[:self.get_effective_dim_len()],
            'Bound': boundedness,
            'C/M ratio': com_to_mem_ratio,
            'Op Intensity': op_intensity,
            f'Latency ({unit.unit_time})': unit.raw_to_unit(exec_time, type='T'),
            f'Cycles': exec_time*system.frequency,
            f'Num ops ({unit.unit_flop})': unit.raw_to_unit(num_ops, type='O'),
            f'Input_a ({unit.unit_mem})': unit.raw_to_unit(input_a_size, type='M'),
            f'Input_w ({unit.unit_mem})': unit.raw_to_unit(input_w_size, type='M'),
            f'Output ({unit.unit_mem})': unit.raw_to_unit(output_size, type='M'),
            f'Total Data ({unit.unit_mem})': unit.raw_to_unit(num_data, type='M'),
            f'Throughput ({unit.unit_compute})': unit.raw_to_unit(thrpt, type='C'),
            f'Compute Cycles': ideal_compute_time*system.frequency,
            f'Memory Cycles': ideal_memory_time*system.frequency,
        }

        return ret










