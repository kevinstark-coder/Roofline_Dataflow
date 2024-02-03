import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from unit import Unit


def plot_roofline_background(system, max_x,unit):
    op_intensity = system.flops/system.offchip_mem_bw
    flops = unit.raw_to_unit(system.op_per_sec, type='C')
    max_x = max_x
    turning_points = [[0, 0], [op_intensity, flops], [max(max_x, 1.5*op_intensity), flops]]
    # print(max_x," ",op_intensity," ",flops,"yqk")
    turning_points = np.array(turning_points)
    plt.plot(turning_points[:,0], turning_points[:,1], c='grey')
    # print(turning_points[:,0]," ",turning_points[:,1],"yqk2")
    plt.xlabel('Op Intensity (FLOPs/Byte)')
    plt.ylabel(f'{unit.unit_compute.upper()}')

plot_roofline_background