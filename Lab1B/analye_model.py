
import operators as operators
import pandas as pd
import numpy as np



def analysis_model(model_operators, system, fusion = [], tiling = None, Tm = None, Tk = None, Tn = None, data_format= None,  compute = None):
    roofline_list = []
    for i,operator_instance in enumerate(model_operators):
        #fusion
        write = True
        read = True
        for k in fusion:
            if(i == k[0]):
                write = False
            if(i == k[1]):
                read = False
        roofline = operator_instance.get_roofline(system, read, write, Tm , Tk, Tn, tiling, data_format, compute)
        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)
    

    return df

