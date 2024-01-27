
import operators as operators
import pandas as pd
import numpy as np



def analysis_model(model_operators, system):
    roofline_list = []
    for i,operator_instance in enumerate(model_operators):
        roofline = operator_instance.get_roofline(system=system)
        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)
    

    return df

