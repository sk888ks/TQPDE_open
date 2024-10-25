import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"]=300
# Set font size 15 pt
plt.rcParams.update({'font.size': 15})
from TQPDEinput import get_dir_QPDE_path
dir_QPDE_path = get_dir_QPDE_path()

# 21 point grid from -9 to 9
x_values = [-9,-8.1,-7.2,-6.3,-5.4,-4.5,-3.6,-2.7,-1.8,-0.9,0,0.9,1.8,2.7,3.6,4.5,5.4,6.3,7.2,8.1,9]

# y values
y_values_a = [0.0117,0.0121,0.0131,0.0148,0.0155,0.0176,0.0188,0.0187,0.0216,0.0177,0.0193,0.022,0.0207,0.022,0.0211,0.0221,0.0178,0.016,0.0129,0.0159,0.0138]
y_values_b = [0.0022,0.0011,0.002,0.0022,0.0021,0.0019,0.0017,0.0014,0.0023,0.0023,0.0023,0.0026,0.0017,0.0017,0.0031,0.0019,0.0012,0.0021,0.0022,0.0024,0.0021]

png_save_name = "TQPDE_compare_MPO_Trotter"

# DataFrame
df_a = pd.DataFrame({'Energy': x_values, 'Probability': y_values_a})
df_b = pd.DataFrame({'Energy': x_values, 'Probability': y_values_b})

# plot
plt.figure(figsize=(10, 6))
plt.scatter(df_a['Energy'], df_a['Probability'], color='r', label='MPO', marker='o')
plt.scatter(df_b['Energy'], df_b['Probability'], color='b', label='Trotter', marker='o')
plt.xlabel('Energy (ε)')
plt.ylabel('Probability')
plt.legend()
# plt.show()
plt.savefig(dir_QPDE_path+"/picture/"+png_save_name+".png")
