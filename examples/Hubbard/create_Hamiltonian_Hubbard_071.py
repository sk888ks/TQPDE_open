from qiskit_nature.second_q.operators.fermionic_op import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper

import numpy as np
import pickle

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src")))

from TQPDEinput import get_dir_QPDE_path
QPDE_path = get_dir_QPDE_path()

pickle_path = f"{QPDE_path}/pickle"

# Split H into interactions within and between subsystems
U = 10.0
t = 1.0

# input ========================================
num_sites = 4
Exact = False
save = False
# end input ====================================

name = f"ham_Hubbard_site{num_sites}_U{U}_t{t}_cp_071"

Ham_op = None
for site in range(num_sites):
    if Ham_op is None:  # Initialize the Hamiltonian operator
        Ham_op = U * (FermionicOp({f"+_{int(site*2)} -_{int(site*2)} +_{int(site*2+1)} -_{int(site*2+1)}": 1.0}, num_sites * 2))
    else:
        Ham_op += U * (FermionicOp({f"+_{int(site*2)} -_{int(site*2)} +_{int(site*2+1)} -_{int(site*2+1)}": 1.0}, num_sites * 2))    
    if site != num_sites - 1:
        Ham_op += t*(FermionicOp({f"-_{int(site*2)} +_{int(site*2+2)}": 1.0}, num_sites*2)+FermionicOp({(f"-_{int(site*2+2)} +_{int(site*2)}"):1.0}, num_sites*2) + FermionicOp({f"-_{int(site*2+1)} +_{int(site*2+3)}":1.0}, num_sites*2)+FermionicOp({f"-_{int(site*2+3)} +_{int(site*2+1)}":1.0}, num_sites*2))

for site in range(num_sites):
    Ham_op -= (U/2)*(FermionicOp({f"+_{int(site*2)} -_{int(site*2)}":1.0}, num_sites*2)+FermionicOp({f"+_{int(site*2+1)} -_{int(site*2+1)}":1.0},num_sites*2))

# JW
mapper = JordanWignerMapper()
Ham = mapper.map(Ham_op)

if save:
    with open(f"{pickle_path}/{name}.pickle", "wb") as f:
        pickle.dump(Ham, f)

    with open(f"{pickle_path}/{name}_pso.pickle", "wb") as f:
        pickle.dump(Ham, f)

if Exact:
    # Obtain exact values
    eigenvalue, eigenvec = np.linalg.eigh(np.array(Ham.to_matrix()))

    print("argmax")
    print(eigenvec.T[0][np.argmax(np.abs(eigenvec.T[0]))])
    print(eigenvec.T[1][np.argmax(np.abs(eigenvec.T[1]))])

    print(bin(np.argmax(np.abs(eigenvec.T[0])))[2:].zfill(num_sites * 2))
    print(bin(np.argmax(np.abs(eigenvec.T[1])))[2:].zfill(num_sites * 2))

    print(f"Energy ground {eigenvalue[0]}")

    print(f"Energy gap {eigenvalue[1] - eigenvalue[0]}")
