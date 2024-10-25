This project includes code examples and results for the following paper. Please cite the paper if you use them.

    S. Kanno, K. Sugisaki, H. Nakamura, H. Yamauchi, R. Sakuma, T. Kobayashi, Q. Gao, and N. Yamamoto, Tensor-Based Quantum Phase Difference Estimation for Large-Scale Demonstration, http://arxiv.org/abs/2408.04946.

Note that the result data will overwrite by executing codes.

The "src" directory includes execution source codes. Before the execution, please replace "xxxx" in input.jl and TQPDEinput.py to an appropriate name.  
Also, please set Python ENV in Julia as follows.  
"""  
ENV["PYTHON"] = "/path/to/your/python"  
using Pkg  
Pkg.build("PyCall")  
"""  
get_prob.py in src is a tool to extract mean and variance from the output text of a QPDE; after generating output like "python QPDE_Bayes_Hubbard.py > out.txt", execute "python get_prob_RD.py out.txt"


The "examples" directly include the code examples such as the Hubbard models. 
The codes starts from "MPO", "MPS", and "QPDE" are the MPO, MPS, and QPDE, respectively. The "create_hamiltonian" are codes to create Hamiltonian. plot_comare_Trotter in Hubbard includes results for comparison of the time evolution circuits.


The "pickle_dat", "npy", "molden", and "picture" directories include input and/or output files. Hamiltonian pickle and reference MPO and MPS dat files are in "pickle_dat", brick wall gates (Uprep and Uevol) files are in "npy", molden files for molecular structure depiction is in "molden", and pictures used in our study in "picture".


Code version  
Python                    3.11.8  
qiskit                    1.2.4  
qiskit-aer                0.14.2  
qiskit-algorithms         0.3.0  
qiskit-ibm-runtime        0.29.1  
qiskit-nature             0.7.2  
matplotlib                3.8.3  
scipy                     1.12.0  
tenacity                  8.3.0  
(fire-opal 8.1.2 is used in the study, but it is not included in this repository)  
Julia                     1.10.2  
ITensors                  v0.6.16  
Infiltrator               v1.7.0  
PyCall                    v1.96.4  