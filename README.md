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
get_prob.py is a tool to extract destributions from the output text of a QPDE; for example, after generating output like "python QPDE_Bayes_Hubbard.py > out.txt" in examples/Hubbard directry, execute "python ../../src/get_prob_RD.py out.txt"  
plot_fill.py is a tool to create figures for QPDE from the destributions and to save the figures in "picture" directry, where the QPDE results in the paper are included in the code.


The "examples" directly include the code examples such as the Hubbard models. 
The codes starts from "MPO", "MPS", and "QPDE" are the MPO, MPS, and QPDE, respectively. The "create_hamiltonian" are codes to create Hamiltonian. plot_comare_Trotter.py in Hubbard includes results for comparison of the time evolution circuits.  
The [Fire Opal](https://docs.q-ctrl.com/fire-opal) module is used in the QPDE code, where the reference is as follows. 

    P. S. Mundada et al., Experimental Benchmarking of an Automated Deterministic Error-Suppression Workflow for Quantum Algorithms, Phys. Rev. Appl. 20, 024034 (2023).


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
fire-opal                 8.1.2  
Julia                     1.10.2  
ITensors                  v0.6.16  
Infiltrator               v1.7.0  
PyCall                    v1.96.4  