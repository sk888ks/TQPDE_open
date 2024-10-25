
using Serialization

let
    # input =================================
    src_dir = "../../src/" #Directry of Gate_generation_from_MPS_MPO.jl

    num_sites = 4 #sytem qubits = 2*num_sites

    #DMRG settings
    scratch = false #true: create reference MPS by DMRG, false: load the MPS file
    nsweeps = 20 #sweep
    maxdim =  append!([10 for i in 1:3], [50 for i in 1:12], [1000 for i in 1:5]) # max bond dimention
    cutoff = 1e-12 # threshold for the singular values

    num_depth_MPSBW = 6 #d_prep
    num_sweep_MPSBW = 1000 # optimization cycles for Uprep
    # end input =================================

    mindim = 2
    U = 10.0
    t = 1.0

    include(joinpath(joinpath(pwd(), src_dir), "Gate_generation_from_MPS_MPO.jl"))


    name_MPSBW = "Hubbard_site$(num_sites)_U$(U)_t$(t)_depth$(num_depth_MPSBW)_sweep$(num_sweep_MPSBW)_cp_vac" 
    name_ham = "ham_Hubbard_site$(num_sites)_U$(U)_t$(t)_cp"
    name_target_MPS = "Hubbard_site$(num_sites)_U$(U)_t$(t)_cp_vac"

    #scratch
    if scratch
        psi_sp = get_psi_sp_vac(name_ham, nsweeps, maxdim, mindim, cutoff)
        serialize("$(pickle_dir)/$(name_target_MPS).dat", psi_sp)
    else
        psi_sp = deserialize("$(pickle_dir)/$(name_target_MPS).dat")
    end
 
    psi_list_random, amplitude_list_random = get_and_save_prep_MPSBW_2d_list(psi_sp, num_depth_MPSBW, num_sweep_MPSBW, name_MPSBW)

    nothing

end