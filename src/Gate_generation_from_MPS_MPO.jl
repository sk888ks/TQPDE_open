# using Revise
using ITensors
using PyCall
using Random
# using ITensors.HDF5
using Infiltrator
using LinearAlgebra
using Dates
include("input.jl")
ITensors.set_warn_order(1000) # Suppress warnings

# Convert unitary matrix to NumPy array
np = pyimport("numpy")
pickle = pyimport("pickle")


function term_to_list(Hamiltonian::PyObject)
    # Convert Qiskit's Hamiltonian coefficients and Paulis to list format
    ham_coeff_list, ham_pauli_list = nothing, nothing
    try
        ham_coeff_list = Hamiltonian["coeffs"]
        ham_pauli_list = Hamiltonian.primitive.paulis.settings["data"]
    catch
        try
            ham_coeff_list = [Hamiltonian["coeff"]]
            ham_pauli_list = [Hamiltonian.primitive.settings["data"]]
            println("only one parameter is included in the Hamiltonian")
        catch # sparse Pauli object 
            ham_coeff_list = [i for i in Hamiltonian["_coeffs"]]
            ham_pauli_list = [i for i in Hamiltonian._pauli_list.settings["data"]]
        end
    end

    return ham_coeff_list, ham_pauli_list
end

function get_Hamiltonian(coeffs, paulis,sites)
    # Construct Hamiltonian as MPO from coefficient and Pauli list
    os = OpSum()
    for (coeff, pauli) in zip(coeffs, reverse.(paulis)) # reverse.(paulis)
        term = OpSum()
        term += coeff,"I",1
        for (i, op) in enumerate(pauli)
            if string(op) != "I"
                term *= string(op), i
            end
        end
        os += term
    end
    H = MPO(os, sites)
    return H
end

function get_operators(coeffs, paulis)
    # Construct Hamiltonian as MPO from coefficient and Pauli list
    os = OpSum()
    for (coeff, pauli) in zip(coeffs, reverse.(paulis)) # reverse.(paulis)
        term = OpSum()
        term += coeff,"I",1
        for (i, op) in enumerate(pauli)
            if string(op) != "I"
                term *= string(op), i
            end
        end
        os += term
    end
    return os
end

function get_identity_MPO(N)
    left_ind_list = []
    right_ind_list = []

    # Initialize MPO
    evol_MPO = MPO(N)
    for idx in 1:N
        # Define tensor with three indices
        i = Index(2, "i") # in
        push!(left_ind_list, i)
        j = Index(2, "j") # out
        push!(right_ind_list, j)
        
        # Three-legged tensor
        T = ITensor(i, j)
        for i in 1:2
            for j in 1:2
                if i == j
                    T[i,j] = 1.0
                else
                    T[i,j] = 0.0
                end
            end
        end 
        # @infiltrate
        evol_MPO[idx] = T
        # @infiltrate
    end
    return evol_MPO, left_ind_list, right_ind_list
end

function make_pauli_term(op_str, s, coeff)
    # Construct an operator based on the specified Pauli string and coefficient
    term = ITensor(1.0)
    flag_all_I = true
    for (idx, op_single) in enumerate(op_str)
        # @infiltrate
        if string(op_single) != "I"
            term *= op(string(op_single), s[idx])
            flag_all_I = false
        end
    end
    if flag_all_I == true
        term *= op("I", s[1])
    end
    return coeff * term
end

function get_evol_MPO(pauli, coeff, sliced_time, right_ind_list; cutoff = 1e-12)
    # Get time-evolution MPO

    N = length(pauli)
    
    # Pauli matrices
    I = [1.0 0.0; 0.0 1.0]
    Z = [1.0 0.0; 0.0 -1.0]
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -im; im 0.0]
    
    # Determine where to apply exp to the Pauli matrices
    ope_pauli_idx_list = []
    # @infiltrate
    for (idx, pauli_single) in enumerate(pauli)
        if string(pauli_single) == "I"
            nothing
        else
            push!(ope_pauli_idx_list, idx)
        end
    end

    if length(ope_pauli_idx_list) > 10
        println("Warning: large locality $(length(ope_pauli_idx_list))")
        # @infiltrate
    end

    # Generate indices
    left_ind_list_MPO = right_ind_list

    # Generate MPO
    evol_MPO = MPO(N)
    # Fill the non-exp parts with I
    for idx in 1:N
        # @infiltrate
        if !(idx in ope_pauli_idx_list)
            evol_MPO[idx] = itensor(I, (left_ind_list_MPO[idx], prime(left_ind_list_MPO[idx])))
        end
    end

    # @infiltrate
    if length(ope_pauli_idx_list) == 0
        # If only I is present, apply exp to site 1 and return
        # @infiltrate
        evol_MPO[1] = exp(-im * sliced_time * coeff * evol_MPO[1])
        return evol_MPO
    end

    # @infiltrate
    # Create Pauli for exp
    op_for_exp = nothing
    counter_temp = 1
    tensor_temp = nothing
    for (iter, ope_pauli_idx) in enumerate(ope_pauli_idx_list)
        # @infiltrate
        # Generate tensor
        if string(pauli[ope_pauli_idx]) == "X"
            tensor_temp = itensor(X, (left_ind_list_MPO[ope_pauli_idx], prime(left_ind_list_MPO[ope_pauli_idx])))
        elseif string(pauli[ope_pauli_idx]) == "Y"
            tensor_temp = itensor(Y, (left_ind_list_MPO[ope_pauli_idx], prime(left_ind_list_MPO[ope_pauli_idx])))
        elseif string(pauli[ope_pauli_idx]) == "Z"
            tensor_temp = itensor(Z, (left_ind_list_MPO[ope_pauli_idx], prime(left_ind_list_MPO[ope_pauli_idx])))
        elseif string(pauli[ope_pauli_idx]) == "I"
            error("I was detected in ope_pauli_idx")
        end

        # Merge tensors
        if iter == 1
            op_for_exp = tensor_temp
        else
            op_for_exp *= tensor_temp
        end
        counter_temp += 1
    end
    # @infiltrate

    
    # Time evolution: apply 2nd Trotter or † effects to coeff
    # @infiltrate
    evol_MPO_temp = exp(-im * sliced_time * coeff * op_for_exp)
    evol_MPO[ope_pauli_idx_list[1]] = evol_MPO_temp
    # counter_temp = 1
    # @infiltrate

    for (iter, ope_pauli_idx) in enumerate(ope_pauli_idx_list)
        if iter != length(ope_pauli_idx_list) # If iteration reaches the end of the list, return MPO; otherwise, decompose using SVD
            if iter == 1
                U, S, V = svd(evol_MPO[ope_pauli_idx], (left_ind_list_MPO[ope_pauli_idx], prime(left_ind_list_MPO[ope_pauli_idx])); cutoff = cutoff)
            else iter != 1
                # @infiltrate
                shared_index_before = commonind(evol_MPO[ope_pauli_idx_list[iter]], evol_MPO[ope_pauli_idx_list[iter-1]]) 
                if isnothing(shared_index_before)
                    U, S, V = svd(evol_MPO[ope_pauli_idx], (left_ind_list_MPO[ope_pauli_idx], prime(left_ind_list_MPO[ope_pauli_idx])); cutoff = cutoff)
                else
                    U, S, V = svd(evol_MPO[ope_pauli_idx], (left_ind_list_MPO[ope_pauli_idx], shared_index_before, prime(left_ind_list_MPO[ope_pauli_idx])); cutoff = cutoff)
                end
            end
            evol_MPO[ope_pauli_idx_list[iter]] = U
            evol_MPO[ope_pauli_idx_list[iter+1]] = S*V
            # @infiltrate
        end
    end
    # @infiltrate

    evol_MPO = sweep_MPO(evol_MPO; ind_list = left_ind_list_MPO)
    # @infiltrate

    return evol_MPO
end

function get_MPO_trace(MPO)
    # Get the trace of the MPO. Created for testing purposes
    N = length(MPO)
    MPO_tr = deepcopy(MPO)
    for tensor_idx in 1:N
        if tensor_idx == 1
            shared_index_after = commonind(MPO[tensor_idx], MPO[tensor_idx+1])
            unique_indices = filter(ind -> ind != shared_index_after, inds(MPO[tensor_idx]))
        elseif tensor_idx != N
            shared_index_after = commonind(MPO[tensor_idx], MPO[tensor_idx+1])
            shared_index_before = commonind(MPO[tensor_idx], MPO[tensor_idx-1])
            unique_indices = filter(ind -> ind != shared_index_after && ind != shared_index_before, inds(MPO[tensor_idx]))
        else
            shared_index_before = commonind(MPO[tensor_idx], MPO[tensor_idx-1])
            unique_indices = filter(ind -> ind != shared_index_before, inds(MPO[tensor_idx]))
        end
        MPO_tr[tensor_idx] *= delta(unique_indices[1], unique_indices[2])
    end
    tr_val = contract(MPO_tr)
    return complex(tr_val)[1]
end

function replace_MPS_inds(MPS, replace_ind)
    # Replace the indices of the MPS. Created for testing purposes
    N = length(MPS)
    MPS_rp = deepcopy(MPS)
    for tensor_idx in 1:N
        if tensor_idx == 1
            shared_index_after = commonind(MPS[tensor_idx], MPS[tensor_idx+1])
            unique_indices = filter(ind -> ind != shared_index_after, inds(MPS[tensor_idx]))
        elseif tensor_idx != N
            shared_index_after = commonind(MPS[tensor_idx], MPS[tensor_idx+1])
            shared_index_before = commonind(MPS[tensor_idx], MPS[tensor_idx-1])
            unique_indices = filter(ind -> ind != shared_index_after && ind != shared_index_before, inds(MPS[tensor_idx]))
        else
            shared_index_before = commonind(MPS[tensor_idx], MPS[tensor_idx-1])
            unique_indices = filter(ind -> ind != shared_index_before, inds(MPS[tensor_idx]))
        end
        # @infiltrate
        MPS_rp[tensor_idx] = replaceind(MPS_rp[tensor_idx], unique_indices[1], replace_ind[tensor_idx])
    end
    return MPS_rp
end

function sweep_MPO(MPO; cutoff = 1e-12, ind_list = nothing)
    # Sweep through the MPO
    # @infiltrate
    N = length(MPO)
    MPO_for_svd = nothing 
    for idx_tensor in 1:N
        if idx_tensor == 1
            if isnothing(ind_list)
                shared_index_after = commonind(MPO[idx_tensor], MPO[idx_tensor+1]) 
                unique_indices = filter(ind -> ind != shared_index_after, inds(MPO[idx_tensor]))
            else
                unique_indices = (ind_list[idx_tensor], prime(ind_list[idx_tensor]))
            end
            MPO_for_svd = MPO[idx_tensor] * MPO[idx_tensor+1]   
            U, S, V = svd(MPO_for_svd, unique_indices; cutoff = cutoff)    
            MPO[idx_tensor] = U
            MPO[idx_tensor+1] = S * V
            # @infiltrate

        elseif idx_tensor != N
            shared_index_before = commonind(MPO[idx_tensor], MPO[idx_tensor-1])
            if isnothing(ind_list)
                shared_index_after = commonind(MPO[idx_tensor], MPO[idx_tensor+1]) 
                unique_indices = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(MPO[idx_tensor]))
            else
                unique_indices = (ind_list[idx_tensor], prime(ind_list[idx_tensor]))
            end
            MPO_for_svd = MPO[idx_tensor] * MPO[idx_tensor+1]
            U, S, V = svd(MPO_for_svd, (shared_index_before, unique_indices...); cutoff = cutoff)    
            MPO[idx_tensor] = U
            MPO[idx_tensor+1] = S * V
            # @infiltrate
        elseif idx_tensor == N
            # @infiltrate
            nothing
        end
    end

    for idx_tensor in N:-1:1
        # @infiltrate
        if idx_tensor == N
            if isnothing(ind_list)
                shared_index_before = commonind(MPO[idx_tensor], MPO[idx_tensor-1]) 
                unique_indices = filter(ind -> ind != shared_index_before, inds(MPO[idx_tensor]))
            else
                unique_indices = (ind_list[idx_tensor], prime(ind_list[idx_tensor]))
            end
            MPO_for_svd = MPO[idx_tensor] * MPO[idx_tensor-1]   
            U, S, V = svd(MPO_for_svd, unique_indices; cutoff = cutoff)
            MPO[idx_tensor] = U
            MPO[idx_tensor-1] = S * V
            # @infiltrate
        elseif idx_tensor != 1
            shared_index_after = commonind(MPO[idx_tensor], MPO[idx_tensor+1]) 
            if isnothing(ind_list)
                shared_index_before = commonind(MPO[idx_tensor], MPO[idx_tensor-1]) 
                unique_indices = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(MPO[idx_tensor]))
            else
                unique_indices = (ind_list[idx_tensor], prime(ind_list[idx_tensor]))
            end
            MPO_for_svd = MPO[idx_tensor] * MPO[idx_tensor-1]
            U, S, V = svd(MPO_for_svd, (shared_index_after, unique_indices...); cutoff = cutoff)    
            MPO[idx_tensor] = U
            MPO[idx_tensor-1] = S * V
            # @infiltrate
        elseif idx_tensor == 1
            nothing
        end
    end
    return MPO

end

function get_gate_second_trotter(ham, N, time_step, num_time_slice)
    # Create gates for 2nd order Trotter
    coeffs, paulis = term_to_list(ham)

    # The site order in Qiskit and ITensor is reversed
    paulis = reverse.(paulis)

    # @infiltrate

    # Initialize MPO
    evol_MPO, left_ind_list, right_ind_list = get_identity_MPO(N)
    # @infiltrate
    sliced_time = time_step/num_time_slice

    # @infiltrate
    for (pauli, coeff) in zip(paulis, coeffs) # Time evolution corresponding to exp(iΔτT/2)
        # Get time evolution MPO
        coeff = coeff * (-1) * 0.5 # The minus from † and 1/2 from 2nd Trotter
        # @infiltrate
        evol_MPO_next = get_evol_MPO(pauli, coeff, sliced_time, right_ind_list)
        # evol_MPO_next = sweep_MPO(evol_MPO_next)
        # @infiltrate
        # Perform time evolution
        evol_MPO *= evol_MPO_next
        # @infiltrate
        # Align indices
        noprime!(evol_MPO)
        # Sweep
        evol_MPO = sweep_MPO(evol_MPO)
        # @infiltrate
    end

    for (pauli, coeff) in zip(reverse([i for i in paulis]), reverse([i for i in coeffs])) ## Time evolution corresponding to exp(iΔτT/2) in reverse
        # Get time evolution MPO
        coeff = coeff * (-1) * 0.5 # The minus from † and 1/2 from 2nd Trotter
        # @infiltrate
        evol_MPO_next = get_evol_MPO(pauli, coeff, sliced_time, right_ind_list)
        # evol_MPO_next = sweep_MPO(evol_MPO_next)
        # @infiltrate
        # Perform time evolution
        evol_MPO *= evol_MPO_next
        # @infiltrate
        # Align indices
        noprime!(evol_MPO)
        # Sweep
        evol_MPO = sweep_MPO(evol_MPO)
        # @infiltrate
    end
    evol_MPO = sweep_MPO(evol_MPO)

    # Multiply the obtained evol_MPO as many times as needed
    # Change the legs to make it easier to multiply initially
    evol_MPO_next = deepcopy(evol_MPO)
    for site_idx in 1:N
        evol_MPO_next[site_idx] = replaceind(evol_MPO_next[site_idx], right_ind_list[site_idx], prime(right_ind_list[site_idx]))
        # @infiltrate
        evol_MPO_next[site_idx] = replaceind(evol_MPO_next[site_idx], left_ind_list[site_idx], right_ind_list[site_idx])
        # @infiltrate
    end
    # Multiply step-1 times
    for idx in 1:num_time_slice-1
        evol_MPO *= evol_MPO_next
        # Align indices
        # @infiltrate
        noprime!(evol_MPO)
        # Sweep
        # @infiltrate
        evol_MPO = sweep_MPO(evol_MPO)
        # @infiltrate
    end

    # @infiltrate

    return evol_MPO
end

function get_psi_sp(name_ham, nsweeps, maxdim, mindim, cutoff)
    ham = hamcall(name_ham)

    # Convert coefficients for ITensor
    coeffs, paulis = term_to_list(ham)
    # println("#coeffs: ", length(coeffs))
    # println(paulis)

    # Create N spin-one indices
    N = length(paulis[1])
    println("#sites: ",N)
    sites = siteinds("S=1/2",N)

    # Hamiltonian = MPO
    H = get_Hamiltonian(coeffs, paulis, sites)

    # Create an initial random matrix product state
    Random.seed!(1)
    psi0 = randomMPS(sites)

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized MPS
    println("After sweep 0 energy=$(real(inner(psi0' , H , psi0)))")
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, mindim=mindim, cutoff=cutoff)
    println("Final energy = $energy")

    # Move singular values to the right
    orthogonalize!(psi, 1)
    # Calculate magnetization
    magz_g = expect(psi,"Sz")
    println("magz ground $(sum(magz_g))")

    # DMRG settings
    nsweeps_ex = nsweeps # Number of sweeps
    maxdim_ex = maxdim # Maximum dimension for each sweep
    mindim_ex = mindim
    cutoff_ex = cutoff # Threshold for discarding singular values

    psi_list = [psi]
    num_ex = 1
    psi_ex = get_excited_state(num_ex, psi_list, psi0, H, nsweeps_ex=nsweeps_ex, maxdim_ex=maxdim_ex, mindim_ex=mindim_ex, cutoff_ex=cutoff_ex)
    # psi_ex = psi_list[end]
    energy_ex = real(inner(psi_ex' , H , psi_ex))
    println("Energy gap $(energy_ex-energy)")
    # @infiltrate
    orthogonalize!(psi_ex, 1)

    # Calculate magnetization
    magz_ex = expect(psi_ex,"Sz")
    println("magz excited $(sum(magz_ex))")
    flush(stdout)

    # Create |0>|psi_g>
    psi_anc0 = MPS(N+1)
    anc0 = Index(2, "anc0")
    Asite0_anc0 = ITensor(anc0)
    Asite0_anc0[1] = 1.0
    Asite0_anc0[2] = 0.0
    psi_anc0[1] = Asite0_anc0

    for site_idx in 1:N
        psi_anc0[site_idx+1] = psi[site_idx]
    end
    normalize!(psi_anc0)
    orthogonalize!(psi_anc0, 1)
    # @infiltrate

    # Create |1>|psi_ex>
    psi_anc1 = MPS(N+1)
    anc1 = Index(2, "anc1")
    Asite0_anc1 = ITensor(anc1)
    Asite0_anc1[1] = 0.0
    Asite0_anc1[2] = 1.0
    psi_anc1[1] = Asite0_anc1

    for site_idx in 1:N
        psi_anc1[site_idx+1] = psi_ex[site_idx]
    end

    # @infiltrate
    normalize!(psi_anc1)
    orthogonalize!(psi_anc1, 1)
    # @infiltrate

    # Make psi_anc1 share the index of psi_anc0
    for idx in 1:N+1
        if idx == 1
            shared_ind_after = commonind(psi_anc0[idx],psi_anc0[idx+1])
            unique_ind = filter(ind -> ind != shared_ind_after, inds(psi_anc0[idx]))[1]
            shared_ind_after_ex = commonind(psi_anc1[idx],psi_anc1[idx+1])
            unique_ind_ex = filter(ind -> ind != shared_ind_after_ex, inds(psi_anc1[idx]))[1]
        elseif idx != N+1
            # @infiltrate
            shared_ind_before = commonind(psi_anc0[idx],psi_anc0[idx-1])
            shared_ind_after = commonind(psi_anc0[idx],psi_anc0[idx+1])
            unique_ind = filter(ind -> ind != shared_ind_before && ind != shared_ind_after, inds(psi_anc0[idx]))[1]
            shared_ind_before_ex = commonind(psi_anc1[idx],psi_anc1[idx-1])
            shared_ind_after_ex = commonind(psi_anc1[idx],psi_anc1[idx+1])
            unique_ind_ex = filter(ind -> ind != shared_ind_before_ex && ind != shared_ind_after_ex, inds(psi_anc1[idx]))[1]
        else
            shared_ind_before = commonind(psi_anc0[idx],psi_anc0[idx-1])
            unique_ind = filter(ind -> ind != shared_ind_before, inds(psi_anc0[idx]))[1]
            shared_ind_before_ex = commonind(psi_anc1[idx],psi_anc1[idx-1])
            unique_ind_ex = filter(ind -> ind != shared_ind_before_ex, inds(psi_anc1[idx]))[1]
        end
        psi_anc1[idx] = replaceind(psi_anc1[idx], unique_ind_ex, unique_ind)
    end

    # @infiltrate
    psi_sp = psi_anc0 + psi_anc1
    normalize!(psi_sp)
    orthogonalize!(psi_sp, 1)
    # @infiltrate

    # Optional sweep of psi =====================
    # println(psi_sp)
    psi_sp = sweep_MPS(psi_sp, N+1;cutoff = cutoff)
    normalize!(psi_sp)
    orthogonalize!(psi_sp, 1)

    return psi_sp
end

function get_psi_sp_vac(name_ham, nsweeps, maxdim, mindim, cutoff)
    ham = hamcall(name_ham)

    # Convert coefficients for ITensor
    coeffs, paulis = term_to_list(ham)
    # println("#coeffs: ", length(coeffs))
    # println(paulis)

    # Create N spin-one indices
    N = length(paulis[1])
    println("#sites: ",N)
    sites = siteinds("S=1/2",N)

    # Hamiltonian = MPO
    H = get_Hamiltonian(coeffs, paulis, sites)

    # Create an initial random matrix product state
    Random.seed!(1)
    psi0 = randomMPS(sites)

    # Run the DMRG algorithm, returning energy
    # (dominant eigenvalue) and optimized MPS
    println("After sweep 0 energy=$(real(inner(psi0' , H , psi0)))")
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, mindim=mindim, cutoff=cutoff)
    println("Final energy = $energy")

    # Move singular values to the right
    orthogonalize!(psi, 1)
    # Calculate magnetization
    magz_g = expect(psi,"Sz")
    println("magz ground $(sum(magz_g))")

    # Create |0>|psi_g>
    psi_anc0 = MPS(N+1)
    anc0 = Index(2, "anc0")
    Asite0_anc0 = ITensor(anc0)
    Asite0_anc0[1] = 1.0
    Asite0_anc0[2] = 0.0
    psi_anc0[1] = Asite0_anc0

    for site_idx in 1:N
        psi_anc0[site_idx+1] = psi[site_idx]
    end
    normalize!(psi_anc0)
    orthogonalize!(psi_anc0, 1)
    # @infiltrate

    # Create |1>|vac>
    psi_anc1 = MPS(N+1)
    anc1 = Index(2, "anc1")
    Asite0_anc1 = ITensor(anc1)
    Asite0_anc1[1] = 0.0
    Asite0_anc1[2] = 1.0
    psi_anc1[1] = Asite0_anc1

    for site_idx in 1:N
        i = Index(2, "i")
        vac_tensor = ITensor(i)
        vac_tensor[1] = 1.0
        vac_tensor[2] = 0.0
        psi_anc1[site_idx+1] = vac_tensor
    end

    # @infiltrate
    normalize!(psi_anc1)
    orthogonalize!(psi_anc1, 1)
    # @infiltrate

    # Make psi_anc1 share the index of psi_anc0
    for idx in 1:N+1
        if idx == 1
            shared_ind_after = commonind(psi_anc0[idx],psi_anc0[idx+1])
            unique_ind = filter(ind -> ind != shared_ind_after, inds(psi_anc0[idx]))[1]
            shared_ind_after_ex = commonind(psi_anc1[idx],psi_anc1[idx+1])
            unique_ind_ex = filter(ind -> ind != shared_ind_after_ex, inds(psi_anc1[idx]))[1]
        elseif idx != N+1
            # @infiltrate
            shared_ind_before = commonind(psi_anc0[idx],psi_anc0[idx-1])
            shared_ind_after = commonind(psi_anc0[idx],psi_anc0[idx+1])
            unique_ind = filter(ind -> ind != shared_ind_before && ind != shared_ind_after, inds(psi_anc0[idx]))[1]
            shared_ind_before_ex = commonind(psi_anc1[idx],psi_anc1[idx-1])
            shared_ind_after_ex = commonind(psi_anc1[idx],psi_anc1[idx+1])
            unique_ind_ex = filter(ind -> ind != shared_ind_before_ex && ind != shared_ind_after_ex, inds(psi_anc1[idx]))[1]
        else
            shared_ind_before = commonind(psi_anc0[idx],psi_anc0[idx-1])
            unique_ind = filter(ind -> ind != shared_ind_before, inds(psi_anc0[idx]))[1]
            shared_ind_before_ex = commonind(psi_anc1[idx],psi_anc1[idx-1])
            unique_ind_ex = filter(ind -> ind != shared_ind_before_ex, inds(psi_anc1[idx]))[1]
        end
        psi_anc1[idx] = replaceind(psi_anc1[idx], unique_ind_ex, unique_ind)
    end

    # @infiltrate
    psi_sp = psi_anc0 + psi_anc1
    normalize!(psi_sp)
    orthogonalize!(psi_sp, 1)
    # @infiltrate

    # Optional sweep of psi =====================
    # println(psi_sp)
    psi_sp = sweep_MPS(psi_sp, N+1;cutoff = cutoff)
    normalize!(psi_sp)
    orthogonalize!(psi_sp, 1)

    return psi_sp
end

# Extract left and right indices from the MPO. Use the fact that the left tag is "i"
function get_left_right_list(MPO)
    N = length(MPO)
    left_ind_list = []
    right_ind_list = []
    for idx in 1:N 
        if idx == 1
            shared_indices_after = commonind(MPO[idx], MPO[idx+1])
            unique_indices = filter(ind -> ind != shared_indices_after, inds(MPO[idx]))
            left_index = filter(x -> string(tags(x)) == "\"i\"", unique_indices)[1]
            push!(left_ind_list, left_index)
            right_index = filter(x -> string(tags(x)) != "\"i\"", unique_indices)[1]
            push!(right_ind_list, right_index)
            # @infiltrate
        elseif idx != N
            shared_indices_after = commonind(MPO[idx], MPO[idx+1])
            shared_indices_before = commonind(MPO[idx], MPO[idx-1])
            unique_indices = filter(ind -> ind != shared_indices_after && ind != shared_indices_before, inds(MPO[idx]))
            left_index = filter(x -> string(tags(x)) == "\"i\"", unique_indices)[1]
            push!(left_ind_list, left_index)
            right_index = filter(x -> string(tags(x)) != "\"i\"", unique_indices)[1]
            push!(right_ind_list, right_index)
            # @infiltrate
        else
            shared_indices_before = commonind(MPO[idx], MPO[idx-1])
            unique_indices = filter(ind -> ind != shared_indices_before, inds(MPO[idx]))
            left_index = filter(x -> string(tags(x)) == "\"i\"", unique_indices)[1]
            push!(left_ind_list, left_index)
            right_index = filter(x -> string(tags(x)) != "\"i\"", unique_indices)[1]
            push!(right_ind_list, right_index)
            # @infiltrate            
        end
    end
    return left_ind_list, right_ind_list
end

# Extract left and right indices from the MPS
function get_left_list_MPSBW(psi)
    N = length(psi)
    left_ind_list = []
    for idx in 1:N
        if idx == 1
            shared_indices_after = commonind(psi[idx], psi[idx+1])
            unique_index = filter(ind -> ind != shared_indices_after, inds(psi[idx]))[1]
            push!(left_ind_list, unique_index)
            # @infiltrate
        elseif idx != N
            shared_indices_after = commonind(psi[idx], psi[idx+1])
            shared_indices_before = commonind(psi[idx], psi[idx-1])
            unique_index = filter(ind -> ind != shared_indices_after && ind != shared_indices_before, inds(psi[idx]))
            push!(left_ind_list, unique_index)
            # @infiltrate
        else
            shared_indices_before = commonind(psi[idx], psi[idx-1])
            unique_index = filter(ind -> ind != shared_indices_before, inds(psi[idx]))
            push!(left_ind_list, unique_index)
            # @infiltrate            
        end
    end
    return left_ind_list
end


function get_identity_operator(T)
    # Initialize with identity operator
    for i in 1:2
        for j in 1:2
            for k in 1:2
                for l in 1:2
                    if i == k && j == l
                        T[i,j,k,l] = 1.0
                    else
                        T[i,j,k,l] = 0.0
                    end
                end
            end
        end
    end
    return T
end


function  get_bond_list(num_depth, N, left_ind_list, right_ind_list)
    # Create bond indices for MPO
    bond_list = [right_ind_list]

    for depth_idx in 1:num_depth-1
        bond_list_temp = []
        if depth_idx % 2 == 1 # Odd depth
            for site_idx in 1:N
                if depth_idx != num_depth-1 # Not the rightmost
                    # @infiltrate
                    b = Index(2, "b")
                    push!(bond_list_temp, b)
                else # Rightmost
                    if site_idx == 1
                        push!(bond_list_temp, left_ind_list[site_idx])
                    elseif site_idx != N
                        b = Index(2, "b")
                        push!(bond_list_temp, b)    
                    else
                        push!(bond_list_temp, left_ind_list[site_idx])
                    end
                end
            end
        else # Even depth
            for site_idx in 1:N
                # @infiltrate
                if site_idx == 1 # Use previous bond for top and bottom ends
                    push!(bond_list_temp,  bond_list[depth_idx][site_idx])
                elseif site_idx != N 
                    b = Index(2, "b")
                    push!(bond_list_temp, b)
                else # Use previous bond for top and bottom ends
                    push!(bond_list_temp,  bond_list[depth_idx][site_idx])
                end
            end 
        end
        push!(bond_list, bond_list_temp)
    end
    push!(bond_list, left_ind_list)
    return bond_list
end

function  get_bond_list_MPSBW(num_depth, N, left_ind_list, init_allzero_list)
    # Create bond indices for MPO
    bond_list = [init_allzero_list]

    for depth_idx in 1:num_depth-1
        bond_list_temp = []
        if depth_idx % 2 == 1 # Odd depth
            for site_idx in 1:N
                if depth_idx == 1 # Leftmost
                    if site_idx != N
                        b = Index(2, "b")
                        push!(bond_list_temp, b)
                    else # Use init for the top end
                        push!(bond_list_temp, init_allzero_list[site_idx])
                    end
                elseif depth_idx != num_depth-1 # Not the rightmost
                    # @infiltrate
                    if site_idx != N
                        b = Index(2, "b")
                        push!(bond_list_temp, b)
                    else # Use previous bond for the top end
                        push!(bond_list_temp, bond_list[depth_idx][site_idx])
                    end
                else # Rightmost
                    if site_idx == 1
                        push!(bond_list_temp, left_ind_list[site_idx])
                    elseif site_idx != N
                        b = Index(2, "b")
                        push!(bond_list_temp, b)    
                    else # Use previous bond for the top end
                        push!(bond_list_temp, bond_list[depth_idx][site_idx])
                    end
                end
            end
        else # Even depth
            for site_idx in 1:N 
                # @infiltrate
                if depth_idx != num_depth-1 # Not the rightmost
                    if site_idx == 1 # Use previous bond for the bottom end
                        push!(bond_list_temp,  bond_list[depth_idx][site_idx])
                    else
                        b = Index(2, "b")
                        push!(bond_list_temp, b)
                    end
                else # Rightmost
                    if site_idx == 1 # Use previous bond for the bottom end
                        push!(bond_list_temp,  bond_list[depth_idx][site_idx])
                    elseif site_idx != N
                        b = Index(2, "b")
                        push!(bond_list_temp, b)
                    else
                        push!(bond_list_temp, left_ind_list[site_idx])
                    end
                end
            end 
        end
        push!(bond_list, bond_list_temp)
    end
    push!(bond_list, left_ind_list)
    return bond_list
end

function get_random_unitary_operator(T)
    # Define a tensor with three indices

    i = Index(2, "i") 
    j = Index(2, "j") 
    k = Index(2, "k")
    l = Index(2, "l")
    random_matrix = [rand()+rand()im for _ in 1:4, _ in 1:4]
    random_tensor = ITensor(random_matrix, (i,j,k,l))

    U, S, V = svd(random_tensor, (i,j))
    U = replaceind(U, commonind(U,S), commonind(S,V))
    gate = U*V
    
    for idx in 1:4
        gate = replaceind(gate, inds(gate)[idx], inds(T)[idx])
    end
    # @infiltrate

    # Initialize with the identity operator
    for a in 1:2
        for b in 1:2
            for c in 1:2
                for d in 1:2
                    if a == c && b == d
                        T[a,b,c,d] = 1.0
                    else
                        T[a,b,c,d] = 0.0
                    end
                end
            end
        end
    end

    # Perturb the identity operator by adding a random unitary
    Td = T + 0.1 * gate

    # Use QR decomposition to make it unitary
    Q, R = qr(Td, (inds(gate)[1],inds(gate)[2],inds(gate)[1] ))

    # Make R a diagonal matrix with sqrt(2)
    R[1,1] = R[2,2] = sqrt(2)
    R[1,2] = R[2,1] = 0.0

    if real(Q[1,1,1,1]) < 0
        Q = -1.0 * Q * R
    else
        Q = Q * R
    end

    # Align indices
    for idx in 1:4
        Q = replaceind(Q, inds(Q)[idx], inds(T)[idx])
    end
    # @infiltrate

    return Q
end

function get_gate_list_init_MPSBW(num_depth, N, bond_list; random = true, seed = 1)
    # Create gates in the form of a 2D list. The first argument is depth, the second is the tensor. All are identity operators
    Random.seed!(seed)
    gate_list = []
    # Create matrices
    for depth_idx in 1:num_depth
        # @infiltrate
        gate_list_temp = []
        if depth_idx % 2 == 1 # Depth odd
            # @infiltrate
            for tensor_idx in 1:(Int((N-1)/2))
                # @infiltrate
                T = ITensor(bond_list[depth_idx][tensor_idx*2], bond_list[depth_idx][tensor_idx*2 - 1], bond_list[depth_idx+1][tensor_idx*2], bond_list[depth_idx+1][tensor_idx*2 - 1])                    
                if random
                    push!(gate_list_temp, get_random_unitary_operator(T)) 
                else
                    push!(gate_list_temp, get_identity_operator(T)) 
                end
                # @infiltrate
            end
        else # Depth even
            for tensor_idx in 1:(Int((N-1)/2))
                # Four-legged tensor
                T = ITensor(bond_list[depth_idx][tensor_idx*2+1], bond_list[depth_idx][tensor_idx*2], bond_list[depth_idx+1][tensor_idx*2+1], bond_list[depth_idx+1][tensor_idx*2])                    
                if random
                    push!(gate_list_temp, get_random_unitary_operator(T)) 
                else
                    push!(gate_list_temp, get_identity_operator(T)) 
                end
            end    
        end
        push!(gate_list, gate_list_temp)
    end
    return gate_list
end

function get_gate_list_init(num_depth, N, bond_list; random = true, seed = 1)
    # Create gates in the form of a 2D list. The first argument is depth, the second is the tensor. All are identity operators
    Random.seed!(seed)
    gate_list = []
    # Create matrices
    for depth_idx in 1:num_depth
        # @infiltrate
        gate_list_temp = []
        if depth_idx % 2 == 1 # Depth odd
            for tensor_idx in 1:(Int(N/2))
                T = ITensor(bond_list[depth_idx][tensor_idx*2], bond_list[depth_idx][tensor_idx*2 - 1], bond_list[depth_idx+1][tensor_idx*2], bond_list[depth_idx+1][tensor_idx*2 - 1])                    
                if random
                    push!(gate_list_temp, get_random_unitary_operator(T)) 
                else
                    push!(gate_list_temp, get_identity_operator(T)) 
                end
                # @infiltrate
            end
        else # Depth even
            for tensor_idx in 1:(Int(N/2))-1
                # Four-legged tensor
                T = ITensor(bond_list[depth_idx][tensor_idx*2+1], bond_list[depth_idx][tensor_idx*2], bond_list[depth_idx+1][tensor_idx*2+1], bond_list[depth_idx+1][tensor_idx*2])                    
                if random
                    push!(gate_list_temp, get_random_unitary_operator(T)) 
                else
                    push!(gate_list_temp, get_identity_operator(T)) 
                end
            end    
        end
        push!(gate_list, gate_list_temp)
    end
    return gate_list
end

function update_upper_env_tensor(env_tensor_idx, evol_MPO, env_tensor_list, N, num_depth, gate_list)
    # Update the upper environment tensor
    # @infiltrate
    if N % 2 == 1
        error("The number of sites must be even")
    end
    # Initial tensor
    if env_tensor_idx == N-2 # Topmost
        env_tensor_temp = ITensor(1.0)
    else # Not the topmost
        env_tensor_temp = env_tensor_list[env_tensor_idx+1]
    end

    if env_tensor_idx % 2 == 0 # Even tensor
        # Multiply tensor
        for depth in 1:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int(env_tensor_idx/2)+1]
            end
        end
        # @infiltrate
        if env_tensor_idx == N-2 # Topmost
            # Multiply MPO
            env_tensor_temp *= evol_MPO[env_tensor_idx+1] * evol_MPO[env_tensor_idx+2]
        else # Not the topmost
            # Multiply MPO
            env_tensor_temp *= evol_MPO[env_tensor_idx+1]
            # Multiply upper tensor for update
        end
        # Update
        env_tensor_list[env_tensor_idx] = env_tensor_temp
        # @infiltrate
    else # Odd tensor
        # Multiply tensor
        for depth in 2:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int((env_tensor_idx-1)/2)+1]
            end
        end
        # Multiply MPO
        env_tensor_temp *= evol_MPO[env_tensor_idx+1]
        # Update
        env_tensor_list[env_tensor_idx] = env_tensor_temp
    end
    return env_tensor_list
end

function update_upper_env_tensor_MPSBW_envMPS(env_tensor_idx, psi, env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=1e-12)
    # Update the upper environment tensor
    # @infiltrate
    if N % 2 == 0
        error("Only applicable for an odd number of sites")
    end
    MPS_temp = MPS(length(env_tensor_list[env_tensor_idx]))
    # @infiltrate
    if env_tensor_idx % 2 == 0 # Even tensor
        for depth in 1:2:num_depth+1
            if depth == 1
                MPS_temp[1] = init_allzero_tensor_list[env_tensor_idx+1] * gate_list[depth][Int(env_tensor_idx/2)+1] * env_tensor_list[env_tensor_idx+1][depth] * env_tensor_list[env_tensor_idx+1][depth+1]
            elseif depth < num_depth
                # @infiltrate
                tensor_temp = env_tensor_list[env_tensor_idx+1][depth] * env_tensor_list[env_tensor_idx+1][depth+1] * gate_list[depth][Int(env_tensor_idx/2)+1]
                idx_left = commonind(env_tensor_list[env_tensor_idx+1][depth-1], env_tensor_list[env_tensor_idx+1][depth])
                idx_lowerleft = commonind(gate_list[depth][Int(env_tensor_idx/2)+1], gate_list[depth-1][Int(env_tensor_idx/2)])
                U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                MPS_temp[depth-1] = U*S
                MPS_temp[depth] = V
            else # Rightmost
                if num_depth % 2 == 0 # Depth even
                    # @infiltrate
                    tensor_temp = psi[env_tensor_idx+1] * env_tensor_list[env_tensor_idx+1][num_depth+1]
                    idx_left = commonind(env_tensor_list[env_tensor_idx+1][num_depth], env_tensor_list[env_tensor_idx+1][num_depth+1])
                    idx_lowerleft = commonind(psi[env_tensor_idx+1], gate_list[num_depth][Int(env_tensor_idx/2)])
                    U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                    MPS_temp[num_depth] = U*S
                    MPS_temp[num_depth+1] = V      
                elseif depth == num_depth # Depth odd
                    tensor_temp = gate_list[num_depth][Int(env_tensor_idx/2)+1] * env_tensor_list[env_tensor_idx+1][num_depth] * env_tensor_list[env_tensor_idx+1][num_depth+1] * env_tensor_list[env_tensor_idx+1][num_depth+2] * psi[env_tensor_idx+1]
                    idx_left = commonind(env_tensor_list[env_tensor_idx+1][num_depth-1], env_tensor_list[env_tensor_idx+1][num_depth])
                    idx_lowerleft = commonind(gate_list[num_depth][Int(env_tensor_idx/2)+1], gate_list[depth-1][Int(env_tensor_idx/2)])
                    U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                    MPS_temp[num_depth-1] = U*S
                    MPS_temp[num_depth] = V 
                end                  
            end
        end
    else # Odd tensor
        for depth in 2:2:num_depth+1
            if env_tensor_idx == N-2 # Topmost
                if depth == 2
                    MPS_temp[1] = init_allzero_tensor_list[env_tensor_idx+1]
                    tensor_temp = init_allzero_tensor_list[env_tensor_idx+2] * gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_left = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-1][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, idx_left;cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                elseif depth < num_depth
                    tensor_temp = gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_left = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-2][Int((env_tensor_idx-1)/2)+1])
                    idx_lowerleft = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-1][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                else # Rightmost
                    if num_depth % 2 == 0 # Depth even
                        tensor_temp = gate_list[num_depth][Int((env_tensor_idx-1)/2)+1]* psi[env_tensor_idx+1] * psi[env_tensor_idx+2] 
                        idx_left = commonind(gate_list[num_depth][Int((env_tensor_idx-1)/2)+1], gate_list[num_depth-2][Int((env_tensor_idx-1)/2)+1])
                        idx_lowerleft = commonind(gate_list[num_depth][Int((env_tensor_idx-1)/2)+1], gate_list[num_depth-1][Int((env_tensor_idx-1)/2)+1])
                        U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                        MPS_temp[num_depth] = U*S
                        MPS_temp[num_depth+1] = V 
                    else # Depth odd
                        tensor_temp = psi[env_tensor_idx+1] * psi[env_tensor_idx+2]
                        idx_left = commonind(psi[env_tensor_idx+2], gate_list[num_depth-1][Int((env_tensor_idx-1)/2)+1])
                        idx_lowerleft = commonind(psi[env_tensor_idx+1], gate_list[num_depth][Int((env_tensor_idx-1)/2)+1])
                        U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                        MPS_temp[num_depth+1] = U*S
                        MPS_temp[num_depth+2] = V      
                    end                  
                end
            else
                if depth == 2
                    MPS_temp[1] = init_allzero_tensor_list[env_tensor_idx+1]
                    tensor_temp = env_tensor_list[env_tensor_idx+1][depth-1] * env_tensor_list[env_tensor_idx+1][depth] * gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_lowerleft = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-1][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, idx_lowerleft;cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                elseif depth < num_depth
                    tensor_temp = env_tensor_list[env_tensor_idx+1][depth-1] * env_tensor_list[env_tensor_idx+1][depth] * gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_left = commonind(env_tensor_list[env_tensor_idx+1][depth-2], env_tensor_list[env_tensor_idx+1][depth-1])
                    idx_lowerleft = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-1][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                else # Rightmost
                    if num_depth % 2 == 0 # Depth even
                        tensor_temp = env_tensor_list[env_tensor_idx+1][num_depth-1] * env_tensor_list[env_tensor_idx+1][num_depth] * env_tensor_list[env_tensor_idx+1][num_depth+1] * gate_list[num_depth][Int((env_tensor_idx-1)/2)+1] * psi[env_tensor_idx+1] 
                        idx_left = commonind(env_tensor_list[env_tensor_idx+1][num_depth-2], env_tensor_list[env_tensor_idx+1][num_depth-1])
                        idx_lowerleft = commonind(gate_list[num_depth][Int((env_tensor_idx-1)/2)+1], gate_list[num_depth-1][Int((env_tensor_idx-1)/2)+1])
                        U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                        MPS_temp[num_depth] = U*S
                        MPS_temp[num_depth+1] = V 
                    else # Depth odd
                        tensor_temp = psi[env_tensor_idx+1] * env_tensor_list[env_tensor_idx+1][num_depth]
                        idx_left = commonind(env_tensor_list[env_tensor_idx+1][num_depth-1], env_tensor_list[env_tensor_idx+1][num_depth])
                        idx_lowerleft = commonind(psi[env_tensor_idx+1], gate_list[num_depth][Int((env_tensor_idx-1)/2)+1])
                        U, S, V = svd(tensor_temp, (idx_left, idx_lowerleft);cutoff=cutoff)
                        # @infiltrate
                        MPS_temp[num_depth+1] = U*S
                        MPS_temp[num_depth+2] = V      
                    end                  
                end
            end
        end
    end
    MPS_temp = sweep_MPS(MPS_temp, length(MPS_temp); cutoff=cutoff)
    env_tensor_list[env_tensor_idx] = MPS_temp
    
    return env_tensor_list
end

function update_upper_env_tensor_MPSBW(env_tensor_idx, psi, env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
    # Update the upper environment tensor
    # @infiltrate
    if N % 2 == 0
        error("This function only supports odd numbers of sites")
    end
    # Initialize the tensor
    if env_tensor_idx == N-2 # Topmost
        env_tensor_temp = init_allzero_tensor_list[env_tensor_idx+1] * init_allzero_tensor_list[env_tensor_idx+2]
    else # Other than the topmost
        env_tensor_temp = env_tensor_list[env_tensor_idx+1] * init_allzero_tensor_list[env_tensor_idx+1]
    end

    if env_tensor_idx %2 == 0 # Even tensor
        # Multiply with the tensor
        for depth in 1:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int(env_tensor_idx/2)+1]
            end
        end
        # @infiltrate
        # Multiply with the MPO
        env_tensor_temp *= psi[env_tensor_idx+1]
        # Update
        env_tensor_list[env_tensor_idx] = env_tensor_temp
        # @infiltrate
    else # Odd tensor
        # Multiply with the tensor
        for depth in 2:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int((env_tensor_idx-1)/2)+1]
            end
        end
        # Multiply with the MPO
        if env_tensor_idx == N-2 # Topmost
            env_tensor_temp *= psi[env_tensor_idx+1] * psi[env_tensor_idx+2]
        else # Other than the topmost
            env_tensor_temp *= psi[env_tensor_idx+1]
        end
        # Update
        env_tensor_list[env_tensor_idx] = env_tensor_temp
    end
    return env_tensor_list
end


function update_lower_env_tensor(env_tensor_idx, evol_MPO, env_tensor_list, N, num_depth, gate_list)
    # Update the lower environment tensor
    if N % 2 == 1 
        error("This function only supports even numbers of sites")
    end
    # @infiltrate
    # Initialize the tensor
    if env_tensor_idx == 1 # Bottom-most
        env_tensor_temp = ITensor(1.0)
    else # Other than the bottom-most
        env_tensor_temp = env_tensor_list[env_tensor_idx-1]
    end

    if env_tensor_idx % 2 == 0 # Even-indexed environment tensor
        # Multiply with the tensor
        for depth in 2:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int(env_tensor_idx/2)]
            end
        end
        # Multiply with the MPO
        env_tensor_temp *= evol_MPO[env_tensor_idx]
        # Update
        # @infiltrate
        env_tensor_list[env_tensor_idx] = env_tensor_temp
    else # Odd-indexed environment tensor
        # Multiply with the tensor
        for depth in 1:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int((env_tensor_idx-1)/2)+1]
            end
        end
        # Multiply with the MPO
        env_tensor_temp *= evol_MPO[env_tensor_idx]
        # Update
        env_tensor_list[env_tensor_idx] = env_tensor_temp
    end
    return env_tensor_list
end

function update_lower_env_tensor_MPSBW(env_tensor_idx, psi, env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
    # Update the lower environment tensor for an MPS block-wise environment
    if N % 2 == 0 
        error("This function only supports odd numbers of sites")
    end
    # @infiltrate
    # Initialize the tensor
    if env_tensor_idx == 1 # Bottom-most
        env_tensor_temp = init_allzero_tensor_list[env_tensor_idx]
    else # Other than the bottom-most
        env_tensor_temp = env_tensor_list[env_tensor_idx-1] * init_allzero_tensor_list[env_tensor_idx]
    end

    if env_tensor_idx % 2 == 0 # Even-indexed environment tensor
        # Multiply with the tensor
        for depth in 2:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int(env_tensor_idx/2)]
            end
        end
        # Multiply with the MPO
        env_tensor_temp *= psi[env_tensor_idx]
        # Update
        # @infiltrate
        env_tensor_list[env_tensor_idx] = env_tensor_temp
    else # Odd-indexed environment tensor
        # Multiply with the tensor
        for depth in 1:2:num_depth
            if depth <= num_depth
                env_tensor_temp *= gate_list[depth][Int((env_tensor_idx-1)/2)+1]
            end
        end
        # Multiply with the MPO
        env_tensor_temp *= psi[env_tensor_idx]
        # Update
        env_tensor_list[env_tensor_idx] = env_tensor_temp
    end
    return env_tensor_list
end

function update_lower_env_tensor_MPSBW_envMPS(env_tensor_idx, psi, env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=1e-12)
    # Update the lower environment tensor for MPS block-wise with environment MPS
    # @infiltrate
    if N % 2 == 0
        error("This function only supports odd numbers of sites")
    end
    MPS_temp = MPS(length(env_tensor_list[env_tensor_idx]))
    # @infiltrate
    if env_tensor_idx % 2 == 0 # Even-indexed environment tensor 
        for depth in 2:2:num_depth+1
            if depth == 2
                tensor_temp = init_allzero_tensor_list[env_tensor_idx] * gate_list[depth][Int(env_tensor_idx/2)] * env_tensor_list[env_tensor_idx-1][depth-1] * env_tensor_list[env_tensor_idx-1][depth] * env_tensor_list[env_tensor_idx-1][depth+1]
                idx_upperleft = commonind(gate_list[depth-1][Int(env_tensor_idx/2)+1], gate_list[depth][Int(env_tensor_idx/2)])
                U, S, V = svd(tensor_temp, idx_upperleft;cutoff=cutoff)
                MPS_temp[depth-1] = U*S
                MPS_temp[depth] = V
            elseif depth < num_depth
                # @infiltrate
                tensor_temp = gate_list[depth][Int(env_tensor_idx/2)] * env_tensor_list[env_tensor_idx-1][depth] * env_tensor_list[env_tensor_idx-1][depth+1]
                idx_upperleft = commonind(gate_list[depth-1][Int(env_tensor_idx/2)+1], gate_list[depth][Int(env_tensor_idx/2)])
                idx_left = commonind(env_tensor_list[env_tensor_idx-1][depth-1], env_tensor_list[env_tensor_idx-1][depth])
                U, S, V = svd(tensor_temp, (idx_upperleft, idx_left);cutoff=cutoff)
                MPS_temp[depth-1] = U*S
                MPS_temp[depth] = V
            else # Rightmost
                if num_depth % 2 == 0 # Depth even
                    # @infiltrate
                    tensor_temp = gate_list[depth][Int(env_tensor_idx/2)] * env_tensor_list[env_tensor_idx-1][depth] * env_tensor_list[env_tensor_idx-1][depth+1] * psi[env_tensor_idx]
                    idx_upperleft = commonind(gate_list[num_depth-1][Int(env_tensor_idx/2)+1], gate_list[num_depth][Int(env_tensor_idx/2)])
                    idx_left = commonind(env_tensor_list[env_tensor_idx-1][num_depth-1], env_tensor_list[env_tensor_idx-1][num_depth])
                    U, S, V = svd(tensor_temp, (idx_upperleft, idx_left);cutoff=cutoff)
                    MPS_temp[num_depth-1] = U*S
                    U2, S2, V2 = svd(V, (commonind(S,V), commonind(gate_list[num_depth][Int(env_tensor_idx/2)], psi[env_tensor_idx+1])))
                    MPS_temp[num_depth] = U2*S2
                    MPS_temp[num_depth+1] = V2
                else # Depth odd
                    # @infiltrate
                    MPS_temp[num_depth] = env_tensor_list[env_tensor_idx-1][num_depth+1] * env_tensor_list[env_tensor_idx-1][num_depth+2] * psi[env_tensor_idx]
                end                  
            end
        end
    else # Odd-indexed environment tensor
        for depth in 1:2:num_depth+2
            if env_tensor_idx == 1 
                if depth == 1
                    tensor_temp = init_allzero_tensor_list[env_tensor_idx] * gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_upperleft = commonind(init_allzero_tensor_list[env_tensor_idx+1], gate_list[depth][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, idx_upperleft;cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                elseif depth <= num_depth
                    tensor_temp = gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_upperleft = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-2][Int((env_tensor_idx-1)/2)+1])
                    idx_left = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-1][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, (idx_upperleft, idx_left);cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                else # Rightmost
                    # @infiltrate
                    if num_depth %2 == 0 
                        MPS_temp[num_depth+1] = psi[env_tensor_idx]
                    else
                        MPS_temp[num_depth+2] = psi[env_tensor_idx]
                    end
                end
            else 
                if depth == 1
                    # @infiltrate
                    tensor_temp = init_allzero_tensor_list[env_tensor_idx] * env_tensor_list[env_tensor_idx-1][depth] * gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_upperleft = commonind(init_allzero_tensor_list[env_tensor_idx+1], gate_list[depth][Int((env_tensor_idx-1)/2)+1])
                    U, S, V = svd(tensor_temp, idx_upperleft;cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                elseif depth < num_depth
                    tensor_temp = env_tensor_list[env_tensor_idx-1][depth-1] * env_tensor_list[env_tensor_idx-1][depth] * gate_list[depth][Int((env_tensor_idx-1)/2)+1]
                    idx_upperleft = commonind(gate_list[depth][Int((env_tensor_idx-1)/2)+1], gate_list[depth-1][Int((env_tensor_idx-1)/2)+1])
                    idx_left = commonind(env_tensor_list[env_tensor_idx-1][depth-2], env_tensor_list[env_tensor_idx-1][depth-1])
                    U, S, V = svd(tensor_temp, (idx_upperleft, idx_left);cutoff=cutoff)
                    MPS_temp[depth] = U*S
                    MPS_temp[depth+1] = V
                    # @infiltrate
                else # Rightmost
                    if num_depth %2 == 0 #depth even
                        # @infiltrate
                        MPS_temp[num_depth+1] = env_tensor_list[env_tensor_idx-1][num_depth] * env_tensor_list[env_tensor_idx-1][num_depth+1] * psi[env_tensor_idx] 
                    elseif depth == num_depth #depth odd
                        # @infiltrate
                        tensor_temp = env_tensor_list[env_tensor_idx-1][num_depth-1] * env_tensor_list[env_tensor_idx-1][num_depth] * gate_list[depth][Int((env_tensor_idx-1)/2)+1] * psi[env_tensor_idx]
                        idx_upperleft = commonind(gate_list[num_depth-1][Int((env_tensor_idx-1)/2)+1], gate_list[num_depth][Int((env_tensor_idx-1)/2)+1])
                        idx_left = commonind(env_tensor_list[env_tensor_idx-1][num_depth-2], env_tensor_list[env_tensor_idx-1][num_depth-1])
                        U, S, V = svd(tensor_temp, (idx_upperleft, idx_left);cutoff=cutoff)
                        # @infiltrate
                        MPS_temp[num_depth] = U*S
                        U2, S2, V2 = svd(V, (commonind(S,V), commonind(gate_list[num_depth][Int((env_tensor_idx-1)/2)+1], psi[env_tensor_idx+1]) );cutoff=cutoff)
                        MPS_temp[num_depth+1] = U2*S2
                        MPS_temp[num_depth+2] = V2         
                    end                  
                end
            end
        end
    end
    # @infiltrate
    MPS_temp = sweep_MPS(MPS_temp, length(MPS_temp); cutoff=cutoff)
    env_tensor_list[env_tensor_idx] = MPS_temp
    # @infiltrate
    
    return env_tensor_list
end

function get_sweep_path_list_MPSBW(N, num_depth)
    # Decide the order of the sweep
    sweep_path_list = []
    for tensor_idx in 1:Int((N-1)/2)
        sweep_path_list_temp = []
        if tensor_idx == 1 # First tensor index
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end
        else
            # When depth is odd, path with one less index
            for depth in 1:num_depth
                if depth %2 == 1
                    push!(sweep_path_list_temp, [depth, tensor_idx])
                else
                    push!(sweep_path_list_temp, [depth, tensor_idx-1])
                end
            end
            
            push!(sweep_path_list, sweep_path_list_temp)
            sweep_path_list_temp = []
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end
        end
        push!(sweep_path_list, sweep_path_list_temp)
    end

    # Reverse order
    for tensor_idx in Int((N-1)/2):-1:1
        sweep_path_list_temp = []
        if tensor_idx == 1 # First tensor index
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end
        else
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end

            push!(sweep_path_list, sweep_path_list_temp)
            sweep_path_list_temp = []
            
            for depth in 1:num_depth
                # When depth is odd, path with one less index
                if depth %2 == 1
                    push!(sweep_path_list_temp, [depth, tensor_idx])
                else
                    push!(sweep_path_list_temp, [depth, tensor_idx-1])
                end
            end
        end
        push!(sweep_path_list, sweep_path_list_temp)
    end
    return sweep_path_list
end



function get_sweep_path_list(N, num_depth)
    # Decide the order of the sweep
    sweep_path_list = []
    for tensor_idx in 1:Int(N/2)
        sweep_path_list_temp = []
        if tensor_idx == 1 # First tensor index
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end
        elseif tensor_idx != Int(N/2)
            # When depth is odd, path with one less index
            for depth in 1:num_depth
                if depth %2 == 1
                    push!(sweep_path_list_temp, [depth, tensor_idx])
                else
                    push!(sweep_path_list_temp, [depth, tensor_idx-1])
                end
            end
            
            push!(sweep_path_list, sweep_path_list_temp)
            sweep_path_list_temp = []
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end
        else # Last tensor index
            # When depth is odd, path with one less index
            for depth in 1:num_depth
                if depth %2 == 1
                    push!(sweep_path_list_temp, [depth, tensor_idx])
                else
                    push!(sweep_path_list_temp, [depth, tensor_idx-1])
                end
            end
        end
        push!(sweep_path_list, sweep_path_list_temp)
    end

    # Reverse order
    for tensor_idx in Int(N/2):-1:1
        sweep_path_list_temp = []
        if tensor_idx == 1 # First tensor index
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end
        elseif tensor_idx != Int(N/2)
            # Path indexed by depth
            for depth in 1:num_depth
                push!(sweep_path_list_temp, [depth, tensor_idx])
            end

            push!(sweep_path_list, sweep_path_list_temp)
            sweep_path_list_temp = []
            
            for depth in 1:num_depth
                # When depth is odd, path with one less index
                if depth %2 == 1
                    push!(sweep_path_list_temp, [depth, tensor_idx])
                else
                    push!(sweep_path_list_temp, [depth, tensor_idx-1])
                end
            end

        else # Last tensor index
            for depth in 1:num_depth
                # When depth is odd, path with one less index
                if depth %2 == 1
                    push!(sweep_path_list_temp, [depth, tensor_idx])
                else
                    push!(sweep_path_list_temp, [depth, tensor_idx-1])
                end
            end
        end
        push!(sweep_path_list, sweep_path_list_temp)
    end
    return sweep_path_list
end

function get_gate_dag(depth_idx, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, evol_MPO)
    # Get gate_dag
    
    tensor_idx_init = depth_tensor_idx[1][2]
    if tensor_idx_init == depth_tensor_idx[2][2] # Path like [1, n], [2, n],...
        if tensor_idx_init == 1 # Bottom
            tensor_temp = upper_env_tensor_list[tensor_idx_init*2]
        else # Not bottom
            tensor_temp = lower_env_tensor_list[tensor_idx_init*2-2]
        end
    else # Path like [1, n], [2, n-1],...
        if tensor_idx_init != Int(N/2) # Not top
            tensor_temp = lower_env_tensor_list[tensor_idx_init*2-3]
        else # Top
            tensor_temp = lower_env_tensor_list[tensor_idx_init*2-3]
        end
    end

    for (depth_path, tensor_path) in depth_tensor_idx
        # Multiply tensors except for the relevant position
        if depth_path == depth_idx && tensor_path == tensor_idx
            nothing
        else
            tensor_temp *= gate_list[depth_path][tensor_path]
        end
    end

    # Multiply remaining environment tensors
    if tensor_idx_init == depth_tensor_idx[2][2] # Path like [1, n], [2, n],...
        if tensor_idx_init == 1 # Bottom
            tensor_temp *= evol_MPO[tensor_idx_init*2] * evol_MPO[tensor_idx_init*2-1]
        else # Not bottom
            tensor_temp *= upper_env_tensor_list[tensor_idx_init*2] * evol_MPO[tensor_idx_init*2] * evol_MPO[tensor_idx_init*2-1]
        end
    else # Path like [1, n], [2, n-1],...
        if tensor_idx_init != Int(N/2) # Not top
            tensor_temp *= upper_env_tensor_list[tensor_idx_init*2-1] * evol_MPO[tensor_idx_init*2-1] * evol_MPO[tensor_idx_init*2-2]
        else # Top
            tensor_temp *= evol_MPO[tensor_idx_init*2] * evol_MPO[tensor_idx_init*2-1] * evol_MPO[tensor_idx_init*2-2]
        end
    end

    return tensor_temp
end

function get_gate_dag_MPSBW_envMPS_reducebonddim(depth_idx, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list;cutoff=1e-12)
    # Get gate_dag
    # Not adopted
    # This was an attempt to prevent the bond dimension from increasing during contraction, but the bond count increased and the process became slower.
    # @infiltrate
    tensor_temp = ITensor(1.0)
    target_MPS = ITensor(1.0)
    tensor_idx_init = depth_tensor_idx[1][2]
    if tensor_idx_init == depth_tensor_idx[2][2] # When the path is [1, n], [2, n],...
        # @infiltrate
        if length(depth_tensor_idx) % 2 == 0
            target_MPS = MPS(length(depth_tensor_idx)+1)
        else length(depth_tensor_idx) % 2 == 1
            target_MPS = MPS(length(depth_tensor_idx))
        end    

        if tensor_idx_init == 1 # Bottom
            for target_idx in 1:length(depth_tensor_idx)
                (depth_path, tensor_path) = depth_tensor_idx[target_idx]
                if target_idx != length(depth_tensor_idx)
                    (depth_path_after, tensor_path_after) = depth_tensor_idx[target_idx+1]
                end
                if target_idx == 1
                    target_MPS[target_idx] = init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
                    if !(depth_path == depth_idx && tensor_path == tensor_idx)
                        target_MPS[target_idx] *= gate_list[depth_path][tensor_path]
                    end
                    if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                        target_MPS[target_idx] *= gate_list[depth_path_after][tensor_path_after]
                    end
                    idx_upperleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after+1])
                    U, S, V = svd(target_MPS[target_idx], idx_upperleft;cutoff=cutoff)
                    target_MPS[target_idx] = U*S
                    target_MPS[target_idx+1] = V
                    # @infiltrate
                elseif target_idx != length(depth_tensor_idx)
                    if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                        target_MPS[target_idx] *= gate_list[depth_path_after][tensor_path_after]
                    end
                    idx_left = commonind(target_MPS[target_idx-1], target_MPS[target_idx])
                    if target_idx %2 == 1
                        idx_upperleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after+1])
                        U, S, V = svd(target_MPS[target_idx], (idx_upperleft, idx_left);cutoff=cutoff)
                        target_MPS[target_idx] = U*S
                        target_MPS[target_idx+1] = V
                    else
                        idx_upperright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path_after+1])
                        U, S, V = svd(target_MPS[target_idx], (idx_upperright, idx_left);cutoff=cutoff)
                        target_MPS[target_idx] = U*S
                        target_MPS[target_idx+1] = V
                    end
                    # @infiltrate
                else # Rightmost
                    target_MPS[target_idx] *= psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1] 
                    if length(depth_tensor_idx) %2 == 0
                        idx_upperright = commonind(gate_list[depth_path][tensor_path], psi[tensor_idx_init*2+1])
                        idx_left = commonind(target_MPS[target_idx-1], target_MPS[target_idx])
                        U, S, V = svd(target_MPS[target_idx], (idx_upperright, idx_left);cutoff=cutoff)
                        target_MPS[target_idx] = U*S
                        target_MPS[target_idx+1] = V
                    end
                    # @infiltrate
                end
            end
            for target_idx in 1:length(target_MPS)
                tensor_temp *= target_MPS[target_idx] * upper_env_tensor_list[tensor_idx_init*2][target_idx]
            end
            # @infiltrate
        elseif tensor_idx_init == Int((N-1)/2) # Top
            # @infiltrate
            for target_idx in 1:length(depth_tensor_idx)
                (depth_path, tensor_path) = depth_tensor_idx[target_idx]
                if target_idx != length(depth_tensor_idx)
                    (depth_path_after, tensor_path_after) = depth_tensor_idx[target_idx+1]
                end

                if target_idx == 1
                    target_MPS[target_idx] = init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
                    if !(depth_path == depth_idx && tensor_path == tensor_idx)
                        target_MPS[target_idx] *= gate_list[depth_path][tensor_path]
                    end
                    target_MPS[target_idx] *= init_allzero_tensor_list[tensor_idx_init*2+1]
                    if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                        target_MPS[target_idx] *= gate_list[depth_path_after][tensor_path_after]
                    end
                    idx_lowerright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path-1])
                    U, S, V = svd(target_MPS[target_idx], idx_lowerright;cutoff=cutoff)
                    target_MPS[target_idx] = U*S
                    target_MPS[target_idx+1] = V
                    # @infiltrate
                elseif target_idx != length(depth_tensor_idx)
                    if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                        target_MPS[target_idx] *= gate_list[depth_path_after][tensor_path_after]
                    end
                    idx_left = commonind(target_MPS[target_idx-1], target_MPS[target_idx])
                    if target_idx %2 == 1
                        idx_lowerright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path-1])
                        U, S, V = svd(target_MPS[target_idx], (idx_left, idx_lowerright);cutoff=cutoff)
                        target_MPS[target_idx] = U*S
                        target_MPS[target_idx+1] = V
                    else
                        idx_lowerleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after-1])
                        U, S, V = svd(target_MPS[target_idx], (idx_left, idx_lowerleft);cutoff=cutoff)
                        target_MPS[target_idx] = U*S
                        target_MPS[target_idx+1] = V
                    end
                    # @infiltrate
                else # Rightmost
                    target_MPS[target_idx] *= psi[tensor_idx_init*2+1] * psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
                    if length(depth_tensor_idx) %2 == 0
                        idx_lowerleft = commonind(gate_list[depth_path][tensor_path-1], psi[tensor_idx_init*2-1])
                        idx_left = commonind(target_MPS[target_idx-1], target_MPS[target_idx])
                        U, S, V = svd(target_MPS[target_idx], (idx_lowerleft, idx_left);cutoff=cutoff)
                        target_MPS[target_idx] = U*S
                        target_MPS[target_idx+1] = V
                    end
                    # @infiltrate
                end
            end
            for target_idx in 1:length(target_MPS)
                # @infiltrate
                tensor_temp *= target_MPS[target_idx] * lower_env_tensor_list[tensor_idx_init*2-2][target_idx]
            end

        else # Other than bottom and top
            target_MPO = MPO(length(target_MPS))
            for target_idx in 1:length(depth_tensor_idx)
                (depth_path, tensor_path) = depth_tensor_idx[target_idx]
                if target_idx != length(depth_tensor_idx)
                    (depth_path_after, tensor_path_after) = depth_tensor_idx[target_idx+1]
                end

                if target_idx == 1
                    target_MPO[target_idx] = init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
                    if !(depth_path == depth_idx && tensor_path == tensor_idx)
                        target_MPO[target_idx] *= gate_list[depth_path][tensor_path]
                    end
                    if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                        target_MPO[target_idx] *= gate_list[depth_path_after][tensor_path_after]
                    end
                    idx_lowerright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path-1])
                    idx_upperleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after+1])
                    U, S, V = svd(target_MPO[target_idx], (idx_lowerright, idx_upperleft);cutoff=cutoff)
                    target_MPO[target_idx] = U*S
                    target_MPO[target_idx+1] = V
                    # @infiltrate
                elseif target_idx != length(depth_tensor_idx)
                    if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                        target_MPO[target_idx] *= gate_list[depth_path_after][tensor_path_after]
                    end
                    idx_left = commonind(target_MPO[target_idx-1], target_MPO[target_idx])
                    if target_idx %2 == 1
                        idx_lowerright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path-1])
                        idx_upperleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after+1])
                        U, S, V = svd(target_MPO[target_idx], (idx_lowerright, idx_upperleft, idx_left);cutoff=cutoff)
                        target_MPO[target_idx] = U*S
                        target_MPO[target_idx+1] = V
                    else
                        idx_upperright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path_after+1])
                        idx_lowerleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after-1])
                        U, S, V = svd(target_MPO[target_idx], (idx_upperright, idx_lowerleft, idx_left);cutoff=cutoff)
                        target_MPO[target_idx] = U*S
                        target_MPO[target_idx+1] = V
                    end
                    # @infiltrate
                else # Rightmost
                    target_MPO[target_idx] *= psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
                    if length(depth_tensor_idx) %2 == 0
                        idx_upperright = commonind(gate_list[depth_path][tensor_path], psi[tensor_idx_init*2+1])
                        idx_lowerleft = commonind(gate_list[depth_path][tensor_path-1], psi[tensor_idx_init*2-1])
                        idx_left = commonind(target_MPO[target_idx-1], target_MPO[target_idx])
                        U, S, V = svd(target_MPO[target_idx], (idx_upperright, idx_lowerleft, idx_left);cutoff=cutoff)
                        target_MPO[target_idx] = U*S
                        target_MPO[target_idx+1] = V
                    end
                    
                end
            end
            for target_idx in 1:length(target_MPO)
                tensor_temp *= target_MPO[target_idx] * upper_env_tensor_list[tensor_idx_init*2][target_idx] * lower_env_tensor_list[tensor_idx_init*2-2][target_idx]
            end
            
        end
    else # When the path is [1, n], [2, n-1],...
        if length(depth_tensor_idx) %2 == 0
            target_MPO = MPO(length(depth_tensor_idx)+1)
        else length(depth_tensor_idx) %2 == 1
            target_MPO = MPO(length(depth_tensor_idx)+2)
        end    
        for target_idx in 1:length(depth_tensor_idx)
            (depth_path, tensor_path) = depth_tensor_idx[target_idx]
            if target_idx != length(depth_tensor_idx)
                (depth_path_after, tensor_path_after) = depth_tensor_idx[target_idx+1]
            end

            if target_idx == 1
                target_MPO[target_idx] = init_allzero_tensor_list[tensor_idx_init*2-1] * init_allzero_tensor_list[tensor_idx_init*2-2]
                if !(depth_path == depth_idx && tensor_path == tensor_idx)
                    target_MPO[target_idx] *= gate_list[depth_path][tensor_path]
                end
                U, S, V = svd(target_MPO[target_idx], (inds(init_allzero_tensor_list[tensor_idx_init*2-2])[1], inds(init_allzero_tensor_list[tensor_idx_init*2])[1]);cutoff=cutoff)
                target_MPO[target_idx] = U*S
                target_MPO[target_idx+1] = V

                if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                    target_MPO[target_idx+1] *= gate_list[depth_path_after][tensor_path_after]
                end

                idx_left = commonind(target_MPO[target_idx],target_MPO[target_idx+1])
                idx_upperright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path])
                idx_lowerleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after])
                U2, S2, V2 = svd(target_MPO[target_idx+1], (idx_upperright, idx_lowerleft, idx_left);cutoff=cutoff)
                target_MPO[target_idx+1] = U2*S2
                target_MPO[target_idx+2] = V2
                # @infiltrate
            elseif target_idx != length(depth_tensor_idx)
                if !(depth_path_after == depth_idx && tensor_path_after == tensor_idx)
                    target_MPO[target_idx+1] *= gate_list[depth_path_after][tensor_path_after]
                end
                idx_left = commonind(target_MPO[target_idx], target_MPO[target_idx+1])
                if target_idx %2 ==1
                    idx_upperright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path])
                    idx_lowerleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after])
                    U, S, V = svd(target_MPO[target_idx+1], (idx_upperright, idx_lowerleft, idx_left);cutoff=cutoff)
                    target_MPO[target_idx+1] = U*S
                    target_MPO[target_idx+2] = V
                else
                    idx_lowerright = commonind(gate_list[depth_path][tensor_path], gate_list[depth_path+1][tensor_path_after])
                    idx_upperleft = commonind(gate_list[depth_path_after][tensor_path_after], gate_list[depth_path_after-1][tensor_path_after])
                    U, S, V = svd(target_MPO[target_idx+1], (idx_lowerright, idx_upperleft, idx_left);cutoff=cutoff)
                    target_MPO[target_idx+1] = U*S
                    target_MPO[target_idx+2] = V
                end
            else # Rightmost
                target_MPO[target_idx+1] *= psi[tensor_idx_init*2-1] * psi[tensor_idx_init*2-2]
                if length(depth_tensor_idx) %2 == 1
                    idx_upperright = commonind(gate_list[depth_path][tensor_path], psi[tensor_idx_init*2])
                    idx_lowerleft = commonind(gate_list[depth_path][tensor_path-1], psi[tensor_idx_init*2-2])
                    idx_left = commonind(target_MPO[target_idx], target_MPO[target_idx+1])
                    U, S, V = svd(target_MPO[target_idx+1], (idx_upperright, idx_lowerleft, idx_left);cutoff=cutoff)
                    target_MPO[target_idx+1] = U*S
                    target_MPO[target_idx+2] = V
                end
                # @infiltrate
            end
        end
        for target_idx in 1:length(target_MPO)
            tensor_temp *= target_MPO[target_idx] * upper_env_tensor_list[tensor_idx_init*2-1][target_idx] * lower_env_tensor_list[tensor_idx_init*2-3][target_idx]
        end
    end
    return tensor_temp
end

function get_gate_dag_MPSBW_envMPS(depth_idx, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list)
    # Get gate_dag

    # @infiltrate
    tensor_idx_init = depth_tensor_idx[1][2]
    if tensor_idx_init == depth_tensor_idx[2][2] # When the path is [1, n], [2, n],...
        # @infiltrate
        if tensor_idx_init == 1 # Bottom
            tensor_temp = init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
            upper_env_tensor = upper_env_tensor_list[tensor_idx_init*2]
        elseif tensor_idx_init == Int((N-1)/2) # Top
            # @infiltrate
            tensor_temp = init_allzero_tensor_list[tensor_idx_init*2+1] * init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
            lower_env_tensor = lower_env_tensor_list[tensor_idx_init*2-2]
        else # Other than bottom and top
            # @infiltrate
            tensor_temp = init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
            lower_env_tensor = lower_env_tensor_list[tensor_idx_init*2-2]
            upper_env_tensor = upper_env_tensor_list[tensor_idx_init*2]
        end
    else # When the path is [1, n], [2, n-1],...
        tensor_temp = init_allzero_tensor_list[tensor_idx_init*2-1] * init_allzero_tensor_list[tensor_idx_init*2-2]
        lower_env_tensor = lower_env_tensor_list[tensor_idx_init*2-3]
        upper_env_tensor = upper_env_tensor_list[tensor_idx_init*2-1]
        # @infiltrate
    end

    for (env_idx,(depth_path, tensor_path)) in enumerate(depth_tensor_idx)
        # Multiply tensors except for the specified position
        # @infiltrate
        if depth_path == depth_idx && tensor_path == tensor_idx
            nothing
        else
            tensor_temp *= gate_list[depth_path][tensor_path]
        end

        if tensor_idx_init == depth_tensor_idx[2][2] # When the path is [1, n], [2, n],...
            # @infiltrate
            if tensor_idx_init == 1 # Bottom
                tensor_temp *= upper_env_tensor[env_idx] 
            elseif tensor_idx_init == Int((N-1)/2) # Top
                # @infiltrate
                tensor_temp *= lower_env_tensor[env_idx] 
            else # Other than bottom and top
                # @infiltrate
                tensor_temp *= lower_env_tensor[env_idx] * upper_env_tensor[env_idx]
            end
        else # When the path is [1, n], [2, n-1],...
            # Both upper and lower environment tensors must exist
            tensor_temp *= lower_env_tensor[env_idx] * upper_env_tensor[env_idx]
            # @infiltrate
        end
    end

    # Multiply the remaining environment tensors
    if tensor_idx_init == depth_tensor_idx[2][2] # When the path is [1, n], [2, n],...
        if tensor_idx_init == 1 # Bottom
            tensor_temp *= psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1] 
            if length(upper_env_tensor) > length(depth_tensor_idx)
                for remain_idx in length(depth_tensor_idx)+1:length(upper_env_tensor)
                    tensor_temp *= upper_env_tensor[remain_idx]
                end
            end
        elseif tensor_idx_init == Int((N-1)/2) # Top
            tensor_temp *= psi[tensor_idx_init*2+1] * psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
            if length(lower_env_tensor) > length(depth_tensor_idx)
                for remain_idx in length(depth_tensor_idx)+1:length(lower_env_tensor)
                    tensor_temp *= lower_env_tensor[remain_idx]
                end
            end
        else # Other than bottom and top
            # @infiltrate
            tensor_temp *= psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
            if length(upper_env_tensor) > length(depth_tensor_idx) 
                for remain_idx in length(depth_tensor_idx)+1:length(upper_env_tensor)
                    tensor_temp *= upper_env_tensor[remain_idx] * lower_env_tensor[remain_idx]
                end
            end
            # @infiltrate
        end
    else # When the path is [1, n], [2, n-1],...
        # Both upper and lower environment tensors must exist
        tensor_temp *= psi[tensor_idx_init*2-1] * psi[tensor_idx_init*2-2]
        if length(lower_env_tensor) > length(depth_tensor_idx)
            for remain_idx in length(depth_tensor_idx)+1:length(lower_env_tensor)
                tensor_temp *= upper_env_tensor[remain_idx] * lower_env_tensor[remain_idx]
            end
        end
    end
    # @infiltrate
    return tensor_temp
end

function get_gate_dag_MPSBW(depth_idx, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list)
    # Get gate_dag
    
    # @infiltrate
    tensor_idx_init = depth_tensor_idx[1][2]
    if tensor_idx_init == depth_tensor_idx[2][2] # When the path is [1, n], [2, n],...
        # @infiltrate
        if tensor_idx_init == 1 # Bottom
            tensor_temp = upper_env_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
        elseif tensor_idx_init == Int((N-1)/2) # Top
            # @infiltrate
            tensor_temp = lower_env_tensor_list[tensor_idx_init*2-2] * init_allzero_tensor_list[tensor_idx_init*2+1] * init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
        else # Other than bottom and top
            # @infiltrate
            tensor_temp = lower_env_tensor_list[tensor_idx_init*2-2] * init_allzero_tensor_list[tensor_idx_init*2] * init_allzero_tensor_list[tensor_idx_init*2-1]
        end
    else # When the path is [1, n], [2, n-1],...
        # Both upper and lower environment tensors must exist
        tensor_temp = lower_env_tensor_list[tensor_idx_init*2-3] * init_allzero_tensor_list[tensor_idx_init*2-1] * init_allzero_tensor_list[tensor_idx_init*2-2]
        # @infiltrate
    end

    for (depth_path, tensor_path) in depth_tensor_idx
        # Multiply tensors except for the specified position
        if depth_path == depth_idx && tensor_path == tensor_idx
            nothing
        else
            tensor_temp *= gate_list[depth_path][tensor_path]
        end
    end

    # Multiply the remaining environment tensors
    if tensor_idx_init == depth_tensor_idx[2][2] # When the path is [1, n], [2, n],...
        if tensor_idx_init == 1 # Bottom
            tensor_temp *= psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
        elseif tensor_idx_init == Int((N-1)/2) # Top
            tensor_temp *= psi[tensor_idx_init*2+1] * psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
        else # Other than bottom and top
            # @infiltrate
            tensor_temp *= upper_env_tensor_list[tensor_idx_init*2] * psi[tensor_idx_init*2] * psi[tensor_idx_init*2-1]
            # @infiltrate
        end
    else # When the path is [1, n], [2, n-1],...
        # Both upper and lower environment tensors must exist
        tensor_temp *= upper_env_tensor_list[tensor_idx_init*2-1] * psi[tensor_idx_init*2-1] * psi[tensor_idx_init*2-2]
    end

    return tensor_temp
end

function get_opt_gate(gate_dag, depth_idx, tensor_idx, gate_list_init; cutoff = 1e-12)
    # Obtain the optimal gate from gate_dag. This method is not used due to a problem where the cost value decreases drastically.
    # Get the legs (indices)
    # @infiltrate
    i = commonind(gate_dag, inds(gate_list_init[depth_idx][tensor_idx])[1])
    j = commonind(gate_dag, inds(gate_list_init[depth_idx][tensor_idx])[2])
    k = commonind(gate_dag, inds(gate_list_init[depth_idx][tensor_idx])[3])
    l = commonind(gate_dag, inds(gate_list_init[depth_idx][tensor_idx])[4])
    # @infiltrate

    gate = ITensor(i,j,k,l)
    # Take the conjugate
    for in1 in 1:2
        for in2 in 1:2
            for out1 in 1:2
                for out2 in 1:2
                    # @infiltrate
                    # gate[in1,in2,out1,out2] = conj(gate_dag[out1, out2, in1, in2])
                    gate[in1, in2, out1, out2] = conj(gate_dag[i => out1, j => out2, k => in1, l => in2])
                end
            end
        end
    end

    # Perform SVD and update U and V tensors
    # @infiltrate
    U, S, V = svd(gate, (i,j); cutoff = cutoff)
    U = replaceind(U, commonind(U,S), commonind(S,V))
    # @infiltrate

    return U*V
end

function get_gate_dag_wo_env_tensor(depth_target_idx, tensor_target_idx, gate_list, evol_MPO)
    tensor_temp = ITensor(1.0)
    # Apply time evolution MPO
    # @infiltrate
    for MPO in evol_MPO
        tensor_temp *= MPO
    end

    # Apply tensors except for the target position
    for (depth_idx, gate_list_single_depth) in enumerate(gate_list)
        for (tensor_idx, gate) in enumerate(gate_list_single_depth)
            # @infiltrate
            if depth_idx == depth_target_idx && tensor_idx == tensor_target_idx
                nothing
            else
                tensor_temp *= gate
            end
        end
    end
    # For testing, apply the target tensor and display cost
    # @infiltrate
    # open("QPDE/costoutput.txt", "a") do file
    #     write(file, string(real(array(tensor_temp * gate_list[depth_target_idx][tensor_target_idx])[1])), "\n")
    # end

    return tensor_temp
end


function get_gate_dag_wo_env_tensor_MPSBW(depth_target_idx, tensor_target_idx, gate_list, psi, init_allzero_tensor_list)
    tensor_temp = ITensor(1.0)
    # Apply time evolution MPO
    # @infiltrate
    for ope in psi
        tensor_temp *= ope
    end
    for ope in init_allzero_tensor_list
        tensor_temp *= ope
    end

    # Apply tensors except for the target position
    for (depth_idx, gate_list_single_depth) in enumerate(gate_list)
        for (tensor_idx, gate) in enumerate(gate_list_single_depth)
            # @infiltrate
            if depth_idx == depth_target_idx && tensor_idx == tensor_target_idx
                nothing
            else
                tensor_temp *= gate
            end
        end
    end
    # For testing, apply the target tensor and display cost
    # @infiltrate
    # open("QPDE/costoutput.txt", "a") do file
    #     write(file, string(real(array(tensor_temp * gate_list[depth_target_idx][tensor_target_idx])[1])), "\n")
    # end

    return tensor_temp
end


function get_sweeped_gate_list(num_depth, gate_list, N, upper_env_tensor_list, lower_env_tensor_list, sweep_path_list, evol_MPO, gate_list_init; cutoff = 1e-12)
    # Sweep the gate list of the time evolution
    # @infiltrate
    length_half_sweep = Int(length(sweep_path_list)/2)
    for (idx, depth_tensor_idx) in enumerate(sweep_path_list[1:length_half_sweep])
        for (depth, tensor_idx) in depth_tensor_idx
            # Get the † of the gate
            # @infiltrate
            gate_dag = get_gate_dag(depth, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, evol_MPO)
            
            # noenv
            # gate_dag = get_gate_dag_wo_env_tensor(depth, tensor_idx, gate_list, evol_MPO)

            # # Convert dag to gate
            # gate_dag_per = permute(gate_dag, inds(gate_list_init[depth][tensor_idx]))
            # gate_per = dag(gate_dag_per)
            # @infiltrate
            # # gate_per = ITensor(adjoint(get_unitary_from_tensor(gate_dag_per)), inds(gate_list_init[depth][tensor_idx]))
            gate_per = conj(permute(gate_dag, (inds(gate_list_init[depth][tensor_idx])[3], inds(gate_list_init[depth][tensor_idx])[4], inds(gate_list_init[depth][tensor_idx])[1], inds(gate_list_init[depth][tensor_idx])[2] )))
            # @infiltrate
            U, S, V = svd(gate_per, inds(gate_list_init[depth][tensor_idx])[1:2]; cutoff = cutoff, mindim=4)
            # @infiltrate
            U = replaceind(U, commonind(U,S), commonind(S,V))
            # U = replaceind(U, commonind(U,S), commonind(S,V))
            gate = U*V
            
            
            # gate = get_opt_gate(gate_dag, depth, tensor_idx, gate_list_init)
            
            # Replace the gate
            # @infiltrate
            gate_list[depth][tensor_idx] = gate
            # @infiltrate
        end

        # Update environment tensors
        if idx == 1
            update_lower_env_tensor(idx, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
            # @infiltrate
        elseif idx != Int(length(sweep_path_list)/2)
            update_upper_env_tensor(idx-1, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
            update_lower_env_tensor(idx, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
            # @infiltrate
        else
            update_upper_env_tensor(idx-1, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
            update_upper_env_tensor(idx, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
            update_lower_env_tensor(idx, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
            # @infiltrate
        end
        # println(real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1]))
    end
    for (idx, depth_tensor_idx) in enumerate(sweep_path_list[length_half_sweep+1:length(sweep_path_list)])
        for (depth, tensor_idx) in depth_tensor_idx
            # Get the † of the gate
            gate_dag = get_gate_dag(depth, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, evol_MPO)

            # noenv
            # gate_dag = gate_dag = get_gate_dag_wo_env_tensor(depth, tensor_idx, gate_list, evol_MPO)

            # # Convert dag to gate
            # # @infiltrate
            # gate_dag_per = permute(gate_dag, inds(gate_list_init[depth][tensor_idx]))
            # gate_per = dag(gate_dag_per)
            # # gate_per = ITensor(adjoint(get_unitary_from_tensor(gate_dag_per)), inds(gate_list_init[depth][tensor_idx]))
            gate_per = conj(permute(gate_dag, (inds(gate_list_init[depth][tensor_idx])[3], inds(gate_list_init[depth][tensor_idx])[4], inds(gate_list_init[depth][tensor_idx])[1], inds(gate_list_init[depth][tensor_idx])[2] )))
            U, S, V = svd(gate_per, inds(gate_list_init[depth][tensor_idx])[1:2]; cutoff = cutoff, mindim=4)
            U = replaceind(U, commonind(U,S), commonind(S,V))
            gate = U*V

            # gate = get_opt_gate(gate_dag, depth, tensor_idx, gate_list_init)
            
            gate_list[depth][tensor_idx] = gate
            # @infiltrate
        end
        
        # Update environment tensors
        if idx == 1
            update_upper_env_tensor(length_half_sweep - idx + 1, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
        elseif idx != Int(length(sweep_path_list)/2)
            update_upper_env_tensor(length_half_sweep - idx + 1, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
            update_lower_env_tensor(length_half_sweep - idx + 2, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
        else
            update_upper_env_tensor(length_half_sweep - idx + 1, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
            update_lower_env_tensor(length_half_sweep - idx + 2, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
            update_lower_env_tensor(length_half_sweep - idx + 1, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
        end
        # println(real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1]))
    end
    cost = real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1])
    # @infiltrate
    return gate_list, upper_env_tensor_list, lower_env_tensor_list, cost
end

function get_sweeped_gate_list_MPSBW_envMPS(num_depth, gate_list, N, upper_env_tensor_list, lower_env_tensor_list, sweep_path_list, psi, gate_list_init, init_allzero_tensor_list; cutoff = 1e-12)
    # Sweep the gate list for time evolution
    # @infiltrate
    length_half_sweep = Int(length(sweep_path_list)/2)
    for (idx, depth_tensor_idx) in enumerate(sweep_path_list[1:length_half_sweep])
        for (depth, tensor_idx) in depth_tensor_idx
            # Obtain the dagger of the gate
            # @infiltrate
            gate_dag = get_gate_dag_MPSBW_envMPS(depth, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list)
            # @infiltrate
            gate_per = conj(permute(gate_dag, (inds(gate_list_init[depth][tensor_idx])[3], inds(gate_list_init[depth][tensor_idx])[4], inds(gate_list_init[depth][tensor_idx])[1], inds(gate_list_init[depth][tensor_idx])[2] )))
            U, S, V = svd(gate_per, inds(gate_list_init[depth][tensor_idx])[1:2]; cutoff = cutoff, mindim=4)
            # @infiltrate
            U = replaceind(U, commonind(U,S), commonind(S,V))
            gate = U*V
            gate_list[depth][tensor_idx] = gate
            # @infiltrate
        end
        # @infiltrate

        # @infiltrate
        # Update environment tensors
        if idx == 1
            update_lower_env_tensor_MPSBW_envMPS(idx, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            # @infiltrate
        elseif idx != Int(length(sweep_path_list)/2)
            update_upper_env_tensor_MPSBW_envMPS(idx-1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            update_lower_env_tensor_MPSBW_envMPS(idx, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            # @infiltrate
        else
            update_upper_env_tensor_MPSBW_envMPS(idx-1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            update_upper_env_tensor_MPSBW_envMPS(idx, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            update_lower_env_tensor_MPSBW_envMPS(idx, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            # @infiltrate
        end
        # println(real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1]))
        # @infiltrate
    end
    for (idx, depth_tensor_idx) in enumerate(sweep_path_list[length_half_sweep+1:length(sweep_path_list)])
        for (depth, tensor_idx) in depth_tensor_idx
            # Obtain the dagger of the gate
            gate_dag = get_gate_dag_MPSBW_envMPS(depth, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list)
            # gate_dag = get_gate_dag_wo_env_tensor_MPSBW_envMPS(depth, tensor_idx, gate_list, psi, init_allzero_tensor_list)
            gate_per = conj(permute(gate_dag, (inds(gate_list_init[depth][tensor_idx])[3], inds(gate_list_init[depth][tensor_idx])[4], inds(gate_list_init[depth][tensor_idx])[1], inds(gate_list_init[depth][tensor_idx])[2] )))
            U, S, V = svd(gate_per, inds(gate_list_init[depth][tensor_idx])[1:2]; cutoff = cutoff, mindim=4)
            U = replaceind(U, commonind(U,S), commonind(S,V))
            gate = U*V

            # gate = get_opt_gate(gate_dag, depth, tensor_idx, gate_list_init)
            
            gate_list[depth][tensor_idx] = gate
            # @infiltrate
        end
        # @infiltrate
        
        # Update environment tensors
        if idx == 1
            update_upper_env_tensor_MPSBW_envMPS(length_half_sweep - idx + 1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
        elseif idx != Int(length(sweep_path_list)/2)
            update_upper_env_tensor_MPSBW_envMPS(length_half_sweep - idx + 1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            update_lower_env_tensor_MPSBW_envMPS(length_half_sweep - idx + 2, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
        else
            update_upper_env_tensor_MPSBW_envMPS(length_half_sweep - idx + 1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            update_lower_env_tensor_MPSBW_envMPS(length_half_sweep - idx + 2, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
            update_lower_env_tensor_MPSBW_envMPS(length_half_sweep - idx + 1, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list;cutoff=cutoff)
        end
        # @infiltrate
        # println(real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1]))
    end
    cost = ITensor(1.0)
    for idx in 1:length(upper_env_tensor_list[1])
        cost *= upper_env_tensor_list[1][idx] * lower_env_tensor_list[1][idx]
    end
    cost = real(array(cost))[1]
    # @infiltrate
    return gate_list, upper_env_tensor_list, lower_env_tensor_list, cost
end 

function get_sweeped_gate_list_MPSBW(num_depth, gate_list, N, upper_env_tensor_list, lower_env_tensor_list, sweep_path_list, psi, gate_list_init, init_allzero_tensor_list; cutoff = 1e-12)
    # Sweep the gate list for time evolution
    # @infiltrate
    length_half_sweep = Int(length(sweep_path_list)/2)
    for (idx, depth_tensor_idx) in enumerate(sweep_path_list[1:length_half_sweep])
        for (depth, tensor_idx) in depth_tensor_idx
            # Obtain the dagger of the gate
            # @infiltrate
            gate_dag = get_gate_dag_MPSBW(depth, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list)
            # gate_dag = get_gate_dag_wo_env_tensor_MPSBW(depth, tensor_idx, gate_list, psi, init_allzero_tensor_list)
            # @infiltrate
            gate_per = conj(permute(gate_dag, (inds(gate_list_init[depth][tensor_idx])[3], inds(gate_list_init[depth][tensor_idx])[4], inds(gate_list_init[depth][tensor_idx])[1], inds(gate_list_init[depth][tensor_idx])[2] )))
            U, S, V = svd(gate_per, inds(gate_list_init[depth][tensor_idx])[1:2]; cutoff = cutoff, mindim=4)
            # @infiltrate
            U = replaceind(U, commonind(U,S), commonind(S,V))
            gate = U*V
            gate_list[depth][tensor_idx] = gate
            # if depth == 3
            #     @infiltrate
            # end
        end

        # Update environment tensors
        if idx == 1
            update_lower_env_tensor_MPSBW(idx, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            # @infiltrate
        elseif idx != Int(length(sweep_path_list)/2)
            update_upper_env_tensor_MPSBW(idx-1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            update_lower_env_tensor_MPSBW(idx, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            # @infiltrate
        else
            update_upper_env_tensor_MPSBW(idx-1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            update_upper_env_tensor_MPSBW(idx, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            update_lower_env_tensor_MPSBW(idx, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            # @infiltrate
        end
        # println(real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1]))
    end
    for (idx, depth_tensor_idx) in enumerate(sweep_path_list[length_half_sweep+1:length(sweep_path_list)])
        for (depth, tensor_idx) in depth_tensor_idx
            # Obtain the dagger of the gate
            gate_dag = get_gate_dag_MPSBW(depth, tensor_idx, depth_tensor_idx, upper_env_tensor_list, lower_env_tensor_list, N, gate_list, psi, init_allzero_tensor_list)
            # gate_dag = get_gate_dag_wo_env_tensor_MPSBW(depth, tensor_idx, gate_list, psi, init_allzero_tensor_list)
            gate_per = conj(permute(gate_dag, (inds(gate_list_init[depth][tensor_idx])[3], inds(gate_list_init[depth][tensor_idx])[4], inds(gate_list_init[depth][tensor_idx])[1], inds(gate_list_init[depth][tensor_idx])[2] )))
            U, S, V = svd(gate_per, inds(gate_list_init[depth][tensor_idx])[1:2]; cutoff = cutoff, mindim=4)
            U = replaceind(U, commonind(U,S), commonind(S,V))
            gate = U*V

            # gate = get_opt_gate(gate_dag, depth, tensor_idx, gate_list_init)
            
            gate_list[depth][tensor_idx] = gate
            # @infiltrate
        end
        
        # Update environment tensors
        if idx == 1
            update_upper_env_tensor_MPSBW(length_half_sweep - idx + 1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
        elseif idx != Int(length(sweep_path_list)/2)
            update_upper_env_tensor_MPSBW(length_half_sweep - idx + 1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            update_lower_env_tensor_MPSBW(length_half_sweep - idx + 2, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
        else
            update_upper_env_tensor_MPSBW(length_half_sweep - idx + 1, psi, upper_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            update_lower_env_tensor_MPSBW(length_half_sweep - idx + 2, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
            update_lower_env_tensor_MPSBW(length_half_sweep - idx + 1, psi, lower_env_tensor_list, N, num_depth, gate_list, init_allzero_tensor_list)
        end
        # println(real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1]))
    end
    cost = real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1])
    # @infiltrate
    return gate_list, upper_env_tensor_list, lower_env_tensor_list, cost
end 


function get_and_save_evol_BW_2d_list(evol_MPO, num_depth, num_sweep, name; save = true, save_dir = save_dir_npy, cutoff = 1e-12, random = true, seed = 1)
    # The number of sites and num_depth must be even
    N=length(evol_MPO)
    # @infiltrate
    if N % 2 == 1 
        error("The number of sites must be even")
    end

    # Get left and right indices
    left_ind_list, right_ind_list = get_left_right_list(evol_MPO)
    # Create the gate list
    bond_list = get_bond_list(num_depth, N, left_ind_list, right_ind_list)
    # @infiltrate
    gate_list_init = get_gate_list_init(num_depth, N, bond_list; random = random, seed = seed)
    # @infiltrate
    gate_list = deepcopy(gate_list_init)

    # Initialize environment tensors
    lower_env_tensor_list = [ITensor(1.0) for _ in 1:N-2]
    upper_env_tensor_list = [ITensor(1.0) for _ in 1:N-2]
    for upper_env_tensor_idx in N-2:-1:1
        upper_env_tensor_list = update_upper_env_tensor(upper_env_tensor_idx, evol_MPO, upper_env_tensor_list, N, num_depth, gate_list)
    end

    for lower_env_tensor_idx in 1:N-2
        lower_env_tensor_list = update_lower_env_tensor(lower_env_tensor_idx, evol_MPO, lower_env_tensor_list, N, num_depth, gate_list)
    end

    # Obtain pre-optimization cost
    cost_list = [real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1])]
    # @infiltrate

    sweep_path_list = get_sweep_path_list(N, num_depth)

    # Gate optimization
    start_time = time()
    println("Sweep Cost Time")
    for sweep in 1:num_sweep
        # @infiltrate
        gate_list, upper_env_tensor_list, lower_env_tensor_list, cost = get_sweeped_gate_list(num_depth, gate_list, N, upper_env_tensor_list, lower_env_tensor_list, sweep_path_list, evol_MPO, gate_list_init; cutoff = cutoff)
        delta = sqrt(2-cost^(1/N))
        push!(cost_list, delta)
        println("$(sweep) $(delta) $(time()-start_time)")
        flush(stdout)
        # @infiltrate
        # @infiltrate
    end
    # @infiltrate

    # Save gates
    if save
        # @infiltrate
        U = []
        for depth in 1:num_depth
            if depth%2 == 1 # When depth is odd, the number of tensors is N/2
                for tensor_idx in 1:Int(N/2)
                    T = gate_list[depth][tensor_idx]
                    U = get_unitary_from_tensor(T)
                    # @infiltrate
                    #save unitary
                    np.save("$(save_dir)/$(name)_MPO_depth$(depth)_$(tensor_idx).npy", np.array(U))
                end
            else # When depth is even, the number of tensors is N/2-1
                for tensor_idx in 1:Int(N/2)-1
                    T = gate_list[depth][tensor_idx]
                    U = get_unitary_from_tensor(T)
                    #save unitary
                    np.save("$(save_dir)/$(name)_MPO_depth$(depth)_$(tensor_idx).npy", np.array(U))
                end
            end            
        end
    end

    return gate_list, cost_list
end

function get_init_allzero_list_MPSBW(N)
    # Initial state of all zeros
    init_allzero_tensor_list = []
    init_allzero_list = []
    for site_idx in 1:N
        i = Index(2, "i")
        push!(init_allzero_tensor_list, itensor([1.0, 0.0], i))
        push!(init_allzero_list, i)
    end
    return init_allzero_tensor_list, init_allzero_list
end


# After modifying envMPS
function get_and_save_prep_MPSBW_2d_list_envMPS(psi, num_depth, num_sweep, name; save = true, save_dir = save_dir_npy, cutoff = 1e-12, random = true, seed = 1)
    N=length(psi)
    # @infiltrate
    if N % 2 == 0 
        error("Only odd numbers of sites are allowed")
    end

    # Tensor for the initial state
    init_allzero_tensor_list, init_allzero_list = get_init_allzero_list_MPSBW(N)

    # Get left indices
    left_ind_list = get_left_list_MPSBW(psi)
    # Create the list of gates
    bond_list = get_bond_list_MPSBW(num_depth, N, left_ind_list, init_allzero_list)
    # @infiltrate
    # Initial tensor, no need to change?
    gate_list_init = get_gate_list_init_MPSBW(num_depth, N, bond_list; random = random, seed = seed) 
    # @infiltrate
    gate_list = deepcopy(gate_list_init)

    # Initialize environment tensors
    println("init env")
    flush(stdout)
    lower_env_tensor_list = []
    upper_env_tensor_list = []
    for env_tensor_idx in 1:N-2
        if num_depth %2 == 0
            push!(lower_env_tensor_list, MPS(num_depth+1))
            push!(upper_env_tensor_list, MPS(num_depth+1))
        elseif env_tensor_idx %2 == 0
            push!(lower_env_tensor_list, MPS(num_depth))
            push!(upper_env_tensor_list, MPS(num_depth))
        else
            push!(lower_env_tensor_list, MPS(num_depth+2))
            push!(upper_env_tensor_list, MPS(num_depth+2))
        end
    end
    # @infiltrate
    for upper_env_tensor_idx in N-2:-1:1
        upper_env_tensor_list = update_upper_env_tensor_MPSBW_envMPS(upper_env_tensor_idx, psi, upper_env_tensor_list, N, num_depth, gate_list,init_allzero_tensor_list;cutoff=cutoff)
    end

    for lower_env_tensor_idx in 1:N-2
        lower_env_tensor_list = update_lower_env_tensor_MPSBW_envMPS(lower_env_tensor_idx, psi, lower_env_tensor_list, N, num_depth, gate_list,init_allzero_tensor_list;cutoff=cutoff)
    end

    # Obtain the pre-optimization cost
    println("init cost")
    flush(stdout)
    cost = ITensor(1.0)
    for idx in 1:length(upper_env_tensor_list[1])
        cost *= upper_env_tensor_list[1][idx] * lower_env_tensor_list[1][idx]
    end
    cost_list = [real(array(cost))[1]]
    
    sweep_path_list = get_sweep_path_list_MPSBW(N, num_depth)
    # @infiltrate

    # Optimize the gates
    start_time = time()
    println("Sweep Fidelity Time")
    flush(stdout)
    for sweep in 1:num_sweep
        # @infiltrate
        gate_list, upper_env_tensor_list, lower_env_tensor_list, cost = get_sweeped_gate_list_MPSBW_envMPS(num_depth, gate_list, N, upper_env_tensor_list, lower_env_tensor_list, sweep_path_list, psi, gate_list_init, init_allzero_tensor_list; cutoff = cutoff)
        # delta = sqrt(2-cost^(1/N))
        # push!(cost_list, delta)
        # println("$(sweep) $(delta)")
        push!(cost_list, cost)
        println("$(sweep) $(cost) $(time()-start_time)")
        flush(stdout)
        # @infiltrate
    end
    # @infiltrate

    # Save the gates
    if save
        # @infiltrate
        U = []
        for depth in 1:num_depth
            for tensor_idx in 1:Int((N-1)/2)
                T = gate_list[depth][tensor_idx]
                U = get_unitary_from_tensor(T)
                # @infiltrate
                if depth == 1
                    U = gram_schmidt(U[:,1])
                elseif depth == 2 && tensor_idx == Int((N-1)/2)
                    # @infiltrate
                    U = gram_schmidt(U[:,1],U[:,2])
                end
                # U = adjoint(U)
                # @infiltrate
                #save unitary
                np.save("$(save_dir)/$(name)_MPSBW_depth$(depth)_$(tensor_idx).npy", np.array(U))
            end
        end
    end

    return gate_list, cost_list
end


# Without envMPS
function get_and_save_prep_MPSBW_2d_list(psi, num_depth, num_sweep, name; save = true, save_dir = save_dir_npy, cutoff = 1e-12, random = true, seed = 1)
    # The number of sites and num_depth must be even
    N=length(psi)
    # @infiltrate
    if N % 2 == 0 
        error("The number of sites is odd")
    end
    if num_depth <= 2
        error("num_depth <= 2")
    end

    # Tensor for the initial state
    init_allzero_tensor_list, init_allzero_list = get_init_allzero_list_MPSBW(N)

    # Get left indices
    left_ind_list = get_left_list_MPSBW(psi)
    # Create the list of gates
    bond_list = get_bond_list_MPSBW(num_depth, N, left_ind_list, init_allzero_list)
    # @infiltrate
    # Initial tensor, no need to change?
    gate_list_init = get_gate_list_init_MPSBW(num_depth, N, bond_list; random = random, seed = seed) 
    # @infiltrate
    gate_list = deepcopy(gate_list_init)

    # Initialize environment tensors
    println("init env for test")
    flush(stdout)
    lower_env_tensor_list = [ITensor(1.0) for _ in 1:N-2]
    upper_env_tensor_list = [ITensor(1.0) for _ in 1:N-2]
    for upper_env_tensor_idx in N-2:-1:1
        upper_env_tensor_list = update_upper_env_tensor_MPSBW(upper_env_tensor_idx, psi, upper_env_tensor_list, N, num_depth, gate_list,init_allzero_tensor_list)
    end

    for lower_env_tensor_idx in 1:N-2
        lower_env_tensor_list = update_lower_env_tensor_MPSBW(lower_env_tensor_idx, psi, lower_env_tensor_list, N, num_depth, gate_list,init_allzero_tensor_list)
    end

    # Obtain the pre-optimization cost
    println("init cost for test")
    flush(stdout)
    cost_list = [real(array(upper_env_tensor_list[1] * lower_env_tensor_list[1])[1])]
    # @infiltrate
    
    sweep_path_list = get_sweep_path_list_MPSBW(N, num_depth)
    # @infiltrate

    # Optimize the gates
    start_time = time()
    println("Sweep Fidelity Time")
    flush(stdout)
    for sweep in 1:num_sweep
        # @infiltrate
        gate_list, upper_env_tensor_list, lower_env_tensor_list, cost = get_sweeped_gate_list_MPSBW(num_depth, gate_list, N, upper_env_tensor_list, lower_env_tensor_list, sweep_path_list, psi, gate_list_init, init_allzero_tensor_list; cutoff = cutoff)
        # delta = sqrt(2-cost^(1/N))
        # push!(cost_list, delta)
        # println("$(sweep) $(delta)")
        push!(cost_list, cost)
        println("$(sweep) $(cost) $(time()-start_time)")
        flush(stdout)
        # @infiltrate
    end
    # @infiltrate

    # Save the gates
    if save
        # @infiltrate
        U = []
        for depth in 1:num_depth
            for tensor_idx in 1:Int((N-1)/2)
                T = gate_list[depth][tensor_idx]
                U = get_unitary_from_tensor(T)
                if depth == 1
                    # @infiltrate
                    U = gram_schmidt(U[:,1])
                    # @infiltrate
                elseif depth == 2 && tensor_idx == Int((N-1)/2)
                    # @infiltrate
                    U = gram_schmidt(U[:,1],U[:,2])
                    # @infiltrate
                end
                # U = adjoint(U)
                # println(depth, tensor_idx)
                # println(U'*U)
                # @infiltrate
                #save unitary
                np.save("$(save_dir)/$(name)_MPSBW_depth$(depth)_$(tensor_idx).npy", np.array(U))
            end
        end
    end

    return gate_list, cost_list
end




function gram_schmidt(v1 = nothing, v2 = nothing, v3 = nothing)
    # Generate unitary matrix from vectors using the Gram-Schmidt method
    if isnothing(v1)
        throw(ErrorException("v1 is nothing"))
    elseif isnothing(v2)
        v1 /= norm(v1)
        v2 = randn(4)
        v3 = randn(4)
        v4 = randn(4)
        V = hcat(v1, v2, v3, v4)  # Arrange as column vectors
        Q, R = qr(V)
        U = Matrix(Q)
        # Correct the sign of diagonal elements
        for i in 1:4
            U[:, i] *= sign(R[i, i])
        end
        # @infiltrate
    elseif isnothing(v3)
        v1 /= norm(v1)
        v2 /= norm(v2)
        v3 = randn(4)
        v4 = randn(4)
        V = hcat(v1, v2, v3, v4)  # Arrange as column vectors
        Q, R = qr(V)
        U = Matrix(Q)
        # Correct the sign of diagonal elements
        for i in 1:4
            U[:, i] *= sign(R[i, i])
        end
        # @infiltrate
    else
        # @infiltrate
        v1 /= norm(v1)
        v2 /= norm(v2)
        v3 /= norm(v3)
        v4 = randn(4)
        V = hcat(v1, v2, v3, v4)  # Arrange as column vectors
        Q, R = qr(V)
        U = Matrix(Q)
        # Correct the sign of diagonal elements
        for i in 1:4
            U[:, i] *= sign(R[i, i])
        end
    end
    return U
end

function get_U(idx_tensor, psi, N)
    # Generate unitary matrix from a tensor in MPS
    # @infiltrate
    if idx_tensor == 1
        shared_index_lm = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
        unique_index_nm = filter(ind -> ind != shared_index_lm, inds(psi[idx_tensor]))[1]

        v1 = []
        for i in 1:2
            for j in 1:2
                # @infiltrate
                append!(v1, psi[idx_tensor][shared_index_lm=> i, unique_index_nm => j])
                #append!(v1, psi[idx_tensor][inds(psi[idx_tensor])[2] => i, inds(psi[idx_tensor])[1] => j])
            end
        end
        # Execute Gram-Schmidt orthogonalization
        U = gram_schmidt(v1)
        # @infiltrate
    
    elseif idx_tensor != N
        shared_index_lm = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
        shared_index_lm_before = commonind(psi[idx_tensor-1], psi[idx_tensor])
        unique_index_nm = filter(ind -> ind != shared_index_lm && ind != shared_index_lm_before, inds(psi[idx_tensor]))[1]

        v1 = []
        for i in 1:2
            for j in 1:2
                append!(v1, psi[idx_tensor][shared_index_lm => i, unique_index_nm => j, shared_index_lm_before => 1])
                #append!(v1, psi[idx_tensor][inds(psi[idx_tensor])[2] => i, inds(psi[idx_tensor])[1] => j, inds(psi[idx_tensor])[3] => 1])
            end
        end
        v2 = []
        for i in 1:2
            for j in 1:2
                append!(v2, psi[idx_tensor][shared_index_lm => i, unique_index_nm => j, shared_index_lm_before => 2])
                #append!(v2, psi[idx_tensor][inds(psi[idx_tensor])[2] => i, inds(psi[idx_tensor])[1] => j, inds(psi[idx_tensor])[3] => 2])
            end
        end
        # Execute Gram-Schmidt orthogonalization
        U = gram_schmidt(v1, v2)
        # @infiltrate
    else #idx_tensor == N
        shared_index_lm_before = commonind(psi[idx_tensor-1], psi[idx_tensor]) 
        unique_index_nm = filter(ind -> ind != shared_index_lm_before, inds(psi[idx_tensor]))[1]
        
        U = [psi[idx_tensor][unique_index_nm => 1, shared_index_lm_before => 1] psi[idx_tensor][unique_index_nm => 1, shared_index_lm_before => 2]; 
            psi[idx_tensor][unique_index_nm => 2, shared_index_lm_before => 1] psi[idx_tensor][unique_index_nm => 2, shared_index_lm_before => 2]]
        # @infiltrate
    end
    return U
end

function hamcall(name; save_dir = save_dir_pickle)
    # Call hamiltonian from pickle
    open("$(save_dir)/$(name).pickle", "r") do file
        ham = pickle.load(file)
        return ham
    end
end

function get_excited_state(num_ex, psi_list, psi0, H; nsweeps_ex = 10, maxdim_ex = 2, mindim_ex = 2, cutoff_ex = 1e-12)
    energy_ex = nothing
    psi_ex = nothing
    for idx_ex in 1:num_ex
        # Run the DMRG algorithm, returning energy
        # (dominant eigenvalue) and optimized MPS
        println("After sweep 0 energy=$(real(inner(psi0' , H , psi0)))")
        energy_ex, psi_ex = dmrg(H, psi_list, psi0; nsweeps=nsweeps_ex, maxdim=maxdim_ex, mindim=mindim_ex, cutoff=cutoff_ex)
        # @infiltrate
        println("Final energy of $idx_ex-th excited state = $energy_ex")
        append!(psi_list, [psi_ex])
    end
    return psi_ex
end


function get_MPS_approx(psi, N)
    # Approximate MPS with bond dimension 2
    psi_temp = deepcopy(psi)
    for idx_tensor in 1:N
        # @infiltrate
        if idx_tensor == 1
            # Get the index not shared with the next tensor
            shared_index_after = commonind(psi_temp[idx_tensor],psi_temp[idx_tensor+1])
            unique_index = filter(ind -> ind != shared_index_after, inds(psi_temp[idx_tensor]))[1]
            # SVD
            U, S, V = svd(psi_temp[idx_tensor], shared_index_after; maxdim=2)

            # Update psi_temp
            psi_temp[idx_tensor] = V
            psi_temp[idx_tensor+1] = psi_temp[idx_tensor+1] * U * S
            # @infiltrate
        elseif idx_tensor != N
            # Get the index not shared with the previous and next tensors
            shared_index_before = commonind(psi_temp[idx_tensor],psi_temp[idx_tensor-1])
            shared_index_after = commonind(psi_temp[idx_tensor],psi_temp[idx_tensor+1])
            unique_index = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(psi_temp[idx_tensor]))[1]
            # SVD
            U, S, V = svd(psi_temp[idx_tensor], shared_index_after; maxdim=2, mindim=2)

            # Update psi_temp
            psi_temp[idx_tensor] = V
            psi_temp[idx_tensor+1] = psi_temp[idx_tensor+1] * U * S
            # @infiltrate
        else
            nothing
        end
    end
    normalize!(psi_temp)
    orthogonalize!(psi_temp, 1)
    return psi_temp
end

function get_unitary_from_tensor(MPO)
    # Obtain unitary matrix from tensor
    U = zeros(Complex{Float64}, 4, 4)
    for i in 1:2
        for j in 1:2
            for k in 1:2
                for l in 1:2
                    binary_string_in = string(i-1, j-1)  # Create a string like "10"
                    decimal_in = parse(Int, binary_string_in, base = 2) + 1  # Interpret as binary and convert to decimal
                    binary_string_out = string(k-1, l-1)  # Create a string like "10"
                    decimal_out = parse(Int, binary_string_out, base = 2) + 1  # Interpret as binary and convert to decimal

                    U[decimal_out, decimal_in] = MPO[i,j,k,l] # When written as a matrix element, "in" comes at the back (column)
                    # @infiltrate
                end
            end
        end
    end
    # @infiltrate
    return U
end

function get_MPO(idx_tensor, psi_approx, psi_approx_U, N)
    # Create MPO from unitary matrix
    MPO_temp = nothing
    if idx_tensor == 1
        new_index1 = Index(2,"new_index1")
        new_index2 = Index(2,"new_index2")
        shared_index_after = commonind(psi_approx[idx_tensor], psi_approx[idx_tensor+1])
        unique_index = filter(ind -> ind != shared_index_after, inds(psi_approx[idx_tensor]))[1]
        MPO_temp = ITensor(new_index1, new_index2, shared_index_after, unique_index)
        for i in 1:2
            for j in 1:2
                for k in 1:2
                    for l in 1:2
                        binary_string_in = string(i-1, j-1)  # Create a string like "10"
                        decimal_in = parse(Int, binary_string_in, base = 2) + 1  # Interpret as binary and convert to decimal
                        binary_string_out = string(k-1, l-1)  # Create a string like "10"
                        decimal_out = parse(Int, binary_string_out, base = 2) + 1  # Interpret as binary and convert to decimal

                        MPO_temp[i,j,k,l] = psi_approx_U[decimal_out, decimal_in] # When written as a matrix element, "in" comes at the back
                    end
                end
            end
        end
        return MPO_temp, new_index2
    elseif idx_tensor != N
        new_index1 = Index(2,"new_index1")
        shared_index_before = commonind(psi_approx[idx_tensor],psi_approx[idx_tensor-1])
        shared_index_after = commonind(psi_approx[idx_tensor],psi_approx[idx_tensor+1])
        unique_index = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(psi_approx[idx_tensor]))[1]
        MPO_temp = ITensor(new_index1, shared_index_before, shared_index_after, unique_index)
        for i in 1:2
            for j in 1:2
                for k in 1:2
                    for l in 1:2
                        binary_string_in = string(i-1, j-1)  # Create a string like "10"
                        decimal_in = parse(Int, binary_string_in, base = 2) + 1  # Interpret as binary and convert to decimal
                        binary_string_out = string(k-1, l-1)  # Create a string like "10"
                        decimal_out = parse(Int, binary_string_out, base = 2) + 1  # Interpret as binary and convert to decimal

                        MPO_temp[i,j,k,l] = psi_approx_U[decimal_out, decimal_in]
                    end
                end
            end
        end
        return MPO_temp
    else
        shared_index_before = commonind(psi_approx[idx_tensor],psi_approx[idx_tensor-1])
        unique_index = filter(ind -> ind != shared_index_before, inds(psi_approx[idx_tensor]))[1]
        MPO_temp = ITensor(shared_index_before, unique_index)
        for i in 1:2
            for j in 1:2
                    MPO_temp[i,j] = psi_approx_U[i, j]
            end
        end
        return MPO_temp
    end
end

function get_new_psi(psi, contracted_MPO, first_index, N; cutoff = 1e-12)
    # Contract MPS with MPO and create new MPS with weakened entanglement
    new_psi_temp = MPS(N)
    new_psi = nothing
    MPS_for_svd = nothing 
    U_MPS = nothing
    for idx_tensor in 1:N
        # @infiltrate
        if idx_tensor == 1
            shared_index_MPS = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            shared_index_MPO = commonind(contracted_MPO[idx_tensor], contracted_MPO[idx_tensor+1]) 
            MPS_for_svd = psi[idx_tensor] * contracted_MPO[idx_tensor]   
            # @infiltrate
            U_MPS, S, V = svd(MPS_for_svd, shared_index_MPS, shared_index_MPO; cutoff = cutoff)
            U_MPS = U_MPS * S
            # @infiltrate

            p1p2 = V 
            shared_index_MPS_2 = commonind(p1p2, U_MPS) 
            second_index = filter(ind -> ind != shared_index_MPS_2 && ind != first_index, inds(p1p2))[1]
            psi2_temp, S_2, psi1_temp = svd(p1p2, shared_index_MPS_2, second_index; cutoff = cutoff)
            psi2_temp = psi2_temp * S_2
            # @infiltrate

            new_psi_temp[idx_tensor] = psi1_temp
            new_psi_temp[idx_tensor+1] = psi2_temp
            # @infiltrate
        elseif idx_tensor < N
            shared_index_MPS = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            shared_index_MPO = commonind(contracted_MPO[idx_tensor], contracted_MPO[idx_tensor+1]) 
            MPS_for_svd = U_MPS * psi[idx_tensor] * contracted_MPO[idx_tensor]
            U_MPS, S, psi_temp = svd(MPS_for_svd, shared_index_MPS, shared_index_MPO; cutoff = cutoff)
            U_MPS = U_MPS * S
            new_psi_temp[idx_tensor+1] = psi_temp
            # @infiltrate
        elseif idx_tensor == N
            # @infiltrate
            new_psi_temp[idx_tensor] = new_psi_temp[idx_tensor] * U_MPS * psi[idx_tensor] * contracted_MPO[idx_tensor]
            # @infiltrate
        end
    end
    new_psi = new_psi_temp 
    # Adjust psi
    new_psi = sweep_MPS(new_psi, N; cutoff=cutoff)
    normalize!(new_psi)
    orthogonalize!(new_psi, 1)
    return new_psi
end

function sweep_MPS(psi, N; cutoff = 1e-12, mindim = 2, maxdim = 1e30)
    # Sweep MPS
    # @infiltrate
    MPS_for_svd = nothing 
    for idx_tensor in 1:N
        if idx_tensor == 1
            shared_index_after = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            unique_index = filter(ind -> ind != shared_index_after, inds(psi[idx_tensor]))[1]
            MPS_for_svd = psi[idx_tensor] * psi[idx_tensor+1]   
            U, S, V = svd(MPS_for_svd, unique_index; cutoff = cutoff, mindim = mindim, maxdim = maxdim)
            psi[idx_tensor] = U
            psi[idx_tensor+1] = S * V
            # @infiltrate

        elseif idx_tensor != N
            shared_index_before = commonind(psi[idx_tensor], psi[idx_tensor-1]) 
            shared_index_after = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            unique_index = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(psi[idx_tensor]))[1]
            MPS_for_svd = psi[idx_tensor] * psi[idx_tensor+1]
            U, S, V = svd(MPS_for_svd, shared_index_before, unique_index; cutoff = cutoff, mindim = mindim, maxdim = maxdim)
            psi[idx_tensor] = U
            psi[idx_tensor+1] = S * V
            # @infiltrate
        elseif idx_tensor == N
            # @infiltrate
            nothing
        end
    end

    for idx_tensor in N:-1:1
        # @infiltrate
        if idx_tensor == N
            shared_index_before = commonind(psi[idx_tensor], psi[idx_tensor-1]) 
            unique_index = filter(ind -> ind != shared_index_before, inds(psi[idx_tensor]))[1]
            MPS_for_svd = psi[idx_tensor] * psi[idx_tensor-1]   
            U, S, V = svd(MPS_for_svd, unique_index; cutoff = cutoff, mindim = mindim)
            psi[idx_tensor] = U
            psi[idx_tensor-1] = S * V
            # @infiltrate
        elseif idx_tensor != 1
            shared_index_before = commonind(psi[idx_tensor], psi[idx_tensor-1]) 
            shared_index_after = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            unique_index = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(psi[idx_tensor]))[1]
            MPS_for_svd = psi[idx_tensor] * psi[idx_tensor-1]
            U, S, V = svd(MPS_for_svd, shared_index_after, unique_index; cutoff = cutoff, mindim = mindim)
            psi[idx_tensor] = U
            psi[idx_tensor-1] = S * V
            # @infiltrate
        elseif idx_tensor == 1
            nothing
        end
    end
    return psi

end

function get_and_save_contracted_MPO(psi_approx, depth, N, name; save = true, save_dir = save_dir_npy)
    # Convert MPS to unitary matrix and save it
    contracted_MPO = MPO(N)
    psi_approx_U_list = []
    psi_approx_U = nothing
    # MPO_list = []
    single_MPO = nothing
    first_index = nothing
    for idx_tensor in 1:N
        # get unitary from psi
        # println("idx tensor $(idx_tensor)")
        # @infiltrate
        psi_approx_U = get_U(idx_tensor, psi_approx, N)
        push!(psi_approx_U_list, psi_approx_U)
        # @infiltrate

        #save unitary
        if save
            np.save("$(save_dir)/$(name)_MPS_array_psi_depth$(depth)_$(idx_tensor).npy", np.array(psi_approx_U))
        end

        #get MPO from unitary
        # @infiltrate
        if idx_tensor == 1
            single_MPO, first_index = get_MPO(idx_tensor, psi_approx, psi_approx_U, N)
        else
            single_MPO = get_MPO(idx_tensor, psi_approx, psi_approx_U, N)
        end
        # push!(MPO_list, single_MPO)
        
        contracted_MPO[idx_tensor] = single_MPO
        # @infiltrate
    end
    return contracted_MPO, first_index
end

function get_amplitude(psi,N)
    # Get the amplitude for all 1's
    V = ITensor(1.)
    v = nothing
    for idx_tensor in 1:N
        if idx_tensor == 1
            shared_index_after = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            unique_index = filter(ind -> ind != shared_index_after, inds(psi[idx_tensor]))[1]
            V *= (psi[idx_tensor]*state(unique_index,1))
        elseif idx_tensor != N
            shared_index_before = commonind(psi[idx_tensor], psi[idx_tensor-1]) 
            shared_index_after = commonind(psi[idx_tensor], psi[idx_tensor+1]) 
            unique_index = filter(ind -> ind != shared_index_before && ind != shared_index_after, inds(psi[idx_tensor]))[1]
            V *= (psi[idx_tensor]*state(unique_index,1))
        else
            shared_index_before = commonind(psi[idx_tensor], psi[idx_tensor-1]) 
            unique_index = filter(ind -> ind != shared_index_before, inds(psi[idx_tensor]))[1]
            V *= (psi[idx_tensor]*state(unique_index,1))
        end
    end
    v = scalar(V)
    return v
end

function get_and_save_MPO_unitary(psi_in, depth_max, N, name; cutoff = 1e-12, maxdim = 1e30, save = true)
    # Repeat the process of obtaining the unitary from the MPS
    psi = deepcopy(psi_in)
    normalize!(psi)
    orthogonalize!(psi, 1)
    psi_approx = nothing
    contracted_MPO = nothing
    psi_list = []
    # infidelity_list =[]
    amplitude_list = []
    amplitude = nothing
    for depth in 1:depth_max
        # @infiltrate
        amplitude = get_amplitude(psi,N)
        push!(amplitude_list, abs(amplitude))
        println("$(depth) $(abs(amplitude))")

        psi = sweep_MPS(psi, N; cutoff = cutoff, maxdim = maxdim)

        psi_approx = get_MPS_approx(psi, N)

        contracted_MPO, first_index = get_and_save_contracted_MPO(psi_approx, depth, N, name; save)

        # contracted_MPO = dag(contracted_MPO) 
        
        # Proper implementation of contraction
        new_psi = get_new_psi(psi, contracted_MPO, first_index, N)

        psi = new_psi
        # print(psi)

        push!(psi_list, psi)
    end
    return psi_list, amplitude_list
end
