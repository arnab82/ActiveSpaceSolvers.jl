using ActiveSpaceSolvers
using QCBase
import LinearMaps
using OrderedCollections
using BlockDavidson
using Printf

#import BlockDavidson: solve
"""
Type containing all the metadata needed to define a FCI problem 

    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
"""
struct FCIAnsatz <: Ansatz
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int
    dimb::Int
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
end

"""
    FCIAnsatz(no, na, nb)

Constructor

# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
"""
function FCIAnsatz(no, na, nb)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    dima = calc_nchk(no, na)
    dimb = calc_nchk(no, nb)
    return FCIAnsatz(no, na, nb, dima, dimb, dima * dimb, false, false, 1, "direct", 1)
end

function Base.display(p::FCIAnsatz)
    @printf(" FCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n", p.no, p.na, p.nb, p.dim)
end

function Base.print(p::FCIAnsatz)
    @printf(" FCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n", p.no, p.na, p.nb, p.dim)
end

"""
    ActiveSpaceSolvers.compute_s2(sol::Solution)

Compute the <S^2> expectation values for each state in `sol`
"""
function ActiveSpaceSolvers.compute_s2(sol::Solution{FCIAnsatz,T}) where {T}
    return compute_S2_expval(sol.vectors, sol.ansatz)
end

"""
"""
function ActiveSpaceSolvers.apply_sminus(v::Matrix, ansatz::FCIAnsatz)
    #={{{=#
    no = ansatz.no
    na = ansatz.na
    nb = ansatz.nb

    if nb + 1 > no
        error(" Can't decrease Ms further")
    end

    #   Create ci_strings
    ket_a = DeterminantString(no, na)
    ket_b = DeterminantString(no, nb)
    bra_a = DeterminantString(no, na - 1)
    bra_b = DeterminantString(no, nb + 1)
    bra_a2 = DeterminantString(no, na - 1)
    bra_b2 = DeterminantString(no, nb + 1)

    sgnK = 1
    if ket_a.ne % 2 != 0
        sgnK = -sgnK
    end

    #sgnK *= 1/sqrt(2)

    w = zeros(bra_a.max * bra_b.max, size(v, 2))

    reset!(ket_b)
    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb - 1) * ket_a.max


            #for ai in ket_a.config
            #    if ai ∉ ket_b.config
            for ai in 1:no

                bra_a = deepcopy(ket_a)
                bra_b = deepcopy(ket_b)

                apply_annihilation!(bra_a, ai)
                bra_a.sign != 0 || continue

                apply_creation!(bra_b, ai)
                bra_b.sign != 0 || continue

                La = calc_linear_index(bra_a)
                Lb = calc_linear_index(bra_b)

                L = La + (Lb - 1) * bra_a2.max
                w[L, :] .+= sgnK * bra_a.sign * bra_b.sign * v[K, :]
            end
            incr!(ket_a)
        end
        incr!(ket_b)
    end

    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w, 1), 0)
    for i in 1:size(w, 2)
        ni = norm(w[:, i])
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w[:, i] ./ ni)
        end
    end

    return wout, FCIAnsatz(no, na - 1, nb + 1)
end
#=}}}=#

"""
"""
function ActiveSpaceSolvers.apply_splus(v::Matrix, ansatz::FCIAnsatz)
    #={{{=#
    no = ansatz.no
    na = ansatz.na
    nb = ansatz.nb

    if na + 1 > no
        error(" Can't decrease Ms further")
    end

    #   Create ci_strings
    ket_a = DeterminantString(no, na)
    ket_b = DeterminantString(no, nb)
    bra_a = DeterminantString(no, na + 1)
    bra_b = DeterminantString(no, nb - 1)
    bra_a2 = DeterminantString(no, na + 1)
    bra_b2 = DeterminantString(no, nb - 1)

    sgnK = 1
    if ket_a.ne % 2 != 0
        sgnK = -sgnK
    end

    w = zeros(bra_a.max * bra_b.max, size(v, 2))

    reset!(ket_b)
    for Kb in 1:ket_b.max

        reset!(ket_a)
        for Ka in 1:ket_a.max
            K = Ka + (Kb - 1) * ket_a.max


            #for ai in ket_a.config
            #    if ai ∉ ket_b.config
            for ai in 1:no

                bra_a = deepcopy(ket_a)
                bra_b = deepcopy(ket_b)

                apply_creation!(bra_a, ai)
                bra_a.sign != 0 || continue

                apply_annihilation!(bra_b, ai)
                bra_b.sign != 0 || continue

                La = calc_linear_index(bra_a)
                Lb = calc_linear_index(bra_b)

                L = La + (Lb - 1) * bra_a2.max
                w[L, :] .+= sgnK * bra_a.sign * bra_b.sign * v[K, :]
            end
            incr!(ket_a)
        end
        incr!(ket_b)
    end

    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w, 1), 0)
    for i in 1:size(w, 2)
        ni = norm(w[:, i])
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w[:, i] ./ ni)
        end
    end

    return wout, FCIAnsatz(no, na + 1, nb - 1)
end
#=}}}=#



"""
    LinearMap(ints, prb::FCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `FCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::FCIAnsatz)
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    #@btime lookup_a = $fill_ca_lookup2($ket_a)
    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v, ket_a.max * ket_b.max, nr)
        else
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ints, prb, lookup_a, lookup_b)
        sig += compute_ss_terms2(v, ints, prb, lookup_a, lookup_b)

        v = reshape(v, ket_a.max * ket_b.max, nr)
        sig = reshape(sig, ket_a.max * ket_b.max, nr)
        sig .+= ints.h0 * v
        return sig
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#


"""
    LinOpMat(ints, prb::FCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `FCIAnsatz` object
"""
function BlockDavidson.LinOpMat(ints::InCoreInts{T}, prb::FCIAnsatz) where {T}
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    #@btime lookup_a = $fill_ca_lookup2($ket_a)
    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v, ket_a.max * ket_b.max, nr)
        else
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ints, prb, lookup_a, lookup_b)
        sig += compute_ss_terms2(v, ints, prb, lookup_a, lookup_b)

        v = reshape(v, ket_a.max * ket_b.max, nr)
        sig = reshape(sig, ket_a.max * ket_b.max, nr)
        sig .+= ints.h0 * v
        return sig
    end
    return LinOpMat{T}(mymatvec, prb.dim, true)
end
#=}}}=#


"""
    build_H_matrix(ints, P::FCIAnsatz)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts{T}, P::FCIAnsatz) where {T}
    #={{{=#
    Hmat = zeros(T, P.dim, P.dim)

    Hdiag_a = precompute_spin_diag_terms(ints, P, P.na)
    Hdiag_b = precompute_spin_diag_terms(ints, P, P.nb)
    # 
    #   Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)
    #   
    #   Add spin diagonal components
    Hmat += kron(Matrix(1.0I, P.dimb, P.dimb), Hdiag_a)
    Hmat += kron(Hdiag_b, Matrix(1.0I, P.dima, P.dima))
    #
    #   Add opposite spin term (todo: make this reasonably efficient)
    Hmat += compute_ab_terms_full(ints, P, T=T)

    Hmat = 0.5 * (Hmat + Hmat')
    Hmat += 1.0I * ints.h0
    return Hmat
end
#=}}}=#


"""
    build_S2_matrix(P::FCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.apply_S2_matrix(P::FCIAnsatz, v::AbstractArray{T}) where {T}
    #={{{=#
    return apply_S2_matrix(P, v)
end
#=}}}=#


"""
    build_S2_matrix(P::FCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_S2_matrix(P::FCIAnsatz) where {T}
    #={{{=#
    return build_S2_matrix(P)
end
#=}}}=#


"""
    compute_operator_c_a(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_a(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_creation(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "alpha")


    #=}}}=#
end



"""
    compute_operator_c_b(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_b(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_creation(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "beta")


    #=}}}=#
end



"""
    compute_operator_ca_aa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_aa(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_Aa(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "alpha")


    #=}}}=#
end

"""
    compute_operator_ca_bb(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_bb(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_Aa(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "beta")


    #=}}}=#
end


"""
    compute_operator_ca_ab(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_ab(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_Ab(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors)


    #=}}}=#
end


"""
    compute_operator_cc_aa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_bb(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_AA(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "beta")


    #=}}}=#
end


"""
    compute_operator_cc_aa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_aa(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_AA(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "alpha")


    #=}}}=#
end


"""
    compute_operator_cc_ab(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_ab(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_AB(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors)


    #=}}}=#
end


"""
    compute_operator_cca_aaa(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aaa(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_AAa(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "alpha")


    #=}}}=#
end


"""
    compute_operator_cca_bbb(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_bbb(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_AAa(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors,
        "beta")


    #=}}}=#
end


"""
    compute_operator_cca_aba(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aba(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_ABa(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors)


    #=}}}=#
end


"""
    compute_operator_cca_abb(bra::Solution{FCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_abb(bra::Solution{FCIAnsatz,T},
    ket::Solution{FCIAnsatz,T}) where {T}
    #={{{=#
    n_orb(bra) == n_orb(ket) || throw(DimensionMismatch)
    return compute_ABb(n_orb(bra),
        n_elec_a(bra), n_elec_b(bra),
        n_elec_a(ket), n_elec_b(ket),
        bra.vectors, ket.vectors)


    #=}}}=#
end


"""
    compute_1rdm(sol::Solution{FCIAnsatz,T}; root=1) where {T}

"""
function ActiveSpaceSolvers.compute_1rdm(sol::Solution{FCIAnsatz,T}; root=1) where {T}
    #={{{=#

    rdma = compute_Aa(n_orb(sol),
        n_elec_a(sol), n_elec_b(sol),
        n_elec_a(sol), n_elec_b(sol),
        reshape(sol.vectors[:, root], dim(sol), 1),
        reshape(sol.vectors[:, root], dim(sol), 1),
        "alpha")

    rdmb = compute_Aa(n_orb(sol),
        n_elec_a(sol), n_elec_b(sol),
        n_elec_a(sol), n_elec_b(sol),
        reshape(sol.vectors[:, root], dim(sol), 1),
        reshape(sol.vectors[:, root], dim(sol), 1),
        "beta")


    rdma = reshape(rdma, n_orb(sol), n_orb(sol))
    rdmb = reshape(rdmb, n_orb(sol), n_orb(sol))
    return rdma, rdmb
end
#=}}}=#


"""
    compute_2rdm(sol::Solution{A,T}; root=1) where {A,T}

"""
function ActiveSpaceSolvers.compute_1rdm_2rdm(sol::Solution{FCIAnsatz,T}; root=1) where {T}
    #={{{=#

    return compute_rdm1_rdm2(sol.ansatz, sol.vectors[:, root], sol.vectors[:, root])
end
#=}}}=#




"""
    svd_state(sol::Solution{FCIAnsatz,T},norbs1,norbs2,svd_thresh; root=1) where T
Do an SVD of the FCI vector partitioned into clusters with (norbs1 | norbs2)
where the orbitals are assumed to be ordered for cluster 1| cluster 2 haveing norbs1 and 
norbs2, respectively.

- `sol`: Solution just defines the current CI states 
- `norbs1`:number of orbitals in left cluster
- `norbs2`:number of orbitals in right cluster
- `svd_thresh`: the threshold below which the states will be discarded
- `root`: which root to SVD
"""
function ActiveSpaceSolvers.svd_state(sol::Solution{FCIAnsatz,T}, norbs1, norbs2, svd_thresh; root=1) where {T}
    #={{{=#

    @assert(norbs1 + norbs2 == n_orb(sol))

    schmidt_basis = OrderedDict()
    #vector = OrderedDict{Tuple{UInt8,UInt8},Float64}()
    vector = OrderedDict{Tuple{Int,Int},Any}()

    #schmidt_basis = Dict{Tuple,Matrix{Float64}}

    println("----------------------------------------")
    println("          SVD of state")
    println("----------------------------------------")

    # Create ci_strings
    ket_a = DeterminantString(n_orb(sol), n_elec_a(sol))
    ket_b = DeterminantString(n_orb(sol), n_elec_b(sol))

    v = sol.vectors[:, root]
    v = reshape(v, (ket_a.max, ket_b.max))
    @assert(size(v, 1) == ket_a.max)
    @assert(size(v, 2) == ket_b.max)

    fock_labels_a = Array{Int,1}(undef, ket_a.max)
    fock_labels_b = Array{Int,1}(undef, ket_b.max)


    # Get the fock space using the bisect method in python
    #bisect = pyimport("bisect")
    for I in 1:ket_a.max
        label = 0
        for i in 1:length(ket_a.config)
            if ket_a.config[i] <= norbs1
                label += 1
            end
        end
        fock_labels_a[I] = label
        #println(ket_a.config, " ", norbs1, " ", label)
        incr!(ket_a)
    end
    for I in 1:ket_b.max
        label = 0
        for i in 1:length(ket_b.config)
            if ket_b.config[i] <= norbs1
                label += 1
            end
        end
        fock_labels_b[I] = label
        #println(ket_b.config, " ", norbs1, " ", label)
        incr!(ket_b)
    end
    for J in 1:ket_b.max
        for I in 1:ket_a.max
            fock = (fock_labels_a[I], fock_labels_b[J])

            #if fock in vector
            #    append!(vector[fock],v[I,J])
            #else
            #    vector[fock] = [v[I,J]]
            #end
            try
                append!(vector[tuple(fock_labels_a[I], fock_labels_b[J])], v[I, J])
            catch
                vector[tuple(fock_labels_a[I], fock_labels_b[J])] = [v[I, J]]
            end
        end
    end

    for (fock, fvec) in vector

        println()
        @printf("Prepare Fock Space:  %iα, %iβ\n", fock[1], fock[2])

        ket_a1 = DeterminantString(norbs1, fock[1])
        ket_b1 = DeterminantString(norbs1, fock[2])

        ket_a2 = DeterminantString(norbs2, n_elec_a(sol) - fock[1])
        ket_b2 = DeterminantString(norbs2, n_elec_b(sol) - fock[2])


        temp_fvec = reshape(fvec, ket_b1.max * ket_b2.max, ket_a1.max * ket_a2.max)'
        #temp_fvec = reshape(fvec,ket_b1.max*ket_b2.max,ket_a1.max*ket_a2.max)'
        #st = "temp_fvec"*string(fock)
        #npzwrite(st, temp_fvec)


        #when swapping alpha2 and beta1 do we flip sign?
        sign = 1
        if (n_elec_a(sol) - fock[1]) % 2 == 1 && fock[2] % 2 == 1
            sign = -1
        end
        #println("sign",sign)
        @printf("   Dimensions: %5i x %-5i \n", ket_a1.max * ket_b1.max, ket_a2.max * ket_b2.max)

        norm_curr = fvec' * fvec
        @printf("   Norm: %12.8f\n", sqrt(norm_curr))
        #println(size(fvec))
        #display(fvec)

        fvec = sign * fvec

        #opposite to python with transpose on fvec
        #fvec2 = reshape(fvec',ket_b2.max,ket_b1.max,ket_a2.max,ket_a1.max)
        fvec2 = reshape(fvec, ket_a1.max, ket_a2.max, ket_b1.max, ket_b2.max)
        fvec3 = permutedims(fvec2, [1, 3, 2, 4])
        fvec4 = reshape(fvec3, ket_a1.max * ket_b1.max, ket_a2.max * ket_b2.max)

        # fvec4 is transpose of what we have in python code
        fvec5 = fvec4'

        F = svd(fvec5, full=true)


        nkeep = 0
        @printf("   %5s %12s\n", "State", "Weight")
        for (ni_idx, ni) in enumerate(F.S)
            if ni > svd_thresh
                nkeep += 1
                @printf("   %5i %12.8f\n", ni_idx, ni)
            else
                @printf("   %5i %12.8f (discarded)\n", ni_idx, ni)
            end
        end


        if nkeep > 0
            schmidt_basis[fock] = Matrix(F.U[:, 1:nkeep])
            #st = "fin_vec"*string(fock)
            #npzwrite(st, F.U[:,1:nkeep])
        end

        #norm += norm_curr
    end

    return schmidt_basis
end
#=}}}=#
"""
    svd_state_project_S2(v, P::FCIAnsatz, norbs1, norbs2; svd_thresh=1e-6)
Do an SVD of the FCI vector partitioned into clusters with (norbs1 | norbs2)
where the orbitals are assumed to be ordered for cluster 1| cluster 2 haveing norbs1 and
norbs2, respectively.
where the states are projected into S^2 eigenstates before the SVD
- `v`: the FCI vector to be decomposed
- `P`: the FCIAnsatz defining the CI space
- `norbs1`:number of orbitals in left cluster
- `norbs2`:number of orbitals in right cluster
- `svd_thresh`: the threshold below which the states will be discarded
"""

function ActiveSpaceSolvers.svd_state_project_S2(sol::Solution{FCIAnsatz,T}, norbs1, norbs2, svd_thresh=1e-8; root=1,verbose=4) where {T}

    @assert(norbs1 + norbs2 == n_orb(sol))
    schmidt_basis = OrderedDict()              # Stores Schmidt blocks labelled by S^2, Nα, Nβ (Fock sector)
    vector = OrderedDict{Tuple{Int,Int},Any}() # (n_α_left, n_β_left) => coefficients

    println("----------------------------------------")
    println("          SVD of state")
    println("----------------------------------------")

    # Create ci_strings
    ket_a = DeterminantString(n_orb(sol), n_elec_a(sol))
    ket_b = DeterminantString(n_orb(sol), n_elec_b(sol))

    # v = reshape(v, (ket_a.max, ket_b.max))
    v = sol.vectors[:, root]
    v = reshape(v, (ket_a.max, ket_b.max))
    @assert(size(v, 1) == ket_a.max)
    @assert(size(v, 2) == ket_b.max)

    fock_labels_a = Array{Int}(undef, ket_a.max)
    fock_labels_b = Array{Int}(undef, ket_b.max)

    # Count number of occupied orbitals in left cluster directly
    for I in 1:ket_a.max
        label = 0
        for idx in ket_a.config
            if idx <= norbs1
                label += 1
            end
        end
        fock_labels_a[I] = label
        incr!(ket_a)
    end
    for I in 1:ket_b.max
        label = 0
        for idx in ket_b.config
            if idx <= norbs1
                label += 1
            end
        end
        fock_labels_b[I] = label
        incr!(ket_b)
    end

    for J in 1:ket_b.max
        for I in 1:ket_a.max
            fock = (fock_labels_a[I], fock_labels_b[J])

            #if fock in vector
            #    append!(vector[fock],v[I,J])
            #else
            #    vector[fock] = [v[I,J]]
            #end
            try
                append!(vector[tuple(fock_labels_a[I], fock_labels_b[J])], v[I, J])
            catch
                vector[tuple(fock_labels_a[I], fock_labels_b[J])] = [v[I, J]]
            end
        end
    end

    # Loop over Fock sectors
    for (fock, fvec) in vector
        println()
        @printf("system+bath electrons:  %iα, %iβ\n", n_elec_a(sol), n_elec_b(sol))
        @printf("Prepare Fock Space:  %iα, %iβ\n", fock[1], fock[2])
        println("norbs1: ", norbs1, " norbs2: ", norbs2)
        ket_a1 = DeterminantString(norbs1, fock[1])
        ket_b1 = DeterminantString(norbs1, fock[2])

        
        ket_a2 = DeterminantString(norbs2, n_elec_a(sol) - fock[1])
        ket_b2 = DeterminantString(norbs2, n_elec_b(sol) - fock[2])
        println("ket_a1.max: ", ket_a1.max, " ket_b1.max: ", ket_b1.max)
        println("ket_a2.max: ", ket_a2.max, " ket_b2.max: ", ket_b2.max)
        temp_fvec = reshape(fvec, ket_b1.max * ket_b2.max, ket_a1.max * ket_a2.max)'

        sign = 1
        if (n_elec_a(sol) - fock[1]) % 2 == 1 && fock[2] % 2 == 1
            sign = -1
        end
        norm_curr = fvec' * fvec
        if verbose > 2
            @printf("   Dimensions: %5i x %-5i \n", ket_a1.max * ket_b1.max, ket_a2.max * ket_b2.max)
            @printf("   Norm: %12.8f\n", sqrt(norm_curr))
        end
        fvec = sign * fvec

        # Prepare block matrix for S2 projection
        fvec2 = reshape(fvec, ket_a1.max, ket_a2.max, ket_b1.max, ket_b2.max)
        fvec3 = permutedims(fvec2, [1, 3, 2, 4])
        fvec4 = reshape(fvec3, ket_a1.max * ket_b1.max, ket_a2.max * ket_b2.max)
        block_matrix = fvec4'

        ### S2-adapted block SVD ###
        norbs_block = norbs1 
        nα_block = fock[1]
        nβ_block = fock[2]
        local_ansatz = FCIAnsatz(norbs_block, nα_block, nβ_block)      
        S2_matrix = build_S2_matrix(local_ansatz)  # Construct S^2 matrix
        eigen_obj = eigen(S2_matrix)
        S2_eigvals = eigen_obj.values
        S2_eigvecs = eigen_obj.vectors
        if verbose >2 
            @printf("   S² eigenvalues computed\n")
            for S2 in S2_eigvals
                @printf(" %f ", S2)
            end
        end
        rounded_S2 = round.(S2_eigvals, digits=8)
        fixed_S2 = map(x -> abs(x), rounded_S2)
        unique_S2 = unique(fixed_S2)
        println()
        if verbose>2
            @printf("   Unique S² eigenvalues: ")
            for S2 in unique_S2
                @printf(" %f ", S2)
            end
            println()
            println("shape of block matrix: ", size(block_matrix))
            println("shape of S2 eigvecs: ", size(S2_eigvecs))
            println()
        end    

        schmidt_cols = Matrix{Float64}(undef, rows, 0)    
        fock_sector_nkeep = 0
        if verbose>2
            @printf("   Project and SVD in each S² block\n")
        end
        
        for S2 in unique_S2
            
            idxs = findall(x -> abs(x - S2) < 1e-3, S2_eigvals)
            idxs_in_block_matrix = filter(i -> i <= rows, idxs)
            block_fvec = block_matrix_S2basis[idxs_in_block_matrix, :]
            @printf("   S² block %f\n", S2)
            @printf("   %5s %12s\n", "State", "Weight")
            rho = block_fvec * block_fvec'
            eigvals_, eigvecs_ = eigen(rho)
            kept_indices = findall(x -> x > svd_thresh^2, eigvals_)
            kept_block_vectors = eigvecs_[:, kept_indices]

            # F_block = svd(block_fvec, full=true)
            # kept_indices = findall(ni -> ni > svd_thresh, F_block.S)
            # kept_block_vectors = F_block.U[:, kept_indices]  # shape: (nblock, nkeep_block)
            nkeep = 0
            for (ni_idx, ni) in enumerate(eigvals_)#(F_block.S)
                if ni > svd_thresh
                    nkeep += 1
                    @printf("   %5i %12.8f\n", ni_idx, ni)
                else
                    @printf("   %5i %12.8f (discarded)\n", ni_idx, ni)
                end
            end
            # Reconstruct global basis columns for each kept vector:
            for j = 1:size(kept_block_vectors, 2)
                col = zeros(rows)
                col[idxs_in_block_matrix] .= kept_block_vectors[:, j]
                schmidt_cols = hcat(schmidt_cols, col)
            end
            fock_sector_nkeep += length(kept_indices)
        end

        if size(schmidt_cols, 2) > 0
            # schmidt_basis[fock] = schmidt_cols[1:fock_sector_nkeep, :]# is this the correct way to keep the right vectors
            schmidt_basis[fock] = schmidt_cols
            println("Final reconstructed Schmidt basis size: ", size(schmidt_cols))
        end
        
        ### END S2-adapted block SVD ###
    end
    println()
    for key in keys(schmidt_basis)
        println("Fock sector ", key, ": size ", size(schmidt_basis[key]))
    end
    return schmidt_basis
end
