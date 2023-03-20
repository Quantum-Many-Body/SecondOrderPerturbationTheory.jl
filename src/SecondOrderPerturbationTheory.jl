module SecondOrderPerturbationTheory
using Printf: @printf
using LinearAlgebra: eigen, Hermitian, diagm, ishermitian, pinv, norm, inv
using ExactDiagonalization: BinaryBasis, BinaryBases, TargetSpace, productable, isone, count
using Base.Iterators: product
using QuantumLattices: Hilbert, AbstractLattice, OperatorGenerator, Operators, Frontend, Term, Boundary, CompositeDict, Algorithm, Assignment, Parameters
using QuantumLattices: Transformation, Bond, id, expand!, isintracell, Neighbors, Point, SpinTerm, Spin, Action, rcoordinate, icoordinate
using QuantumLattices:  CompositeIndex, Index, Metric, OperatorUnitToTuple, plain, bonds, MatrixCoupling, SID
import QuantumLattices: dimension, ⊗, ⊕, matrix, expand, Table, initialize, run!, update!
using DelimitedFiles: writedlm

export ⊠, ProjectState, ProjectStateBond, BinaryConfigure, PickState, SecondOrderPerturbation
export SOPT, SOPTMatrix, hamiltonianeff, projectstate_points, SecondOrderPerturationMetric 
export coefficience_project, Coefficience, observables_project, SpinOperatorGenerator
"""
    ⊠(bs₁::BinaryBases, bs₂::BinaryBases) -> Tuple{BinaryBases, Vector{Int}}

Get the direct product of two sets of binary bases, and the permutation vector.
"""
function ⊠(bs₁::BinaryBases, bs₂::BinaryBases)
    @assert productable(bs₁, bs₂) "⊠ error: the input two sets of bases cannot be direct producted."
    table = Vector{promote_type(eltype(bs₁), eltype(bs₂))}(undef, dimension(bs₁)*dimension(bs₂))
    count₀ = 1
    for (b₁, b₂) in product(bs₁, bs₂)
        table[count₀] = b₁⊗b₂
        count₀ += 1
    end
    p = sortperm(table)
    return BinaryBases(sort!([bs₁.id; bs₂.id]; by=first), table[p]), p
end
"""
    Base.:(<<)(bs::BinaryBases, n::Int) -> BinaryBases
"""
function Base.:(<<)(bs::BinaryBases, n::Int) 
    idlist = eltype(bs.id)[]
    nstate = 0
    for v in bs.id
        nstate += count(first(v))
        vv = (BinaryBasis(v[1].rep << n), v[2])
        push!(idlist, vv)
    end
    @assert sizeof(eltype(bs.table))*8 - nstate ≥ n ":<< error: the bit length of Unsigned is not enough to represent the states."
    table = eltype(bs.table)[]
    for bbasis in bs.table
        rep = bbasis.rep << n
        push!(table, BinaryBasis(rep))
    end
    return BinaryBases(idlist, table)
end

"""
    ProjectState{V<:Real, C<:Number, P<:TargetSpace} 

Projected states contain eigenvalues, eigenvectors, and basis of H₀. 
"""
struct ProjectState{V<:Real, C<:Number, P<:TargetSpace} 
    values :: Vector{V}
    vectors :: Matrix{C}
    basis :: P
    function ProjectState(values::Vector{V}, vectors::Matrix{C}, basis::TargetSpace) where {V<:Real,C<:Number}
        (nbasis, nvec) = size(vectors)
        @assert length(values) == nvec && (sum(dimension.(basis.sectors)) == nbasis ) "ProjectState error: dimensions of values, vectors, and basis do not match each other."
        new{V, C, typeof(basis)}(values, vectors, basis)
    end
end
Base.eltype(ps::ProjectState) = eltype(typeof(ps))
Base.eltype(::Type{ProjectState{V, C, P}}) where {V<:Real, C<:Number, P<:TargetSpace} = C
"""
    dimension(ps::ProjectState) -> Int

The dimension of local low-energy hilbert space.
"""
@inline dimension(ps::ProjectState) = length(ps.values)
"""
    dimension(ps::ProjectState,::Type{T}) where T<:TargetSpace -> Int

The dimension of target space `basis`.
"""
@inline dimension(ps::ProjectState, ::Type{T}) where T<:TargetSpace = sum(dimension.(ps.basis.sectors))

@inline function (Base.:*)(ps::ProjectState, u::Matrix{<:Number}) 
    @assert size(u, 1) == size(ps.vectors, 2) ":* error: dimensions of unitary matrix and eigenvectors donot match each other."
    return ProjectState(ps.values, ps.vectors*u, ps.basis) 
end
@inline (Base.:*)(u::Matrix{<:Number}, ps::ProjectState) = ps*u
function Base.convert(::Type{ProjectState{V, C, P}}, ps::ProjectState{V1, C1, P}) where {V1<:Real,C1<:Number, V<:Real, C<:Number, P<:TargetSpace}
    val = map(x->convert(V, x), ps.values)
    vec = C.(ps.vectors)
    return ProjectState(val, vec, ps.basis)
end
function Base.show(io::IO, ps::ProjectState)
    @printf io "%s(%s)=" ":values" typeof(ps.values)
    Base.show(io, ps.values)
    @printf io "\n %s(%s)=" ":vectors" typeof(ps.vectors)
    Base.show(io, ps.vectors)
    @printf io "\n %s(%s)=" ":basis" nameof(typeof(ps.basis))
    Base.show(io, ps.basis)
end

"""
    ⊕(ps₁::ProjectState, ps₂::ProjectState) -> ProjectState
    
Get the direct sum of two projected states.
"""
function ⊕(ps₁::ProjectState, ps₂::ProjectState)
    dtype₁ = promote_type(eltype(ps₁.values), eltype(ps₂.values))
    dtype₂ = promote_type(eltype(ps₁.vectors), eltype(ps₂.vectors))
    nbasis₁, nvec₁ = size(ps₁.vectors) 
    nbasis₂, nvec₂ = size(ps₂.vectors)
    values = (dtype₁)[ps₁.values; ps₂.values]
    vectors = zeros(dtype₂, nbasis₁ + nbasis₂, nvec₁ + nvec₂)
    vectors[1:nbasis₁, 1:nvec₁] = ps₁.vectors
    vectors[1+nbasis₁:end, 1+nvec₁:end] = ps₂.vectors
    basis = ps₁.basis⊕ps₂.basis  
    return ProjectState(values, vectors, basis)
end 
"""
    ⊗(ps₁::ProjectState, ps₂::ProjectState) -> ProjectState

Get the direct product of two sets of projected states.
"""
function ⊗(ps₁::ProjectState, ps₂::ProjectState)
    dtype₁ = promote_type(eltype(ps₁.values), eltype(ps₂.values))
    values = (dtype₁)[]
    for (val₁, val₂) in product(ps₁.values, ps₂.values)
        push!(values, val₁ + val₂)
    end
    bs = []
    p = Int[]
    count₀ = 0
    for (bs₁, bs₂) in product(ps₁.basis.sectors, ps₂.basis.sectors) 
        bs₀, p₀ = bs₁⊠bs₂
        push!(bs, bs₀)
        append!(p, p₀ .+ count₀)
        count₀ += length(p₀)
    end
    basis = TargetSpace(bs...)
    vectors = (Base.kron(ps₂.vectors, ps₁.vectors))[p, :]
    return ProjectState(values, vectors, basis)
end 
"""
    Base.:(<<)(ps::ProjectState, n::Int) -> ProjectState

Left bit shift opeartor. The BinaryBasis is left shifted by n bits.
"""
function Base.:(<<)(ps::ProjectState, n::Int)    
       basis = TargetSpace([bs << n for bs in ps.basis.sectors]...)
       return ProjectState(ps.values, ps.vectors, basis)
end
"""
    ProjectState(ops::Operators, braket::BinaryBases, table; pick::Union{UnitRange{Int}, Vector{Int}, Colon}=:)
    ProjectState(ops::Operators, ts::TargetSpace, table, pick::Vector{Vector{Int}})

Construct `ProjectState`. The pick::Union{UnitRange{Int}, Vector{Int}, Colon} argument picks the low-energy states. The i-th element of arguement pick vector is the loaction of low-energy states in the i-th `Sector` of `TargetSpace`.
"""
function ProjectState(ops::Operators, braket::BinaryBases, table; pick::Union{UnitRange{Int}, Vector{Int}, Colon}=:)
    hm = matrix(ops, (braket, braket), table)
    hm2 = Hermitian(Array(hm+hm')/2)
    F = eigen(hm2)
    basis = TargetSpace(braket)
    return ProjectState(F.values[pick], F.vectors[:, pick], basis)
end
function ProjectState(ops::Operators, ts::TargetSpace, table, pick::Vector{Vector{Int}})
    res = []
    for (i, braket) in enumerate(ts.sectors)
        ps = ProjectState(ops, braket, table; pick = pick[i])
        push!(res, ps)
    end
    return reduce(⊕, res)
end
"""
    ProjectStateBond(left::ProjectState, right::ProjectState)

Projected states on a bond. Construct `ProjectStateBond` by the two `ProjectState`s defined on point.
"""
struct ProjectStateBond
    left::ProjectState
    right::Union{ProjectState, Nothing}
    both::ProjectState
    function ProjectStateBond(left::ProjectState, right::Union{ProjectState, Nothing})
        isnothing(right) && return new(left, right, left)
        both = left⊗right
        new(left, right, both)
    end
end

"""
    ProjectStateBond(left::ProjectState, right::ProjectState, nleft::Int, nright::Int) 

The `nleft` and `nright` are the number of bits of left bit shift operator for `left` and `right` ProjectState, respectively.
"""
@inline function ProjectStateBond(left::ProjectState, right::ProjectState, nleft::Int, nright::Int) 
    left₁ = (left<<nleft)
    right₁ = (right<<nright)
    return ProjectStateBond(left₁, right₁)
end
function Base.show(io::IO, ps::ProjectStateBond)
    @printf io "%s= \n" "left ProjectState" 
    Base.show(io, ps.left)
    @printf io "\n %s= \n" "right ProjectState" 
    Base.show(io, ps.right)
    @printf io "\n %s = %s" "both ProjectState" "left⊗right"
end
"""
    BinaryConfigure{I<:TargetSpace, P<:Int} <: CompositeDict{P, I}
    BinaryConfigure(ps::Pair...)
    BinaryConfigure(kv)

Construct `BinaryConfigure` at a lattice. The local binary configure is given by `BinaryConfigure`.
"""
struct BinaryConfigure{I<:TargetSpace, P<:Int} <: CompositeDict{P, I}
    contents::Dict{P, I}
end
BinaryConfigure(ps::Pair...) = BinaryConfigure(ps)
function BinaryConfigure(kv)
    contents = Dict(kv)
    return BinaryConfigure{valtype(contents), keytype(contents)}(contents)
end

"""
    PickState{I<:Vector{Vector{Int}}, P<:Int} <: CompositeDict{P, I}
    PickState(ps::Pair...)
    PickState(kv) 

Construct  `PickState`. Pick low-energy states at a lattice.
"""
struct PickState{I<:Vector{Vector{Int}}, P<:Int} <: CompositeDict{P, I}
    contents::Dict{P, I}
end
PickState(ps::Pair...) = PickState(ps)
function PickState(kv)
    contents = Dict(kv)
    return PickState{valtype(contents), keytype(contents)}(contents)
end


#SecondOrderPerturtation
"""
    SecondOrderPerturationMetric <: Metric

Define the table metric, (:spin, :orbital) for a point, (:rcoordinate,:site, :spin, :orbital) for a bond.
"""
struct SecondOrderPerturationMetric <: Metric end
(m::SecondOrderPerturationMetric)(oid::CompositeIndex) = (oid.rcoordinate, oid.index.site, oid.index.iid.spin, oid.index.iid.orbital)
(m::SecondOrderPerturationMetric)(::Type{Point}) = OperatorUnitToTuple(:spin, :orbital)
function Table(bond::Bond, hilbert::Hilbert, m::SecondOrderPerturationMetric)
    if bond|>length == 2
        pid₁, pid₂ = bond[1].site, bond[2].site
        int₁, int₂ = hilbert[pid₁], hilbert[pid₂]
        oids₁ = [CompositeIndex(Index(pid₁, iid), bond[1].rcoordinate, bond[1].icoordinate) for iid in int₁]
        oids₂ = [CompositeIndex(Index(pid₂, iid), bond[2].rcoordinate, bond[2].icoordinate) for iid in int₂]
        return Table([oids₁; oids₂], m)
    elseif length(bond) == 1
        pid = bond[1].site
        int = hilbert[pid]
        oid = [CompositeIndex(Index(pid, iid), bond[1].rcoordinate, bond[1].icoordinate) for iid in int]
        return Table(oid, m(Point))
    else
        error("not support for length(bond) > 2")
    end
end

"""
    SecondOrderPerturbation{B<:BinaryConfigure, L<:PickState} <: Transformation
    (::SecondOrderPerturbation)(H₁::OperatorGenerator, p₀::Dict{T,<:ProjectState}, qₚ::Dict{T,<:ProjectState}, qₘ::Dict{T,<:ProjectState}, bond::Bond) where T<:Int  -> SOPTMatrix
    (::SecondOrderPerturbation)(H₀::OperatorGenerator, H₁::OperatorGenerator, bond::Bond) -> SOPTMatrix

A type. 
"""
struct SecondOrderPerturbation{B<:BinaryConfigure, L<:PickState} <: Transformation
    binaryconfigure::B
    pickstate::L 
    function SecondOrderPerturbation(bc::BinaryConfigure, ls::PickState)
        new{typeof(bc), typeof(ls)}(bc, ls)
    end
end
function (sodp::SecondOrderPerturbation)(H₁::OperatorGenerator, p₀::Dict{T,<:ProjectState}, qₚ::Dict{T,<:ProjectState}, qₘ::Dict{T,<:ProjectState}, bond::Bond) where T<:Int 
    @assert length(bond) == 2 "(::SecondOrderPerturbation) error: the bond should have two points." 
    left = p₀[bond[1].site]
    right = p₀[bond[2].site]
    nleft, nright = (bond[1].rcoordinate, bond[1].site) > (bond[2].rcoordinate, bond[2].site) ? (dimension(H₁.hilbert[bond[2].site])÷2, 0) : (0, dimension(H₁.hilbert[bond[1].site])÷2)
    p = ProjectStateBond(left, right, nleft, nright)
    q₁ = ⊗((qₚ[bond[1].site])<<nleft, (qₘ[bond[2].site])<<nright)
    q₂ = ⊗((qₘ[bond[1].site])<<nleft, (qₚ[bond[2].site])<<nright)
    q = (q₁⊕q₂)
    opts = expand(H₁, bond)
    table = Table(bond, H₁.hilbert, SecondOrderPerturationMetric())
    m₀, m₀₁, m₂ = hamiltonianeff(p.both, q, opts, table) 
    return SOPTMatrix(bond, p, m₀, m₀₁, m₂)
end
function (sodp::SecondOrderPerturbation)(H₀::OperatorGenerator, H₁::OperatorGenerator, bond::Bond)
    points = collect(bond)
    p₀, qₚ, qₘ, qₙ = projectstate_points(sodp.binaryconfigure, sodp.pickstate, H₀, points) 
    length(bond) == 2 && return sodp(H₁, p₀, qₚ, qₘ, bond)
    if length(bond) == 1
        opts = expand(H₁, bond)
        table = Table(bond, H₁.hilbert, SecondOrderPerturationMetric())
        m₀, m₀₁, m₂ = hamiltonianeff(p₀[bond[1].site], qₙ[bond[1].site], opts, table) 
        pb = ProjectStateBond(p₀[bond[1].site], nothing)
        return SOPTMatrix(bond, pb, m₀, m₀₁, m₂)
    end
end
"""
    expand(gen::OperatorGenerator, bond::Bond) -> Operators
"""
function expand(gen::OperatorGenerator, bond::Bond)
    result = zero(valtype(gen))
    map(term->expand!(result, term, bond, gen.hilbert; half=gen.half), gen.terms)
    isintracell(bond) || for opt in result
        result[id(opt)] = gen.operators.boundary(opt)
    end
    return result
end
"""
    projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::OperatorGenerator, points::AbstractVector{<:Point}) -> Tuple{Dict, Dict, Dict, Dict}

Construct the `ProjectState`` type of low-energy states with N-particle space, high-energy states with (N+1)-particle space, high-energy states with (N-1)-particle space, and high-energy states with N-particle space.
"""
function projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::OperatorGenerator, points::AbstractVector{<:Point}) 
    pp₀, ppₚ, ppₘ, ppₙ = [], [], [], []
    dtype = Float64
    for point in points
        table = Table(Bond(point), H₀.hilbert, SecondOrderPerturationMetric())
        p₀, pₚ, pₘ, pₙ =  _projectstate_points(bc, ls, H₀, point, table)
        push!(pp₀, point.site=>p₀)
        push!(ppₚ, point.site=>pₚ)
        push!(ppₘ, point.site=>pₘ)
        push!(ppₙ, point.site=>pₙ)
        dtype = promote_type(eltype(p₀), dtype, eltype(pₚ), eltype(pₘ))
    end
    T = ProjectState{Float64, dtype, valtype(bc)}
    psp, psqₚ, psqₘ = Dict{keytype(bc),T}(pp₀), Dict{keytype(bc),T}(ppₚ), Dict{keytype(bc),T}(ppₘ)
    psqₙ = Dict{keytype(bc),T}(ppₙ)
    return psp, psqₚ, psqₘ, psqₙ
end
function _projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::OperatorGenerator, point::Point, table::Table)
        tsₚ, tsₘ, pcₚ, pcₘ = _high_configure(bc, ls, point.site)
        opts = expand(H₀, Bond(point))
        pick = ls[point.site]
        ts = bc[point.site]   
        pick2 = Vector{Int}[]     
        for (i, sect) in enumerate(ts)
            temp = setdiff(collect(1:length(sect)), pick[i])
            push!(pick2, temp)
        end
        p = ProjectState(opts, ts, table, pick)
        qₙ = ProjectState(opts, ts, table, pick2)
        qₚ = ProjectState(opts, tsₚ, table, pcₚ)
        qₘ = ProjectState(opts, tsₘ, table, pcₘ) 
    return p, qₚ, qₘ, qₙ
end
function _high_configure(bc::BinaryConfigure, ls::PickState, pid::Int)
        key = pid
        plus = []
        minus = []
        dimp = Vector{Int}[]
        dimm = Vector{Int}[]
        for bb in bc[key].sectors
            rep = reduce(|, map(x->first(x), bb.id)) 
            nparticle = reduce(+, map(x->Int(x[2]), bb.id))
            bbp = BinaryBases(findone(BinaryBasis(rep)), nparticle + 1)
            push!(plus, bbp)
            bbm = BinaryBases(findone(BinaryBasis(rep)), nparticle - 1)
            push!(minus, bbm)
            push!(dimp, collect(1:dimension(bbp)))
            push!(dimm, collect(1:dimension(bbm)))
        end
        for (i, bb) in enumerate( bc[key].sectors )
            bb ∈ plus && (dimp[i] = setdiff(dimp[i], ls[key][i]))
            bb ∈ minus && (dimm[i] = setdiff(dimm[i], ls[key][i]))   
        end
    return TargetSpace(plus...), TargetSpace(minus...), dimp, dimm
end
@inline function findone(basis::BinaryBasis)
    stop = ndigits(basis.rep, base=2)
    res = Int[]
    for i = 1:stop
        isone(basis, i) && ( push!(res, i) )
    end
    return res 
end
"""  
    hamiltonianeff(psp::ProjectState, psq::ProjectState, h1::Operators, table::Table)  ->Tuple{Matrix, Matrix, Matrix}

Get the effective Hamiltonian, the first and second terms of the result correspond to the zero-th and 2nd perturbations respectively.
"""
function hamiltonianeff(psp::ProjectState, psq::ProjectState, h1::Operators, table::Table) 
    heff0 = diagm(psp.values)
    heff01 = psp.vectors'*Array(matrix(h1, psp.basis, table))*psp.vectors
    m₀ = heff0 
    m₀₁ = Hermitian((heff01 + heff01')/2)
    @assert maximum(norm.(heff01 - heff01'))<1e-12 "hamiltonianeff error: the zero-th perturbations matrix should be hermitian."
    tqp = psq.vectors'*Array(matrix(h1, psq.basis, psp.basis, table))*psp.vectors
    m, n = size(psp.vectors, 2), size(psq.vectors, 2)
    sqp = zeros(eltype(psp.vectors), n, m)
    for i = 1:m
        for j = 1:n
            sqp[j,i] = -tqp[j,i]/(-psp.values[i] + psq.values[j])
        end
    end
    spq = - sqp'
    tpq = tqp' 
    m₂ = (tpq*sqp - spq*tqp )/2.0
    @assert maximum(norm.(m₂ - m₂'))<1e-14 "hamiltonianeff error: the 2nd perturbations matrix should be hermitian."
    return m₀, m₀₁, m₂
end

"""
    SOPT{L<:AbstractLattice, G₁<:OperatorGenerator, G₀<:OperatorGenerator, PT<:SecondOrderPerturbation} <: Frontend

Second order perturbation theory method of a electronic quantum lattice system.
"""
struct SOPT{L<:AbstractLattice, G₁<:OperatorGenerator, G₀<:OperatorGenerator, PT<:SecondOrderPerturbation} <: Frontend
    lattice::L
    H₁::G₁
    H₀::G₀
    configure::PT
    function SOPT(lattice::AbstractLattice, H₁::OperatorGenerator, H₀::OperatorGenerator, configure::SecondOrderPerturbation) 
        new{typeof(lattice), typeof(H₁), typeof(H₀), typeof(configure)}(lattice, H₁, H₀, configure)
    end
end
"""
    SOPT(lattice::AbstractLattice, hilbert::Hilbert, terms₁::Tuple{Vararg{Term}}, terms₀::Tuple{Vararg{Term}}, binaryconfigure::BinaryConfigure, lowstate::PickState; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)

Construct the second order perturbation method for a quantum lattice system.
"""
function SOPT(lattice::AbstractLattice, hilbert::Hilbert, terms₁::Tuple{Vararg{Term}}, terms₀::Tuple{Vararg{Term}}, binaryconfigure::BinaryConfigure, lowstate::PickState; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)
    isnothing(neighbors) && (neighbors = maximum(terms₁->terms₁.bondkind, terms₁))
    H₁ = OperatorGenerator(terms₁, bonds(lattice, neighbors), hilbert; half=false, boundary=boundary)
    H₀ = OperatorGenerator(terms₀, bonds(lattice, 0), hilbert; half=false, boundary=boundary)
    configure = SecondOrderPerturbation(binaryconfigure, lowstate)
    return SOPT(lattice, H₁, H₀, configure)
end
@inline function update!(sopt::SOPT; kwargs...)
    if length(kwargs)>0
        update!(sopt.H₀; kwargs...)
        update!(sopt.H₁; kwargs...)
    end
    return sopt
end
@inline Parameters(sopt::SOPT) = Parameters{(keys(Parameters(sopt.H₀))...,keys(Parameters(sopt.H₁))...)}((Parameters(sopt.H₀))...,(Parameters(sopt.H₁))... )

"""
    matrix(sopt::SOPT, bond::Bond) -> SOPTMatrix

Return matrix of the projected hamiltionian onto a bond of which the length is greater than 1.
"""
@inline function matrix(sopt::SOPT, bond::Bond) 
    return (sopt.configure)(sopt.H₀, sopt.H₁, bond)  
end
@inline function matrix(sopt::SOPT, bonds::Vector{<:Bond}) 
    H₀, H₁ = sopt.H₀, sopt.H₁
    points = [ p[1] for p in H₀.bonds if length(p) == 1]
    bc, ls = sopt.configure.binaryconfigure, sopt.configure.pickstate
    p₀, qₚ, qₘ, qₙ = projectstate_points(bc, ls, H₀, points) 
    res = []
    for bond in bonds
        length(bond) == 2 && push!(res, (sopt.configure)(sopt.H₁, p₀, qₚ, qₘ, bond))
        if length(bond) == 1
            opts = expand(H₁, bond)
            table = Table(bond, H₁.hilbert, SecondOrderPerturationMetric())
            m₀, m₀₁, m₂ = hamiltonianeff(p₀[bond[1].site], qₙ[bond[1].site], opts, table) 
            pb = ProjectStateBond(p₀[bond[1].site], nothing)
            push!(res, SOPTMatrix(bond, pb, m₀, m₀₁, m₂))
        end 
    end 
    return res 
end
"""
    projectstate_points(sopt::SOPT) -> Tuple{Dict{Int, ProjectState}, Dict{Int, ProjectState}, Dict{Int, ProjectState}, Dict{Int, ProjectState}}

Construct `ProjectState` on all points. Construct the `ProjectState`` type of low-energy states with N-particle space, high-energy states with (N+1)-particle space, high-energy states with (N-1)-particle space, and high-energy states with N-particle space.
"""
function projectstate_points(sopt::SOPT)
    bc, ls = sopt.configure.binaryconfigure, sopt.configure.pickstate
    points = [ p[1] for p in sopt.H₀.bonds if length(p) == 1]
    p₀, qₚ, qₘ, qₙ = projectstate_points(bc, ls, sopt.H₀, points) 
    return p₀, qₚ, qₘ, qₙ
end
"""
    SOPTMatrix(bond::Bond, P₀::ProjectStateBond, m₀::Matrix, m₂::Matrix)

Matrix representation of the low-energy hamiltionian. The order of basis of representation is the order of (space of bond[2], space of bond[1]), i.e. (left space of bond[1])⊗(right space of bond[2])
# Arguments
-`bond`: bond of lattice
-`P₀`: projected state
-`m₀`: matrix representation of the zeroth order (<H₀>) of the low-energy hamiltionian
-`m₀₁`: matrix representation of the zeroth order (<H₁>) of the low-energy hamiltionian
-`m₂`: matrix representation of second order of the low-energy hamiltionian
"""
struct SOPTMatrix
    bond::Bond
    P₀::ProjectStateBond
    m₀::AbstractMatrix{<:Number}
    m₀₁::AbstractMatrix{<:Number}
    m₂::AbstractMatrix{<:Number}
end
function Base.show(io::IO, ps::SOPTMatrix)
    @printf io "%s: \n" nameof(typeof(ps))
    @printf io "%s= \n" "bond" 
    Base.show(io, ps.bond)
    @printf io "\n %s= \n" "zeroth order matrix(:m₀)" 
    Base.show(io, ps.m₀)
    @printf io "\n %s= \n" "zeroth order matrix(:m₀₁)" 
    Base.show(io, ps.m₀₁)
    @printf io "\n %s = \n" "second order matrix(:m₂)" 
    Base.show(io, ps.m₂)
end


"""
    Coefficience{P<:Int, I<:AbstractVector{<:Matrix{<:Number}}} <: Action

Decompose the projected-hamiltionian matrix with respect to input physical observables.
"""
struct Coefficience{I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}, O} <: Action
    bonds::AbstractVector{<:Bond}
    observables::Dict{Int, I}
    options::O
    function Coefficience(bonds::AbstractVector{<:Bond}, observables::Dict{Int, I}, options) where {I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}}
        new{valtype(observables), typeof(options)}(bonds, observables, options)
    end
end
"""
    Coefficience(bonds::AbstractVector{<:Bond}, observables::Dict{Int, I}; options...) where {I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}}
    Coefficience(bonds::AbstractVector{<:Bond}, observables::Dict{Int, I}; options...) where {I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}}
    Coefficience(bonds::AbstractVector{<:Bond}, npoints::Int, ob::I=Tuple{}(); options...) where {I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}}

`η` attribute in options is truncation of completeness of physical quantities. 
`order` attribute (order=-1,0,2) choose the order of matrix of effective hamiltonian to obtain the exchange coefficiences. 
`npoints` = length(lattice).
"""
@inline Coefficience(bonds::AbstractVector{<:Bond}, observables::Dict{Int, I}; options...) where {I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}} = Coefficience(bonds, observables, options)
@inline function Coefficience(bonds::AbstractVector{<:Bond}, npoints::Int, ob::I=Tuple{}(); options...) where {I<:Union{AbstractVector{<:AbstractMatrix}, Tuple{Vararg{Term}}}}
    observables = Dict{Int, typeof(ob)}()
    for pid in 1:npoints
        observables[pid] = ob 
    end 
    return Coefficience(bonds, observables, options)
end
@inline function initialize(cof::Coefficience, sopt::SOPT)
    x = [] #matrix_coef
    y = [] # coeff
    ob = [] 
    return x, y, ob
end
function run!(sopt::Algorithm{<:SOPT}, coef::Assignment{<:Coefficience}) 
    order = get(coef.action.options, :order, -1)
    η = get(coef.action.options, :η, 1e-14)
    eye = get(coef.action.options, :identitymatrix, true)
    hilbert = sopt.frontend.H₀.hilbert
    matbds = matrix(sopt.frontend, coef.action.bonds)
    append!(coef.data[1], matbds)

    for matbd in matbds
        # matbd = matrix(sopt.frontend, bond) 
        # push!(coef.data[1], matbd)
        ob = Dict{Int, Vector{Matrix{ComplexF64}}}()
        bond = matbd.bond
        if length(bond) == 2
            nleft, nright = (bond[1].rcoordinate, bond[1].site) > (bond[2].rcoordinate, bond[2].site) ? (dimension(hilbert[bond[2].site])÷2, 0) : (0, dimension(hilbert[bond[1].site])÷2)
        else
            nleft, nright = 0, 0 
        end
        for (i, point) in enumerate(bond) # length(bond) <= 2
            site = point.site
            psp = i==1 ? (matbd.P₀.left<<-nleft) : (matbd.P₀.right<<-nright)
            if (isa(coef.action.observables[site], Tuple{Vararg{Term}}))
                ob[site] = observables_project(coef.action.observables[site], point, hilbert, psp)
            elseif eye
                ob[site] = coef.action.observables[site]
            else
                mat = map(x->psp.vectors'*x*psp.vectors, coef.action.observables[site])
                ob[site] = mat
            end
        end
        order == -1 && (comat = length(bond)==2 ? coefficience_project(matbd.m₂+matbd.m₀₁, bond, ob, η) : coefficience_project(matbd.m₂+matbd.m₀₁+matbd.m₀, bond, ob, η))
        order == 2 && (comat = coefficience_project(matbd.m₂, bond, ob, η))
        order == 0 && (comat = coefficience_project(matbd.m₀₁+matbd.m₀, bond, ob, η))
        push!(coef.data[2], comat)   
        push!(coef.data[3], ob)
    end
end

"""
    coefficience_project(m₂::Matrix{<:Number}, bond::Bond, ob::Dict{Int, <:AbstractVector{<:Matrix{<:Number}}}, η::Float64=1e-14) -> Matrix
    
Get the coefficience of exchange interaction. The row index corresponds to bond[1], column index corresponds to bond[2].
"""
function coefficience_project(m₂::Matrix{<:Number}, bond::Bond, ob::Dict{Int, <:AbstractVector{<:Matrix{<:Number}}}, η::Float64=1e-14)
    if length(bond) == 2
        pids = [bond[1].site, bond[2].site]
        gsg = eltype(valtype(ob))[]
        n₁ = length(ob[pids[1]])
        n₂ = length(ob[pids[2]])
        for i in 1:n₂
            mi = ob[pids[2]][i]
            for j in 1:n₁
                mj = ob[pids[1]][j]
                push!(gsg, kron(mi, mj))
            end
        end
        return _coefficience_project(m₂, gsg, (n₁, n₂); η=η)
    elseif length(bond) == 1
        pids = [bond[1].site]
        gsg = eltype(valtype(ob))[]
        n₁ = length(ob[pids[1]])
        for j in 1:n₁
            mj = ob[pids[1]][j]
            push!(gsg, mj)
        end
        return _coefficience_project(m₂, gsg, (n₁, 1); η=η)
    end
end
function _coefficience_project(m₂::Matrix{<:Number}, gsg::AbstractVector{T}, nshape::Tuple{Int, Int}; η::Float64=1e-12) where T<:Matrix{<:Number} 
    b = m₂[:]
    nn = length(gsg)
    @assert length(b) == size(gsg[1])|>prod "coefficience error: length(m₂) [$(length(m₂[:]))] == the number of element of matrix of gsg [$(prod(size(gsg[1])))]."
    a = zeros(eltype(eltype(gsg)), length(b), nn)
    for i = 1:nn
        a[:, i] = gsg[i][:]
    end
    res = pinv(a)*b
    # res = inv(a)*b
    data = norm(b - a*res)/norm(b)
    maximum(abs.(imag.(res))) > 1e-9 && @warn "coefficience warning: the imaginary of coefficience $(maximum(abs.(imag.(res)))) is not ignorable."
    data >= η &&  @warn "coefficience warning: the number of physical observables is not enough ($(data)>η(=$(η)))."
    return reshape(res, nshape)
end
"""
    coefficience_project(matbd::SOPTMatrix, coef::Coefficience, hilbert::Hilbert) -> Matrix

"""
function coefficience_project(matbd::SOPTMatrix, coef::Coefficience, hilbert::Hilbert)
    order = get(coef.options, :order, -1)
    η = get(coef.options, :η, 1e-14)
    ob = Dict{Int, Vector{Matrix{ComplexF64}}}()
    bond = matbd.bond
    if length(bond) == 2
        nleft, nright = (bond[1].rcoordinate, bond[1].site) > (bond[2].rcoordinate, bond[2].site) ? (dimension(hilbert[bond[2].site])÷2, 0) : (0, dimension(hilbert[bond[1].site])÷2)
    else
        nleft, nright = 0, 0 
    end
    for (i, point) in enumerate(bond) # length(bond) <= 2
        site = point.site
        if (isa(coef.observables[site], Tuple{Vararg{Term}}))    
            psp = i==1 ? matbd.P₀.left<<-nleft : ( matbd.P₀.right<<-nright)
            ob[site] = observables_project(coef.observables[site], point, hilbert, psp)
        else
            ob[site] = coef.observables[site]
        end
    end
    order == -1 && (comat = length(bond)==2 ? coefficience_project(matbd.m₂+matbd.m₀₁, bond, ob, η) : coefficience_project(matbd.m₂+matbd.m₀₁+matbd.m₀, bond, ob, η))
    order == 2 && (comat = coefficience_project(matbd.m₂, bond, ob, η))
    order == 0 && (comat = coefficience_project(matbd.m₀₁+matbd.m₀, bond, ob, η))
    return comat
end

"""
    observables_project(terms::Tuple{Vararg{Term}}, point::Point, hilbert::Hilbert, psp::ProjectState) -> Vector{Matrix}

The order of `Term` in terms (Onsite) tuple determines the order of matrix in the result.
"""
function observables_project(terms::Tuple{Vararg{Term}}, point::Point, hilbert::Hilbert, psp::ProjectState)
    table = Table(Bond(point), hilbert, SecondOrderPerturationMetric())
    ops = map(term->expand(term, Bond(point), hilbert; half=false), terms)
    res = map(op->(psp.vectors'*Array(matrix(op, psp.basis, table))*psp.vectors), ops)
    return collect(res)
end

#output
import Base.write
using Latexify
function Base.write(filename::AbstractString, jcoef::Tuple{Algorithm{<:SOPT}, Assignment{<:Coefficience}}; scale=1)
    assign = jcoef[2]
    dist = []
    sites = []
    mats = []
    dist0 = []
    mats0 = []
    for (i, bond) in enumerate(assign.action.bonds)
        if !(bond.kind == 0)
            push!(dist, round(rcoordinate(bond)|>norm,digits=4))
            push!(sites, (bond[1].site, bond[2].site, icoordinate(bond)))
            push!(mats, assign.data[2][i]*scale)
        else
            push!(dist0, bond[1].site)
            push!(mats0, assign.data[2][i]*scale)
        end
    end
    ind = sortperm(dist)
    open(filename, "w") do f
        write(f, "J_ij S_i*S_j matrix of spin model (R = Rj-Ri in Cartesian coordinate): \n\n")
        for (i, site) in enumerate(dist0)
            mats0[i][abs.(mats0[i]).<1e-6] .=0
            write(f, "On-site J of atom = $(site) : \n")
            writedlm(f, round.(real.(mats0[i]), digits=5))
            write(f,"\n")
        end 
        for i in ind
            mat = mats[i]
            mat[abs.(mat).<1e-6] .=0
            write(f, "distance = $(dist[i]), atom_i = $(sites[i][1]), atom_j =$(sites[i][2]), R =$(sites[i][3]): \n")
            writedlm(f, round.(real.(mat), digits=5))
            write(f,"\n")
        end
    end
    open("Latex_"*filename, "w") do f
        write(f, "J_ij S_i*S_j matrix of spin model (R = Rj-Ri in Cartesian coordinate): \n\n")
        for (i, site) in enumerate(dist0)
            mats0[i][abs.(mats0[i]).<1e-6] .=0
            write(f, "On-site J of atom = $(site) : \n")
            write(f, latexify(round.(real.(mats0[i]), digits=5), fmt="%.5f"))
            write(f,"\n")
        end 
        for i in ind
            mat = mats[i]
            mat[abs.(mat).<1e-6] .=0
            write(f, "distance = $(dist[i]), atom_i = $(sites[i][1]), atom_j =$(sites[i][2]), R =$(sites[i][3]): \n")
            write(f, latexify(round.(real.(mat), digits=5), fmt="%.5f"))
            write(f,"\n")
        end
    end
end




# only for spin-1/2 case

"""
    SpinOperatorGenerator(st::SOPT, coef::Coefficience; η::Float64=1e-14) -> OperatorGenerator

Only support the pseudospin-1/2 case. The zeeman term (Onsite term) is ommited when the ground states are not Krameter states.
"""
function SpinOperatorGenerator(st::SOPT, coef::Coefficience; η::Float64=1e-14)
    flag = get(coef.options, :halfspin, false)
    if flag
        function spincp(j::Matrix{ComplexF64})
            j₀ = real.(j)
            return MatrixCoupling((1, 2), SID, j₀[2:4, 2:4])
        end
        function spincoupling(bond::Bond)
            soptm = matrix(st, bond )
            j = coefficience_project(soptm, coef, st.H₀.hilbert)
            j[norm.(j) .< η] .= 0.0 
            j[imag.(j) .< η] = real.(j[imag.(j) .< η])
            ex = spincp(j)
            return ex
        end
        bonds = [p for p in st.H₁.bonds if length(p)==2]
        terms = []
        cache = Set()
        for (i, bond) in enumerate(bonds)
            symb = Meta.parse("h$(i)b$(bond.kind)")
            if bond.kind ∉ cache
                push!(terms, SpinTerm(symb, one(ComplexF64), bond.kind, spincoupling))
                push!(cache, bond.kind)
            end
        end 
        hilbert = Hilbert(pid=>Spin{1//2}() for pid in 1:length(st.lattice))
        return OperatorGenerator(tuple(terms...), st.H₁.bonds, hilbert; half=false, boundary=plain)
    end
end
"""
    matrix(ops::Operators, ts₁::TargetSpace, ts₂::TargetSpace, table) -> Matrix
    matrix(ops::Operators, ts::TargetSpace, table) -> Matrix

Get the matrix of direct sum of submatrices.
"""
function matrix(ops::Operators, ts₁::TargetSpace, ts₂::TargetSpace, table)
    return hcat([vcat([matrix(ops, (bra, ket), table) for bra in ts₁.sectors]...) for ket in ts₂.sectors]...)
end
matrix(ops::Operators, ts::TargetSpace, table) = matrix(ops, ts, ts, table)
end #module 