module SecondOrderPerturbationTheory
using Printf: @printf
using LinearAlgebra: eigen, Hermitian, diagm, ishermitian, pinv, norm
using ExactDiagonalization: BinaryBasis, BinaryBases, TargetSpace, productable, isone
using Base.Iterators: product
using QuantumLattices: Hilbert, AbstractLattice, Generator, Operators, Engine, Term, Boundary, CompositeDict, AbstractPID
using QuantumLattices: Transformation, Bond, id, expand!, isintracell, AbstractBond, rank, PID, Point
using QuantumLattices:  OID, Index, Metric, OIDToTuple, plain, Bonds
import QuantumLattices: dimension, ⊗, ⊕, matrix, expand, Table

using ExactDiagonalization: isone, one, zero, count, Operator, Operators, creation
using QuantumLattices: statistics, Action
using SparseArrays: SparseMatrixCSC, spzeros

export ⊠, ProjectState, ProjectStateBond, BinaryConfigure, PickState, SecondOrderPerturbation
export SOPT, SOPTMatrix, high_configure, hamiltonianeff, projectstate_points, SecondOrderPerturationMetric 
export coefficience_project, Coefficience, observables_project
"""
    ⊠(bs₁::BinaryBases, bs₂::BinaryBases) -> Tuple{BinaryBases, Vector{Int}}

Get the direct product of two sets of binary bases, and the permutation vector.
"""
function ⊠(bs₁::BinaryBases, bs₂::BinaryBases)
    @assert productable(bs₁, bs₂) "⊠ error: the input two sets of bases cannot be direct producted."
    table = Vector{promote_type(eltype(bs₁), eltype(bs₂))}(undef, dimension(bs₁)*dimension(bs₂))
    count = 1
    for (b₁, b₂) in product(bs₁, bs₂)
        table[count] = b₁⊗b₂
        count += 1
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
Base.eltype(::Type{ProjectState{V,C,P}}) where {V<:Real, C<:Number, P<:TargetSpace} = C
"""
    dimension(ps::ProjectState) -> Int

The dimension of local low-energy hilbert space.
"""
@inline dimension(ps::ProjectState) = length(ps.values)
"""
    dimension(ps::ProjectState,::Type{T}) where T<:TargetSpace -> Int

The dimension of target space `basis`.
"""
@inline dimension(ps::ProjectState,::Type{T}) where T<:TargetSpace = sum(dimension.(ps.basis.sectors))

@inline function (Base.:*)(ps::ProjectState, u::Matrix{<:Number}) 
    @assert size(u, 1) == size(ps.vectors, 2) ":* error: dimensions of unitary matrix and eigenvectors donot match each other."
    return ProjectState(ps.values, ps.vectors*u, ps.basis) 
end
@inline (Base.:*)(u::Matrix{<:Number}, ps::ProjectState) = ps*u
function Base.convert(::Type{ProjectState{V,C,P}}, ps::ProjectState{V1,C1,P}) where {V1<:Real,C1<:Number, V<:Real,C<:Number,P<:TargetSpace}
    val = map(x->convert(V,x), ps.values)
    vec = C.(ps.vectors)
    return ProjectState(val,vec,ps.basis)
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
    count = 0
    for (bs₁, bs₂) in product(ps₁.basis.sectors, ps₂.basis.sectors) 
        bs₀, p₀ = bs₁⊠bs₂
        push!(bs, bs₀)
        append!(p, p₀ .+ count)
        count += length(p₀)
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

Construct `ProjectState`.
"""
function ProjectState(ops::Operators, braket::BinaryBases, table; pick::Union{UnitRange{Int}, Vector{Int}, Colon}=:)
    hm = matrix(ops, (braket, braket), table)
    hm2 = Hermitian(Array(hm+hm')/2)
    F = eigen(hm2)
    basis = TargetSpace(braket)
    return ProjectState(F.values[pick], F.vectors[:,pick], basis)
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

Construct `ProjectStateBond`.
"""
struct ProjectStateBond
    left::ProjectState
    right::ProjectState
    both::ProjectState
    function ProjectStateBond(left::ProjectState, right::ProjectState)
        both = left⊗right
        new(left, right, both)
    end
end
@inline ProjectStateBond(ps::ProjectState) = ProjectStateBond(ps, ps)
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
    BinaryConfigure{I<:TargetSpace, P<:AbstractPID} <: CompositeDict{P, I}
    BinaryConfigure(ps::Pair...)
    BinaryConfigure(kv)

Construct `BinaryConfigure` at a lattice. The local binary configure is given by `BinaryConfigure`.
"""
struct BinaryConfigure{I<:TargetSpace, P<:AbstractPID} <: CompositeDict{P, I}
    contents::Dict{P, I}
end
BinaryConfigure(ps::Pair...) = BinaryConfigure(ps)
function BinaryConfigure(kv)
    contents = Dict(kv)
    return BinaryConfigure{valtype(contents),keytype(contents)}(contents)
end

"""
    PickState{I<:Vector{Vector{Int}}, P<:AbstractPID} <: CompositeDict{P, I}
    PickState(ps::Pair...)
    PickState(kv) 

Construct  `PickState`. Pick low-energy states at a lattice.
"""
struct PickState{I<:Vector{Vector{Int}}, P<:AbstractPID} <: CompositeDict{P, I}
    contents::Dict{P, I}
end
PickState(ps::Pair...) = PickState(ps)
function PickState(kv)
    contents = Dict(kv)
    return PickState{valtype(contents),keytype(contents)}(contents)
end


#SecondOrderPerturtation
struct SecondOrderPerturationMetric <: Metric end
(m::SecondOrderPerturationMetric)(oid::OID) = (oid.rcoord, oid.index.pid.site, oid.index.iid.spin, oid.index.iid.orbital)
(m::SecondOrderPerturationMetric)(::Type{Point}) = OIDToTuple(:spin, :orbital)
function Table(bond::AbstractBond, hilbert::Hilbert, m::SecondOrderPerturationMetric)
    if bond|>rank == 2
        pid₁, pid₂ = bond.spoint.pid, bond.epoint.pid
        int₁, int₂ = hilbert[pid₁], hilbert[pid₂]
        oids₁ = [OID(Index(pid₁, iid), bond.spoint.rcoord, bond.spoint.icoord) for iid in int₁]
        oids₂ = [OID(Index(pid₂, iid), bond.epoint.rcoord, bond.epoint.icoord) for iid in int₂]
        return Table([oids₁; oids₂], m)
    elseif rank(bond) == 1
        pid = bond.pid
        int = hilbert[pid]
        oid = [OID(Index(pid, iid), bond.rcoord, bond.icoord) for iid in int]
        return Table(oid, m(Point))
    else
        error("not support for rank(bond) >2")
    end
end

"""
    SecondOrderPerturbation{B<:BinaryConfigure, L<:PickState} <: Transformation

A type.
"""
struct SecondOrderPerturbation{B<:BinaryConfigure, L<:PickState} <: Transformation
    binaryconfigure::B
    pickstate::L 
    function SecondOrderPerturbation(bc::BinaryConfigure, ls::PickState)
        new{typeof(bc), typeof(ls)}(bc, ls)
    end
end
"""
    (sodp::SecondOrderPerturbation)(H₁::Generator, p₀::Dict{T,<:ProjectState}, qₚ::Dict{T,<:ProjectState}, qₘ::Dict{T,<:ProjectState}, bond::Bond) where T<:AbstractPID  -> SOPTMatrix
    (sodp::SecondOrderPerturbation)(H₀::Generator, H₁::Generator, bond::Bond) -> SOPTMatrix

"""
function (sodp::SecondOrderPerturbation)(H₁::Generator, p₀::Dict{T,<:ProjectState}, qₚ::Dict{T,<:ProjectState}, qₘ::Dict{T,<:ProjectState}, bond::Bond) where T<:AbstractPID  
    left = p₀[bond.spoint.pid]
    right = p₀[bond.epoint.pid]
    nleft, nright = (bond.spoint.rcoord, bond.spoint.pid.site) > (bond.epoint.rcoord, bond.epoint.pid.site) ? (dimension(right, typeof(right.basis)), 0) : (0, dimension(left, typeof(left.basis)))
    p = ProjectStateBond(left, right, nleft, nright)
    q₁ = ⊗((qₚ[bond.spoint.pid])<<nleft, (qₘ[bond.epoint.pid])<<nright)
    q₂ = ⊗((qₘ[bond.spoint.pid])<<nleft, (qₚ[bond.epoint.pid])<<nright)
    q = q₁⊕q₂
    opts = expand(H₁, bond)
    table = Table(bond, H₁.hilbert, SecondOrderPerturationMetric())
    m₀, m₂ = hamiltonianeff(p.both, q, opts, table) 
    return SOPTMatrix(bond, p, m₀, m₂)
end
function (sodp::SecondOrderPerturbation)(H₀::Generator, H₁::Generator, bond::Bond)
    points = [bond.spoint, bond.epoint]
    p₀, qₚ, qₘ = projectstate_points(sodp.binaryconfigure, sodp.pickstate, H₀, points) 
    return sodp(H₁, p₀, qₚ, qₘ, bond)
end
"""
    expand(gen::Generator, bond::AbstractBond) -> Operators
"""
function expand(gen::Generator, bond::AbstractBond)
    result = zero(valtype(gen))
    map(term->expand!(result, term, bond, gen.hilbert, half=gen.half, table=gen.table), gen.terms)
    isintracell(bond) || for opt in result
        result[id(opt)] = gen.operators.boundary(opt)
    end
    return result
end
"""
    projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator) -> Tuple{Dict{PID,ProjectState},Dict{PID,ProjectState}, Dict{PID,ProjectState}}
    projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator, points::AbstractVector{<:Point}) -> Tuple{Dict{PID,ProjectState},Dict{PID,ProjectState}, Dict{PID,ProjectState}}

Construct the `ProjectState`` type of low-energy states within N-particle space, high-energy states with (N+1)-particle space, and high-energy states with (N-1)-particle space.
"""
function projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator)
    points = [ p for p in H₀.bonds if rank(p) == 1]
    return projectstate_points(bc, ls, H₀, points)
end
function projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator, points::AbstractVector{<:Point}) 
    pp₀, ppₚ, ppₘ = [], [], []
    dtype = Float64
    for point in points
        table = Table(point, H₀.hilbert, SecondOrderPerturationMetric())
        p₀, pₚ, pₘ =  _projectstate_points(bc, ls, H₀, point, table)
        push!(pp₀, point.pid=>p₀)
        push!(ppₚ, point.pid=>pₚ)
        push!(ppₘ, point.pid=>pₘ)
        dtype = promote_type(eltype(p₀), dtype, eltype(pₚ), eltype(pₘ))
    end
   T = ProjectState{Float64, dtype,valtype(bc)}
    psp, psqₚ, psqₘ = Dict{keytype(bc),T}(pp₀), Dict{keytype(bc),T}(ppₚ), Dict{keytype(bc),T}(ppₘ)
    return psp, psqₚ, psqₘ 
end
function _projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator, point::Point, table::Table)
        tsₚ, tsₘ, pcₚ, pcₘ = _high_configure(bc, ls, point.pid)
        opts = expand(H₀, point)
        pick = ls[point.pid]
        ts = bc[point.pid]
        p = ProjectState(opts, ts, table, pick)
        qₚ = ProjectState(opts, tsₚ, table, pcₚ)
        qₘ= ProjectState(opts, tsₘ, table, pcₘ) 
    return p, qₚ, qₘ
end
"""
    high_configure(bc::BinaryConfigure, ls::PickState) -> Tuple
    high_configure(bc::BinaryConfigure, ls::PickState, pids::AbstractVector{PID}) -> Tuple

Get the high-energy configure of local space.
"""
high_configure(bc::BinaryConfigure, ls::PickState) = high_configure(bc, ls, collect(keys(bc)))
function high_configure(bc::BinaryConfigure, ls::PickState, pids::AbstractVector{<:AbstractPID})
    bc₁, bc₂, ls₁, ls₂ = Dict(), Dict(), Dict(), Dict()
    for key in pids
        tsₚ, tsₘ, pcₚ, pcₘ = _high_configure(bc, ls, key)
        bc₁[key], bc₂[key], ls₁[key], ls₂[key] = tsₚ, tsₘ, pcₚ, pcₘ
    end
    return bc₁, bc₂, ls₁, ls₂ 
end
function _high_configure(bc::BinaryConfigure, ls::PickState, pid::AbstractPID)
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
    hamiltonianeff(psp::ProjectState, psq::ProjectState, h1::Operators, table::Table)  ->Tuple{Matrix,Matrix}

Get the effective Hamiltonian, the first and second terms of the result correspond to the zero-th and 2nd perturbations respectively.
"""
function hamiltonianeff(psp::ProjectState, psq::ProjectState, h1::Operators, table::Table) 
    heff0 = diagm(psp.values)
    heff01 = psp.vectors'*Array(matrix(h1, psp.basis, table))*psp.vectors
    m₀ = heff0 + heff01
    @assert ishermitian(m₀) "hamiltonianeff error: the zero-th perturbations matrix should be hermitian."
    tqp = psq.vectors'*Array(matrix(h1, psq.basis, psp.basis, table))*psp.vectors
    m, n = size(psp.vectors,2), size(psq.vectors,2)
    sqp = zeros(eltype(psp.vectors), n, m)
    for i = 1:m
        for j = 1:n
            sqp[j,i] = -tqp[j,i]/(-psp.values[i] + psq.values[j])
        end
    end
    spq = - sqp'
    tpq = tqp' 
    m₂ = (tpq*sqp - spq*tqp )/2.0
    @assert ishermitian(m₂) "hamiltonianeff error: the 2nd perturbations matrix should be hermitian."
    return m₀, m₂
end

"""
    SOPT{L<:AbstractLattice, G₁<:Generator, G₀<:Generator, PT<:SecondOrderPerturbation} <: Engine

Second order perturbation theory method of a electronic quantum lattice system.
"""
struct SOPT{L<:AbstractLattice, G₁<:Generator, G₀<:Generator, PT<:SecondOrderPerturbation} <: Engine
    lattice::L
    H₁::G₁
    H₀::G₀
    configure::PT
    function SOPT(lattice::AbstractLattice, H₁::Generator, H₀::Generator, configure::SecondOrderPerturbation) 
        new{typeof(lattice), typeof(H₁), typeof(H₀), typeof(configure)}(lattice, H₁, H₀, configure)
    end
end
"""
    SOPT(lattice::AbstractLattice, hilbert::Hilbert, terms₁::Tuple{Vararg{Term}}, terms₀::Tuple{Vararg{Term}}, binaryconfigure::BinaryConfigure, lowstate::PickState; boundary::Boundary=plain)

Construct the second order perturbation method for a quantum lattice system.
"""
function SOPT(lattice::AbstractLattice, hilbert::Hilbert, terms₁::Tuple{Vararg{Term}}, terms₀::Tuple{Vararg{Term}}, binaryconfigure::BinaryConfigure, lowstate::PickState; boundary::Boundary=plain)
    H₁ = Generator(terms₁, Bonds(lattice), hilbert; half=false, boundary=boundary)
    H₀ = Generator(terms₀, Bonds(lattice), hilbert; half=false, boundary=boundary)
    configure = SecondOrderPerturbation(binaryconfigure, lowstate)
    return SOPT(lattice, H₁, H₀, configure)
end
@inline function update!(sopt::SOPT; kwargs...)
    if length(kwargs)>0
        update!(sopt.H₀; kwargs...)
        update!(sopt.H₁, kwargs...)
    end
    return sopt
end
"""
    matrix(sopt::SOPT, bond::Bond) -> SOPTMatrix
"""
@inline function matrix(sopt::SOPT, bond::Bond) 
    return (sopt.configure)(sopt.H₀, sopt.H₁, bond)  
end
"""
    matrix(sopt::SOPT) -> Vector

Obtain `SOPTMatrix` on all bonds.
"""
function matrix(sopt::SOPT)
    bc, ls = sopt.configure.binaryconfigure, sopt.configure.pickstate
    p₀, qₚ, qₘ = projectstate_points(bc, ls, sopt.H₀) 
    res = []
    bonds = [ b for b in sopt.H₁.bonds if rank(b)==2]
    for bond in bonds
        push!(res, (sopt.configure)(sopt.H₁, p₀, qₚ, qₘ, bond))
    end
    return res
end
"""
    projectstate_points(sopt::SOPT) -> Tuple{Dict{PID, ProjectState},Dict{PID, ProjectState},Dict{PID, ProjectState}}

Construct `ProjectState` on all points.
"""
function projectstate_points(sopt::SOPT)
    bc, ls = sopt.configure.binaryconfigure, sopt.configure.pickstate
    p₀, qₚ, qₘ = projectstate_points(bc, ls, sopt.H₀) 
    return p₀, qₚ, qₘ
end

"""
    SOPTMatrix(bond::Bond, P₀::ProjectStateBond, m₀::Matrix, m₂::Matrix)

Matrix representation of the low-energy hamiltionian.
# Arguments
-`bond`: bond of lattice
-`P₀`: projected state
-`m₀`: matrix representation of the zeroth order of the low-energy hamiltionian
-`m₂`: matrix representation of second order of the low-energy hamiltionian
"""
struct SOPTMatrix
    bond::Bond
    P₀::ProjectStateBond
    m₀::Matrix{<:Number}
    m₂::Matrix{<:Number}
end
function Base.show(io::IO, ps::SOPTMatrix)
    @printf io "%s: \n" nameof(typeof(ps))
    @printf io "%s= \n" "bond" 
    Base.show(io, ps.bond)
    @printf io "\n %s= \n" "zeroth order matrix" 
    Base.show(io, ps.m₀)
    @printf io "\n %s = \n" "second order matrix" 
    Base.show(io, ps.m₂)
end


struct Coefficience{P<:AbstractPID, I<:AbstractVector{<:Matrix{<:Number}}} <: Action
    observables:: Dict{P, I}
    η:: Float64
    order::Int
    dim::Int
    function Coefficience(observables::Dict{P,I}, η::Float64, order::Int, dim::Int ) where {P<:AbstractPID, I<:AbstractVector{<:Matrix{<:Number}}}
        new{keytype(observables),valtype(observables)}(observables, η, order, dim)
    end
end
@inline Coefficience(observables::Dict{<:AbstractPID, <:AbstractVector{<:Matrix{<:Number}}}; η::Float64=1e-12, order::Int= -1, dim::Int=2) = Coefficience(observables, η, order, dim) 
@inline function Coefficience(ob::AbstractVector{<:Matrix{<:Number}}, lattice::AbstractLattice; order::Int=-1, η::Float64=1e-12, dim::Int=2)
    coeff = Dict{PID,typeof(ob)}()
    for pid in lattice.pids
        coeff[pid] = ob 
    end 
    return Coefficience(coeff; η=η, order=order, dim=dim)
end
Base.eltype(coeff::Coefficience) = eltype(valtype(coeff.observables))
function Coefficience(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, p₀::Dict{<:AbstractPID, <:ProjectState}; η::Float64=1e-12, order::Int=-1, dim::Int=2)
    points = [ p for p in Bonds(lattice) if rank(p) == 1]
    ob = Dict{keytype(p₀), Vector{Matrix{ComplexF64}}}()
    for point in points 
        ob[point.pid] = observables_project(terms, point, hilbert, p₀[point.pid])
    end
    return Coefficience(ob; η=η, order=order,dim=dim)
end
"""
    observables_project(term::Term, point::Point, hilbert::Hilbert, psp::ProjectState) -> Vector
"""
function observables_project(terms::Tuple{Vararg{Term}}, point::Point, hilbert::Hilbert, psp::ProjectState)
    table = Table(point, hilbert, SecondOrderPerturationMetric())
    ops = map(term->expand(term, point, hilbert; half=false), terms)
    res = map(op->psp.vectors'*Array(matrix(op, psp.basis, table))*psp.vectors, ops)
    return collect(res)
end

function coefficience_project(m₂::Matrix{<:Number}, gsg::AbstractVector{T}, nshape::Tuple{Int,Int}; η::Float64=1e-12) where T<:Matrix{<:Number} 
    b = m₂[:]
    nn = length(gsg)
    @assert length(b) == size(gsg[1])|>prod "coefficience error: length(m₂) [$(length(m₂))] == the number of element of matrix of gsg [$(prod(size(gsg[1])))]."
    a = zeros(eltype(eltype(gsg)), nn, nn)
    for i = 1:nn
        a[:,i] = gsg[i][:]
    end
    res = pinv(a)*b
    data = norm(b - a*res)
    data >= η &&  @warn "coefficience warning: the physical observables is not enough ($(data)>η(=$(η)))."
    return reshape(res, nshape)
end
function coefficience_project(m₂::Matrix{<:Number}, bond::Bond, coeff::Coefficience)
    pids = [bond.spoint.pid, bond.epoint.pid]
    gsg = eltype(coeff)[]
    n₁ = length(coeff.observables[pids[1]])
    n₂ = length(coeff.observables[pids[2]])
    for i in 1:n₂
        mi = coeff.observables[pids[2]][i]
        for j in 1:n₁
            mj = coeff.observables[pids[1]][j]
            push!(gsg, kron(mi, mj))
        end
    end
    return coefficience_project(m₂, gsg, (n₁, n₂); η=coeff.η)
end
function coefficience_project(soptm::SOPTMatrix, coeff::Coefficience)
    if coeff.order < 0
       return coefficience_project(soptm.m₂+soptm.m₀, soptm.bond, coeff) 
    elseif coeff.order == 0
        return  coefficience_project(soptm.m₀, soptm.bond, coeff)
    elseif coeff.order == 2
        return  coefficience_project(soptm.m₂, soptm.bond, coeff)
    else
        error("coefficience_project error: not support when coeff.order != 1 or 2 or -1")
    end
end
# only for spin-1/2 case
using QuantumLattices: SpinCoupling, SpinTerm, Couplings, Spin, rcoord, kind
function coefficience_project(sopt::SOPT, coeff::Coefficience; η::Float64=1e-14)
    st, cof = sopt, coeff
    if cof.dim == 2
        function spincp(j::Matrix{ComplexF64})
            j = real.(j)
            jxx, jxy, jxz = j[2,2], j[2,3], j[2,4]
            jyx, jyy, jyz = j[3,2], j[3,3], j[3,4]
            jzx, jzy, jzz = j[4,2], j[4,3], j[4,4]
            return (Couplings(SpinCoupling(jxx,('x','x')),
                SpinCoupling(jxy,('x','y')),
                SpinCoupling(jxz,('x','z')),
                SpinCoupling(jyx,('y','x')),
                SpinCoupling(jyy,('y','y')),
                SpinCoupling(jyz,('y','z')),
                SpinCoupling(jzx,('z','x')),
                SpinCoupling(jzy,('z','y')),
                SpinCoupling(jzz,('z','z'))
                ))
        end
        function spincoupling(bond)
            soptm = matrix(st, bond )
            j = coefficience_project(soptm, cof)
            j[norm.(j) .< η] .= 0.0 
            j[imag.(j) .< η] = real.(j[imag.(j) .< η])
            println(bond)
            ex = spincp(j)
            return ex
        end
        bonds=[p for p in Bonds(st.lattice) if rank(p)==2]
        terms = []
        cache = Set()
        for (i, bond) in enumerate(bonds)
            symb = Meta.parse("h$(i)b$(bond.neighbor)")
            if kind(bond) ∉ cache
                push!(terms, SpinTerm(symb, one(ComplexF64), bond.neighbor, spincoupling))

                push!(cache, kind(bond))
            end
        end 
        hilbert = Hilbert(pid=>Spin{1//2}(norbital=1) for pid in st.lattice.pids)
        return Generator(tuple(terms...), Bonds(st.lattice), hilbert; half=false, boundary=plain)
    end
end

function matrix(ops::Operators, ts₁::TargetSpace, ts₂::TargetSpace, table)
    return hcat([vcat([matrix2(ops, (bra, ket), table) for bra in ts₁.sectors]...) for ket in ts₂.sectors]...)
end
matrix(ops::Operators, ts::TargetSpace, table) = matrix(ops, ts, ts, table)

# matrix of ED
function matrix2(op::Operator, braket::NTuple{2, BinaryBases}, table; dtype=valtype(op))
    bra, ket = braket[1], braket[2]
    ndata, intermediate = 1, zeros(ket|>eltype, rank(op)+1)
    data, indices, indptr = zeros(dtype, dimension(ket)), zeros(Int, dimension(ket)), zeros(Int, dimension(ket)+1)
    sequence = NTuple{rank(op), Int}(table[op[i]] for i in reverse(1:rank(op)))
    iscreation = NTuple{rank(op), Bool}(oid.index.iid.nambu==creation for oid in reverse(id(op)))
    for i = 1:dimension(ket)
        flag = true
        indptr[i] = ndata
        intermediate[1] = ket[i]
        for j = 1:rank(op)
            isone(intermediate[j], sequence[j])==iscreation[j] && (flag = false; break)
            intermediate[j+1] = iscreation[j] ? one(intermediate[j], sequence[j]) : zero(intermediate[j], sequence[j])
        end
        if flag
            nsign = 0
            statistics(eltype(op))==:f && for j = 1:rank(op)
                nsign += count(intermediate[j], 1, sequence[j]-1)
            end
            index = searchsortedfirst(intermediate[end], bra)
            if index <= dimension(bra) && bra[index] == intermediate[end]
                indices[ndata] = index
                data[ndata] = op.value*(-1)^nsign    
                ndata += 1
            end
        end
    end
    indptr[end] = ndata
    return SparseMatrixCSC(dimension(bra), dimension(ket), indptr, indices[1:ndata-1], data[1:ndata-1])
end
function matrix2(ops::Operators, braket::NTuple{2, BinaryBases}, table; dtype=valtype(eltype(ops)))
    result = spzeros(dtype, dimension(braket[1]), dimension(braket[2]))
    for op in ops
        result += matrix2(op, braket, table; dtype=dtype)
    end
    return result
end

end #module 