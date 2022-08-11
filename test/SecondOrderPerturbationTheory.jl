using Test
using QuantumLattices: Lattice, dimension, ⊗, ⊕, matrix, expand, Table, PID, Point, Fock
using QuantumLattices: Hopping, Hubbard, Onsite, Generator, Bonds, Hilbert, Bond, @σˣ_str,@σʸ_str,@σᶻ_str
using ExactDiagonalization: BinaryBases, TargetSpace
using LinearAlgebra:diag
using SecondOrderPerturbationTheory


lattice=Lattice(:sq,[Point(PID(1),(0.0,)),Point(PID(2),(1.0,))], vectors=[[2.0]])
hilbert=Hilbert(pid=>Fock{:f}(norbital=1, nspin=2, nnambu=2) for pid in lattice.pids)
t=Hopping(:t,1.0,1)
U=Hubbard(:U,4.0)
μ=Onsite(:μ,-0.5)
h0=Generator((U,μ),Bonds(lattice),hilbert,half=false)
@testset "BinaryBases" begin
    bb = BinaryBases([1,2,3],2)
    bb₁ = BinaryBases([4,5,6],2)
    bb₂ = bb<<3
    @test bb₂ == bb₁
    bbbond = ⊠(bb₂, bb)
    bbbond₁ = ⊗(bb₂,bb)
    @test bbbond[1] == bbbond₁
    @test bbbond[1].table == bbbond₁.table
end
@testset "ProjectState" begin
    vectors₁ = rand(3,2)
    values₁ = [1, 4]
    bb = BinaryBases([1,2,3],2)
    ps₁ = ProjectState(values₁, vectors₁, TargetSpace(bb))
    @test dimension(ps₁) == 2
    @test dimension(ps₁,typeof(ps₁.basis)) == 3
    u = rand(2,2)
    @test (u*ps₁).vectors == (ps₁*u).vectors == (ps₁.vectors)*u
    
    vectors₂ = rand(3,3) + im*rand(3,3)
    values₂ = [2, 4.6, 3.5]
    ps₂ = ProjectState(values₂, vectors₂, TargetSpace(BinaryBases([1,2,3],1)))
    ps₁₂ = ps₁⊕ps₂
    @test ps₁₂.vectors[1:3,1:2] == ps₁.vectors
    @test ps₁₂.vectors[4:end,3:end] == ps₂.vectors
    @test ps₁₂.basis == [ps₁.basis.sectors; ps₂.basis.sectors]
    @test eltype(ps₂) == eltype(vectors₂) == ComplexF64
    
    ps₃ = (ProjectState([10,11],rand(2,2),TargetSpace(BinaryBases([1,2],1)))<<3)
    ps₃₂ = ps₂⊗ps₃
    @test ps₃₂.values == [ps₂.values .+ 10; ps₂.values .+ 11]
    @test ps₃₂.vectors == kron(ps₃.vectors, ps₂.vectors)
    @test ps₃₂.basis[1] == ps₂.basis[1]⊗ps₃.basis[1]
    
    ts = TargetSpace(BinaryBases([1,2], 2),BinaryBases([1,2],1))
    point = Point(PID(1), [0.0], [0.0])
    ops = expand(h0, point )
    table = Table(point, hilbert, SecondOrderPerturationMetric())
    ps = ProjectState(ops, ts, table, [[1],[1,2]])
    @test ps.values == Float64[3.0, -0.5, -0.5]
    @test ps.vectors == Float64[1.0 0 0; 0 0 1; 0 1 0]
    @test ps.basis == ts
end
@testset "ProjectStateBond" begin
    vectors₂ = rand(3,3) + im*rand(3,3)
    values₂ = [2, 4.6, 3.5]
    ps₂ = ProjectState(values₂, vectors₂, TargetSpace(BinaryBases([1,2,3],1)))
    ps₃ = (ProjectState([10,11],rand(2,2),TargetSpace(BinaryBases([1,2],1))))
    ps₃₂ = ps₂⊗(ps₃<<3)
    psb₂₃ = ProjectStateBond(ps₂, ps₃,0,3)
    @test psb₂₃.both.values == ps₃₂.values
    @test psb₂₃.both.vectors == ps₃₂.vectors
    @test psb₂₃.both.basis == ps₃₂.basis
end
@testset "SOPT,SecondOrderPerturbation and Coefficience" begin
    bond = Bond(1, Point(PID(1), [0.0], [0.0]), Point(PID(2), [1.0], [0.0]))
    bc = BinaryConfigure(pid=>TargetSpace(BinaryBases([1,2],1)) for pid in lattice.pids)
    ls = PickState(PID(1)=>[[1,2]], PID(2)=>[[1,2]])
    sopt = SOPT(lattice, hilbert, (t,), (U,μ), bc, ls)
    soptmat = matrix(sopt, bond)
    @test soptmat.bond == Bond(1, Point(PID(1), [0.0], [0.0]), Point(PID(2), [1.0], [0.0]))
    @test soptmat.m₀ ≈ Float64[-1.0 0.0 0.0 0.0; 0.0 -1.0 0.0 0.0; 0.0 0.0 -1.0 0.0; 0.0 0.0 0.0 -1.0]
    @test soptmat.m₂ ≈ Float64[0.0 0.0 0.0 0.0; 0.0 -0.5 0.5 0.0; 0.0 0.5 -0.5 0.0; 0.0 0.0 0.0 0.0]
    
    p₀,pp = projectstate_points(sopt::SOPT)
    σx=[0 1.0; 1 0]; σy=[0 -im; im 0]; σz=[1 0; 0 -1]; σ0=[1 0; 0 1]
    ob = Dict{PID,Vector{Matrix{ComplexF64}}}(PID(1)=>[σ0,σx,σy,σz],PID(2)=>[σ0,σx,σy,σz])
    coeff = Coefficience(ob;η=1e-12,order=-1)
    @test eltype(coeff) == Matrix{ComplexF64}
    coeff2 = Coefficience(Matrix{ComplexF64}[σ0,1.0*σx,σy,σz],lattice;η=1e-12,order=-1)
    @test coeff2.observables == coeff.observables
    s0 = Onsite(:s0,1.0)
    sx=Onsite(:sx,1.0+0im,couplings=σˣ"sp")
    sy=Onsite(:sx,1.0+0im,couplings=σʸ"sp")
    sz=Onsite(:sx,1.0+0im,couplings=σᶻ"sp")
    coeff3 = Coefficience(lattice,hilbert,(s0,sx,sy,sz),p₀)
    @test coeff3.observables==coeff.observables
    Jcoef3=coefficience_project(soptmat,coeff3)
    Jcoef2 = coefficience_project(soptmat,coeff2)
    @test real.(diag(Jcoef3)) ≈ real.(diag(Jcoef2))≈ [-1.25,0.25,0.25,0.25]
end