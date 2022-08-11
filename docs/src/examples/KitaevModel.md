```@meta
CurrentModule = SecondOrderPerturbationTheory
```

# Kitaev model on honeycomb lattice  

Construct the Kitaev model by projecting multi-orbital Hubbard model into the low-energy hilbert space.

```@example KitaevModel
using QuantumLattices: Lattice, Hopping, Hubbard, Onsite, Generator, Bonds, Hilbert, @σˣ_str,@σʸ_str,@σᶻ_str, @fc_str
using QuantumLattices: Couplings, InterOrbitalInterSpin, InterOrbitalIntraSpin, SpinFlip, PairHopping
using QuantumLattices: PID, Point, Fock, FockCoupling, ⊗ ,rcoord, azimuthd 
using ExactDiagonalization: BinaryBases,TargetSpace
using SecondOrderPerturbationTheory
using Test 

#define lattice, hilbert, and multi-orbital Hubbard model
lattice = Lattice(:Honeycomb,
    [Point(PID(1),(0.0,0.0),(0.0,0.0)), Point(PID(2),[1/2,1/(2*sqrt(3))],[0.0,0.0])],
    vectors = [[1.0,0.0],[1/2.0,sqrt(3)/2]],
    neighbors = 1)
hilbert = Hilbert(pid=>Fock{:f}(norbital=3, nspin=2, nnambu=2) for pid in lattice.pids)

tij = [58.7 113.9 -7.0;113.9 58.7 -7.0; -7.0 -7.0 -194.1]

function fcmatrixob(tij::Matrix{<:Number}; kwargs...)
    fc = []
    n, m = size(tij)
    for i=1:m
        for j=1:n
            push!(fc, FockCoupling{2}(tij[j,i]; orbitals=(j,i), nambus=(2,1), kwargs...) )
        end
    end 
    return Couplings(fc...)
end

tfc = fcmatrixob(tij)
macro Lˣ_str(::String) fc"1.0im ob[2 3]" - fc"1.0im ob[3 2]" end
macro Lʸ_str(::String) fc"1.0im ob[3 1]" - fc"1.0im ob[1 3]" end
macro Lᶻ_str(::String) fc"1.0im ob[1 2]" - fc"1.0im ob[2 1]" end
macro L⁰_str(::String) fc"1.0 ob[1 1]" + fc"1.0 ob[2 2]" + fc"1.0 ob[3 3]" end
macro soc_str(::String) 0.5*(Lˣ""⊗σˣ"sp" + Lʸ""⊗σʸ"sp"+ Lᶻ""⊗σᶻ"sp") end
macro Jˣ_str(::String) 0.5*(L⁰""⊗σˣ"sp") - Lˣ""⊗σ⁰"sp" end
macro Jʸ_str(::String) 0.5*(L⁰""⊗σʸ"sp") - Lʸ""⊗σ⁰"sp" end
macro Jᶻ_str(::String) 0.5*(L⁰""⊗σᶻ"sp") - Lᶻ""⊗σ⁰"sp" end
macro J⁰_str(::String) L⁰""⊗σ⁰"sp" end

Lx = [0 0 0; 0  0 im;0 -im 0]
Ly = [0 0 -im; 0 0 0; im 0 0]
Lz = [0 im 0; -im 0 0; 0 0 0]
L0 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

UU = 4000.0
Jₕ = 0.2*UU 
lambda = 140.0
t = Hopping(:t, 1.0, 1, couplings=tfc, amplitude=bond->((bond|>rcoord|>azimuthd ≈ 270) ? 1 : 0) )
U = Hubbard(:U, UU)
U′ = InterOrbitalInterSpin(Symbol("U′"), UU-2*Jₕ)
UmJ = InterOrbitalIntraSpin(Symbol("U′-J"), UU-3*Jₕ)
J = SpinFlip(Symbol("J"), Jₕ)
Jp = PairHopping(:Jp, Jₕ)
λ = Onsite(:λ, 1.0+0.0im, couplings=lambda*soc"")

#define the low-energy configure
bc = BinaryConfigure(pid=>TargetSpace(BinaryBases([1,2,3,4,5,6],5)) for pid in lattice.pids)
ls = PickState(PID(1) =>[[1,2]], PID(2) =>[[1,2]])
sopt = SOPT(lattice, hilbert, (t,), (U, U′, UmJ, J, Jp, λ), bc, ls)

#define the physical observables
s0 = Onsite(:s0, 1.0)
sx=Onsite(:sx, 1.0+0im, couplings=Jˣ"")
sy=Onsite(:sx, 1.0+0im, couplings=Jʸ"")
sz=Onsite(:sx, 1.0+0im, couplings=Jᶻ"")
p₀ = projectstate_points(sopt)
coeff = Coefficience(lattice, hilbert, (s0,sx,sy,sz), p₀[1];η=1e-10 )

#obtain the exchange interactions of spin model
bond = Bonds(lattice)[4]
soptmatrix = matrix(sopt,bond)
Jmat = coefficience_project(soptmatrix, coeff)

#test
txx,tyx,tzx,txy,tyy,tzy,txz,tyz,tzz = tij[1,1],tij[2,1],tij[3,1],tij[1,2],tij[2,2],tij[3,2],tij[1,3],tij[2,3],tij[3,3]
lambda1=lambda
A = -1/3*(Jₕ+9*lambda1+3*UU)/(6*Jₕ^2-UU*(3*lambda1+UU)+Jₕ*(4*lambda1+UU))
η = Jₕ/(6*Jₕ^2+(3*lambda1+UU)*(3*lambda1+2*UU)-Jₕ*(17*lambda1+8*UU))
B = 4/3*(3*Jₕ-3*lambda1-UU)/(6*Jₕ-3*lambda1-2*UU)*η

J23 = 8*A/9*(-(txy-tyx)*(txz-tzx)-(tyz-tzy)*(txx+tyy+tzz)) + 4*B/9*(txy*(5*txz-2*tzx)+5*tyx*(tzx+2*txz)+(5*tyz+tzy)*(tyy+tzz-2*txx))
J11 = 4*A/9*(-(txy-tyx)^2-(txz-tzx)^2+(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txy-tyx)^2+(txz-tzx)^2-2(2tyz+tzy)*(tyz+2tzy)+2(txx+tyy-2tzz)*(txx-2tyy+tzz))
J33 = 4*A/9*(-(txz-tzx)^2-(txy-tyx)^2+(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txz-tzx)^2+(tyz-tzy)^2-2(2txy+tyx)*(txy+2tyx)+2(tzz+tyy-2txx)*(txx-2tyy+tzz))
J12 = 8*A/9*(-(tyz-tzy)*(txz-tzx)-(txy-tyx)*(txx+tyy+tzz)) + + 4*B/9*(tzx*(5*tzy-2*tyz)+5*txz*(tyz+2*tzy)+(5*txy+tyx)*(tyy+txx-2*tzz))
@testset "KitaevModel" begin
    @test Jmat[2,2] ≈ J11
    @test Jmat[4,4] ≈ J33
    @test Jmat[2,3] ≈ J12
    @test Jmat[3,4] ≈ J23
end

#define `Generator` of spin-1/2
gen = coefficience_project(sopt, coeff;η=1e-10)
```