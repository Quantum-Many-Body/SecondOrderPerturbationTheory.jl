```@meta
CurrentModule = SecondOrderPerturbationTheory
```

# Kitaev model on honeycomb lattice  

Construct the Kitaev model by projecting multi-orbital Hubbard model into the low-energy hilbert space.

```@example KitaevModel
using QuantumLattices: Lattice, Hopping, Hubbard, Onsite, OperatorGenerator, bonds, Hilbert,  Algorithm
using QuantumLattices:  InterOrbitalInterSpin, InterOrbitalIntraSpin, SpinFlip, PairHopping, matrix, Bond
using QuantumLattices: Point, Fock, MatrixCoupling, ⊗ ,rcoordinate, azimuthd, @σ_str, @L_str, FID
using ExactDiagonalization: BinaryBases,TargetSpace
using SecondOrderPerturbationTheory

#define lattice, hilbert, and multi-orbital Hubbard model
lattice = Lattice(
    [0.0, 0.0], [1/2, 1/(2*sqrt(3))];
    vectors = [[1.0, 0.0], [1/2.0, sqrt(3)/2]],
    )
hilbert = Hilbert(pid=>Fock{:f}(3, 2) for pid in 1:length(lattice))

tijz = [58.7 113.9 -7.0;
        113.9 58.7 -7.0;
        -7.0 -7.0 -194.1]
tijx = [-194.1 -7.0 -7.0;
        -7.0  58.7  113.9;
         -7.0 113.9 58.7]
tijy = [58.7 -7.0  113.9 ;
        -7.0  -194.1 -7.0;
         113.9 -7.0  58.7]


Lx = [0 0 0; 0  0 im; 0 -im 0]
Ly = [0 0 -im; 0 0 0; im 0 0]
Lz = [0 im 0; -im 0 0; 0 0 0]
L0 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

Jx = MatrixCoupling(:, FID, L0, σ"x", :)*0.5 + MatrixCoupling(:, FID, -L"x",σ"0", :)
Jy = MatrixCoupling(:, FID, L0, σ"y", :)*0.5 + MatrixCoupling(:, FID, -L"y",σ"0", :)
Jz = MatrixCoupling(:, FID, L0, σ"z", :)*0.5 + MatrixCoupling(:, FID, -L"z",σ"0", :)
J0 = MatrixCoupling(:, FID, L0, σ"0", :)

UU = 4000.0
Jₕ = 0.2*UU 
lambda = 140.0
function H1z(t_hop)
    function temp(bond::Bond)
    theta = bond|>rcoordinate|>azimuthd
    ( bond[1].site==1 && (theta≈270 ) ) && (return MatrixCoupling(:, FID,t_hop, σ"0",:))
     ( bond[1].site==2 && (theta≈90 ) ) && (return MatrixCoupling(:, FID,t_hop', σ"0",:))
    (return MatrixCoupling(:, FID, zeros(3,3), σ"0",:))
    end
    return temp
end
function H1x(t_hop)
    function temp(bond::Bond)
    theta = bond|>rcoordinate|>azimuthd
    ( bond[1].site==1 && (theta≈150 ) ) && (return MatrixCoupling(:, FID,t_hop, σ"0",:))
     ( bond[1].site==2 && (theta≈330 ) ) && (return MatrixCoupling(:, FID,t_hop', σ"0",:))
    (return MatrixCoupling(:, FID, zeros(3,3), σ"0",:))
    end
    return temp
end
function H1y(t_hop)
    function temp(bond::Bond)
    theta = bond|>rcoordinate|>azimuthd
    ( bond[1].site==1 && (theta≈30 ) ) && (return MatrixCoupling(:, FID,t_hop, σ"0",:))
     ( bond[1].site==2 && (theta≈210 ) ) && (return MatrixCoupling(:, FID,t_hop', σ"0",:))
    (return MatrixCoupling(:, FID, zeros(3,3), σ"0",:))
    end
    return temp
end
t = Hopping(:tz, 1.0, 1,H1z(tijz) )
t1 = Hopping(:tx, 1.0, 1,H1x(tijx) )
t2 = Hopping(:ty, 1.0, 1,H1y(tijy) )

U = Hubbard(:U, UU)
U′ = InterOrbitalInterSpin(Symbol("U′"), UU-2*Jₕ)
UmJ = InterOrbitalIntraSpin(Symbol("U′-J"), UU-3*Jₕ)
J = SpinFlip(Symbol("J"), Jₕ)
Jp = PairHopping(:Jp, Jₕ)

soc = 0.5*(MatrixCoupling(:, FID, L"x", σ"x", :) + MatrixCoupling(:, FID, L"y", σ"y", :) + MatrixCoupling(:, FID, L"z", σ"z", :))
λ = Onsite(:λ, Complex(lambda), soc)

#define the low-energy configure
bc = BinaryConfigure(pid=>TargetSpace(BinaryBases([1, 2, 3, 4, 5, 6], 5)) for pid in 1:length(lattice))
ls = PickState(1=>[[1, 2]], 2=>[[1, 2]])
sopt = SOPT(lattice, hilbert, (t, t1, t2), (U, U′, UmJ, J, Jp, λ), bc, ls)

#define the physical observables
s0 = Onsite(:s0, 1.0)
sx = Onsite(:sx, 1.0 + 0im, Jx)
sy = Onsite(:sy, 1.0 + 0im, Jy)
sz = Onsite(:sz, 1.0 + 0im, Jz)

p₀, = projectstate_points(sopt)


#obtain the exchange interactions of spin model
bond = bonds(lattice, 1)[4]
coeff = Coefficience([bond], 2, (s0, sx, sy, sz); halfspin=true)
res = Algorithm(:sopt, sopt)(:zbond, coeff)
#soptmatrix = matrix(sopt, bond)
#Jmat = coefficience_project(soptmatrix, coeff)
Jmat = res[2].data[2][1]

#test
txx, tyx, tzx, txy, tyy, tzy, txz, tyz, tzz = tijz[1, 1], tijz[2, 1], tijz[3, 1], tijz[1, 2], tijz[2, 2], tijz[3, 2], tijz[1, 3], tijz[2, 3], tijz[3, 3]
lambda1 = lambda
A = -1/3 * (Jₕ+9*lambda1+3*UU) / (6*Jₕ^2-UU*(3*lambda1+UU)+Jₕ*(4*lambda1+UU))
η = Jₕ / (6*Jₕ^2+(3*lambda1+UU)*(3*lambda1+2*UU)-Jₕ*(17*lambda1+8*UU))
B = 4/3 * (3*Jₕ-3*lambda1-UU) / (6*Jₕ-3*lambda1-2*UU)*η

J23 = 8*A/9*(-(txy-tyx)*(txz-tzx)-(tyz-tzy)*(txx+tyy+tzz)) + 4*B/9*(txy*(5*txz-2*tzx)+5*tyx*(tzx+2*txz)+(5*tyz+tzy)*(tyy+tzz-2*txx))
J32 = 8*A/9*(-(tyx-txy)*(tzx-txz)-(tzy-tyz)*(txx+tyy+tzz)) + 4*B/9*(tyx*(5*tzx-2*txz)+5*txy*(txz+2*tzx)+(5*tzy+tyz)*(tyy+tzz-2*txx))

J11 = 4*A/9*(-(txy-tyx)^2-(txz-tzx)^2+(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txy-tyx)^2+(txz-tzx)^2-2(2tyz+tzy)*(tyz+2tzy)+2(txx+tyy-2tzz)*(txx-2tyy+tzz))
J33 = 4*A/9*(-(txz-tzx)^2+(txy-tyx)^2-(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txz-tzx)^2+(tyz-tzy)^2-2(2txy+tyx)*(txy+2tyx)+2(tzz+tyy-2txx)*(txx-2tyy+tzz))

J12 = 8*A/9*(-(tyz-tzy)*(txz-tzx)-(txy-tyx)*(txx+tyy+tzz)) + + 4*B/9*(tzx*(5*tzy-2*tyz)+5*txz*(tyz+2*tzy)+(5*txy+tyx)*(tyy+txx-2*tzz))
J21 = 8*A/9*(-(tzy-tyz)*(tzx-txz)-(tyx-txy)*(txx+tyy+tzz)) + + 4*B/9*(txz*(5*tyz-2*tzy)+5*tzx*(tzy+2*tyz)+(5*tyx+txy)*(tyy+txx-2*tzz))

J22 = 4*A/9*(-(txy-tyx)^2+(txz-tzx)^2-(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txy-tyx)^2+(tyz-tzy)^2-2(2txz+tzx)*(txz+2tzx)+2(txx+tyy-2tzz)*(tyy-2txx+tzz))
J13 = 8*A/9*(-(tyz-tzy)*(tyx-txy)-(txz-tzx)*(txx+tyy+tzz)) + 4*B/9*(tyx*(5*tyz-2*tzy)+5*txy*(tzy+2*tyz)+(5*txz+tzx)*(tzz+txx-2*tyy))
J31 = 8*A/9*(-(tzy-tyz)*(txy-tyx)-(tzx-txz)*(txx+tyy+tzz)) + 4*B/9*(txy*(5*tzy-2*tyz)+5*tyx*(tyz+2*tzy)+(5*tzx+txz)*(tzz+txx-2*tyy))


(Jmat[2, 2] ≈ J11, Jmat[4, 4] ≈ J33, Jmat[2, 3] ≈ J12, Jmat[3, 4] ≈ J23, Jmat[3, 3] ≈ J22, Jmat[2, 4] ≈ J13)
```

## Construct the Generator of pseudospin-1/2
```@example KitaevModel
#define Generator of spin-1/2
gen = SpinOperatorGenerator(sopt, coeff;η=1e-10)

#latexformat of spin-1/2, add :icoord subscript.
using QuantumLattices: idtype, latexformat, LaTeX, expand
optspin = gen|>expand
T = optspin|>typeof|>eltype|>idtype|>eltype
latexformat(T, LaTeX{(:tag,), (:site, :icoordinate)}(:S, vectors=lattice.vectors))
optspin
```