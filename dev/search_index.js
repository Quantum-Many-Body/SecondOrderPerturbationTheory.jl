var documenterSearchIndex = {"docs":
[{"location":"examples/KitaevModel/","page":"Kitaev model on honeycomb lattice","title":"Kitaev model on honeycomb lattice","text":"CurrentModule = SecondOrderPerturbationTheory","category":"page"},{"location":"examples/KitaevModel/#Kitaev-model-on-honeycomb-lattice","page":"Kitaev model on honeycomb lattice","title":"Kitaev model on honeycomb lattice","text":"","category":"section"},{"location":"examples/KitaevModel/","page":"Kitaev model on honeycomb lattice","title":"Kitaev model on honeycomb lattice","text":"Construct the Kitaev model by projecting multi-orbital Hubbard model into the low-energy hilbert space.","category":"page"},{"location":"examples/KitaevModel/","page":"Kitaev model on honeycomb lattice","title":"Kitaev model on honeycomb lattice","text":"using QuantumLattices: Lattice, Hopping, Hubbard, Onsite, Generator, Bonds, Hilbert, @σˣ_str,@σʸ_str,@σᶻ_str, @fc_str, @σ⁰_str\nusing QuantumLattices: Couplings, InterOrbitalInterSpin, InterOrbitalIntraSpin, SpinFlip, PairHopping, matrix\nusing QuantumLattices: PID, Point, Fock, FockCoupling, ⊗ ,rcoord, azimuthd \nusing ExactDiagonalization: BinaryBases,TargetSpace\nusing SecondOrderPerturbationTheory\n\n#define lattice, hilbert, and multi-orbital Hubbard model\nlattice = Lattice(:Honeycomb,\n    [Point(PID(1),(0.0,0.0),(0.0,0.0)), Point(PID(2),[1/2,1/(2*sqrt(3))],[0.0,0.0])],\n    vectors = [[1.0,0.0],[1/2.0,sqrt(3)/2]],\n    neighbors = 1)\nhilbert = Hilbert(pid=>Fock{:f}(norbital=3, nspin=2, nnambu=2) for pid in lattice.pids)\n\ntij = [58.7 113.9 -7.0;\n        113.9 58.7 -7.0;\n        -7.0 -7.0 -194.1]\ntijx = [-194.1 -7.0 -7.0;\n        -7.0  58.7  113.9;\n         -7.0 113.9 58.7]\ntijy = [58.7 -7.0  113.9 ;\n        -7.0  -194.1 -7.0;\n         113.9 -7.0  58.7]\n\nfunction fcmatrixob(tij::Matrix{<:Number}; kwargs...)\n    fc = []\n    n, m = size(tij)\n    for i=1:m\n        for j=1:n\n            push!(fc, FockCoupling{2}(tij[j,i]; orbitals=(j,i), nambus=(2,1), kwargs...) )\n        end\n    end \n    return Couplings(fc...)\nend\n\ntfc = fcmatrixob(tij)\ntfcx = fcmatrixob(tijx)\ntfcy = fcmatrixob(tijy)\nmacro Lˣ_str(::String) fc\"1.0im ob[2 3]\" - fc\"1.0im ob[3 2]\" end\nmacro Lʸ_str(::String) fc\"1.0im ob[3 1]\" - fc\"1.0im ob[1 3]\" end\nmacro Lᶻ_str(::String) fc\"1.0im ob[1 2]\" - fc\"1.0im ob[2 1]\" end\nmacro L⁰_str(::String) fc\"1.0 ob[1 1]\" + fc\"1.0 ob[2 2]\" + fc\"1.0 ob[3 3]\" end\nmacro soc_str(::String) 0.5*(Lˣ\"\"⊗σˣ\"sp\" + Lʸ\"\"⊗σʸ\"sp\"+ Lᶻ\"\"⊗σᶻ\"sp\") end\nmacro Jˣ_str(::String) 0.5*(L⁰\"\"⊗σˣ\"sp\") - Lˣ\"\"⊗σ⁰\"sp\" end\nmacro Jʸ_str(::String) 0.5*(L⁰\"\"⊗σʸ\"sp\") - Lʸ\"\"⊗σ⁰\"sp\" end\nmacro Jᶻ_str(::String) 0.5*(L⁰\"\"⊗σᶻ\"sp\") - Lᶻ\"\"⊗σ⁰\"sp\" end\nmacro J⁰_str(::String) L⁰\"\"⊗σ⁰\"sp\" end\n\nLx = [0 0 0; 0  0 im; 0 -im 0]\nLy = [0 0 -im; 0 0 0; im 0 0]\nLz = [0 im 0; -im 0 0; 0 0 0]\nL0 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]\n\nUU = 4000.0\nJₕ = 0.2*UU \nlambda = 140.0\nt = Hopping(:tz, 1.0, 1, couplings=tfc, amplitude=bond->((bond|>rcoord|>azimuthd ≈ 270 || bond|>rcoord|>azimuthd ≈ 90) ? 1 : 0) )\nt1 = Hopping(:tx, 1.0, 1, couplings=tfcx, amplitude=bond->((bond|>rcoord|>azimuthd ≈ 150 || bond|>rcoord|>azimuthd ≈ 330) ? 1 : 0) )\nt2 = Hopping(:ty, 1.0, 1, couplings=tfcy, amplitude=bond->((bond|>rcoord|>azimuthd ≈ 30 || bond|>rcoord|>azimuthd ≈ 210) ? 1 : 0) )\n\nU = Hubbard(:U, UU)\nU′ = InterOrbitalInterSpin(Symbol(\"U′\"), UU-2*Jₕ)\nUmJ = InterOrbitalIntraSpin(Symbol(\"U′-J\"), UU-3*Jₕ)\nJ = SpinFlip(Symbol(\"J\"), Jₕ)\nJp = PairHopping(:Jp, Jₕ)\nλ = Onsite(:λ, 1.0+0.0im, couplings=lambda*soc\"\")\n\n#define the low-energy configure\nbc = BinaryConfigure(pid=>TargetSpace(BinaryBases([1,2,3,4,5,6],5)) for pid in lattice.pids)\nls = PickState(PID(1)=>[[1,2]], PID(2)=>[[1,2]])\nsopt = SOPT(lattice, hilbert, (t, t1, t2), (U, U′, UmJ, J, Jp, λ), bc, ls)\n\n#define the physical observables\ns0 = Onsite(:s0, 1.0)\nsx = Onsite(:sx, 1.0+0im, couplings=Jˣ\"\")\nsy = Onsite(:sx, 1.0+0im, couplings=Jʸ\"\")\nsz = Onsite(:sx, 1.0+0im, couplings=Jᶻ\"\")\np₀ = projectstate_points(sopt)\ncoeff = Coefficience(lattice, hilbert, (s0,sx,sy,sz), p₀[1]; η=1e-10 )\n\n#obtain the exchange interactions of spin model\nbond = Bonds(lattice)[4]\nsoptmatrix = matrix(sopt,bond)\nJmat = coefficience_project(soptmatrix, coeff)\n\n#test\ntxx, tyx, tzx, txy, tyy, tzy, txz, tyz, tzz = tij[1, 1], tij[2, 1], tij[3, 1], tij[1, 2], tij[2, 2], tij[3, 2], tij[1, 3], tij[2, 3], tij[3, 3]\nlambda1 = lambda\nA = -1/3 * (Jₕ+9*lambda1+3*UU) / (6*Jₕ^2-UU*(3*lambda1+UU)+Jₕ*(4*lambda1+UU))\nη = Jₕ / (6*Jₕ^2+(3*lambda1+UU)*(3*lambda1+2*UU)-Jₕ*(17*lambda1+8*UU))\nB = 4/3 * (3*Jₕ-3*lambda1-UU) / (6*Jₕ-3*lambda1-2*UU)*η\n\nJ23 = 8*A/9*(-(txy-tyx)*(txz-tzx)-(tyz-tzy)*(txx+tyy+tzz)) + 4*B/9*(txy*(5*txz-2*tzx)+5*tyx*(tzx+2*txz)+(5*tyz+tzy)*(tyy+tzz-2*txx))\nJ11 = 4*A/9*(-(txy-tyx)^2-(txz-tzx)^2+(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txy-tyx)^2+(txz-tzx)^2-2(2tyz+tzy)*(tyz+2tzy)+2(txx+tyy-2tzz)*(txx-2tyy+tzz))\nJ33 = 4*A/9*(-(txz-tzx)^2-(txy-tyx)^2+(tyz-tzy)^2+(txx+tyy+tzz)^2)+4*B/9*((txz-tzx)^2+(tyz-tzy)^2-2(2txy+tyx)*(txy+2tyx)+2(tzz+tyy-2txx)*(txx-2tyy+tzz))\nJ12 = 8*A/9*(-(tyz-tzy)*(txz-tzx)-(txy-tyx)*(txx+tyy+tzz)) + + 4*B/9*(tzx*(5*tzy-2*tyz)+5*txz*(tyz+2*tzy)+(5*txy+tyx)*(tyy+txx-2*tzz))\n\n(Jmat[2,2] ≈ J11, Jmat[4,4] ≈ J33, Jmat[2,3] ≈ J12, Jmat[3,4] ≈ J23)","category":"page"},{"location":"examples/KitaevModel/#Construct-the-Generator-of-pseudospin-1/2","page":"Kitaev model on honeycomb lattice","title":"Construct the Generator of pseudospin-1/2","text":"","category":"section"},{"location":"examples/KitaevModel/","page":"Kitaev model on honeycomb lattice","title":"Kitaev model on honeycomb lattice","text":"#define Generator of spin-1/2\ngen = coefficience_project(sopt, coeff;η=1e-10)\n\n#latexformat of spin-1/2, add :icoord subscript.\nusing QuantumLattices: idtype, latexformat, slatex, LaTeX, expand\noptspin = gen|>expand\nT = optspin|>typeof|>eltype|>idtype|>eltype\nlatexformat(T, LaTeX{(:tag,), (:site, :icoord)}(:S, vectors=lattice.vectors))\noptspin","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = SecondOrderPerturbationTheory","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"KitaevModel.md\",\n        ]\nDepth = 2","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SecondOrderPerturbationTheory","category":"page"},{"location":"#SecondOrderPerturbationTheory","page":"Home","title":"SecondOrderPerturbationTheory","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SecondOrderPerturbationTheory.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of second order perturbation theory","category":"page"},{"location":"#Manuals","page":"Home","title":"Manuals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [SecondOrderPerturbationTheory]","category":"page"},{"location":"#SecondOrderPerturbationTheory.BinaryConfigure","page":"Home","title":"SecondOrderPerturbationTheory.BinaryConfigure","text":"BinaryConfigure{I<:TargetSpace, P<:AbstractPID} <: CompositeDict{P, I}\nBinaryConfigure(ps::Pair...)\nBinaryConfigure(kv)\n\nConstruct BinaryConfigure at a lattice. The local binary configure is given by BinaryConfigure.\n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.Coefficience","page":"Home","title":"SecondOrderPerturbationTheory.Coefficience","text":"Coefficience{P<:AbstractPID, I<:AbstractVector{<:Matrix{<:Number}}} <: Action\nCoefficience(ob::AbstractVector{<:Matrix{<:Number}}, lattice::AbstractLattice; order::Int=-1, η::Float64=1e-12, dim::Int=2)\nCoefficience(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, p₀::Dict{<:AbstractPID, <:ProjectState}; η::Float64=1e-12, order::Int=-1, dim::Int=2)\nCoefficience(observables::Dict{<:AbstractPID, <:AbstractVector{<:Matrix{<:Number}}}; η::Float64=1e-12, order::Int= -1, dim::Int=2)\n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.PickState","page":"Home","title":"SecondOrderPerturbationTheory.PickState","text":"PickState{I<:Vector{Vector{Int}}, P<:AbstractPID} <: CompositeDict{P, I}\nPickState(ps::Pair...)\nPickState(kv)\n\nConstruct  PickState. Pick low-energy states at a lattice.\n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.ProjectState","page":"Home","title":"SecondOrderPerturbationTheory.ProjectState","text":"ProjectState{V<:Real, C<:Number, P<:TargetSpace}\n\nProjected states contain eigenvalues, eigenvectors, and basis of H₀. \n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.ProjectState-Tuple{QuantumLattices.Essentials.QuantumOperators.Operators, ExactDiagonalization.BinaryBases, Any}","page":"Home","title":"SecondOrderPerturbationTheory.ProjectState","text":"ProjectState(ops::Operators, braket::BinaryBases, table; pick::Union{UnitRange{Int}, Vector{Int}, Colon}=:)\nProjectState(ops::Operators, ts::TargetSpace, table, pick::Vector{Vector{Int}})\n\nConstruct ProjectState.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.ProjectStateBond","page":"Home","title":"SecondOrderPerturbationTheory.ProjectStateBond","text":"ProjectStateBond(left::ProjectState, right::ProjectState)\n\nConstruct ProjectStateBond.\n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.ProjectStateBond-Tuple{ProjectState, ProjectState, Int64, Int64}","page":"Home","title":"SecondOrderPerturbationTheory.ProjectStateBond","text":"ProjectStateBond(left::ProjectState, right::ProjectState, nleft::Int, nright::Int)\n\nThe nleft and nright are the number of bits of left bit shift operator for left and right ProjectState, respectively.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.SOPT","page":"Home","title":"SecondOrderPerturbationTheory.SOPT","text":"SOPT{L<:AbstractLattice, G₁<:Generator, G₀<:Generator, PT<:SecondOrderPerturbation} <: Engine\n\nSecond order perturbation theory method of a electronic quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.SOPT-Tuple{QuantumLattices.Essentials.Spatials.AbstractLattice, QuantumLattices.Essentials.DegreesOfFreedom.Hilbert, Tuple{Vararg{QuantumLattices.Essentials.DegreesOfFreedom.Term}}, Tuple{Vararg{QuantumLattices.Essentials.DegreesOfFreedom.Term}}, BinaryConfigure, PickState}","page":"Home","title":"SecondOrderPerturbationTheory.SOPT","text":"SOPT(lattice::AbstractLattice, hilbert::Hilbert, terms₁::Tuple{Vararg{Term}}, terms₀::Tuple{Vararg{Term}}, binaryconfigure::BinaryConfigure, lowstate::PickState; boundary::Boundary=plain)\n\nConstruct the second order perturbation method for a quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.SOPTMatrix","page":"Home","title":"SecondOrderPerturbationTheory.SOPTMatrix","text":"SOPTMatrix(bond::Bond, P₀::ProjectStateBond, m₀::Matrix, m₂::Matrix)\n\nMatrix representation of the low-energy hamiltionian.\n\nArguments\n\n-bond: bond of lattice -P₀: projected state -m₀: matrix representation of the zeroth order of the low-energy hamiltionian -m₂: matrix representation of second order of the low-energy hamiltionian\n\n\n\n\n\n","category":"type"},{"location":"#SecondOrderPerturbationTheory.SecondOrderPerturbation","page":"Home","title":"SecondOrderPerturbationTheory.SecondOrderPerturbation","text":"SecondOrderPerturbation{B<:BinaryConfigure, L<:PickState} <: Transformation\n(::SecondOrderPerturbation)(H₁::Generator, p₀::Dict{T,<:ProjectState}, qₚ::Dict{T,<:ProjectState}, qₘ::Dict{T,<:ProjectState}, bond::Bond) where T<:AbstractPID  -> SOPTMatrix\n(::SecondOrderPerturbation)(H₀::Generator, H₁::Generator, bond::Bond) -> SOPTMatrix\n\nA type.\n\n\n\n\n\n","category":"type"},{"location":"#Base.:<<-Tuple{ExactDiagonalization.BinaryBases, Int64}","page":"Home","title":"Base.:<<","text":"Base.:(<<)(bs::BinaryBases, n::Int) -> BinaryBases\n\n\n\n\n\n","category":"method"},{"location":"#Base.:<<-Tuple{ProjectState, Int64}","page":"Home","title":"Base.:<<","text":"Base.:(<<)(ps::ProjectState, n::Int) -> ProjectState\n\nLeft bit shift opeartor. The BinaryBasis is left shifted by n bits.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Essentials.QuantumOperators.matrix-Tuple{QuantumLattices.Essentials.QuantumOperators.Operators, ExactDiagonalization.TargetSpace, ExactDiagonalization.TargetSpace, Any}","page":"Home","title":"QuantumLattices.Essentials.QuantumOperators.matrix","text":"matrix(ops::Operators, ts₁::TargetSpace, ts₂::TargetSpace, table) -> Matrix\n\nGet the matrix of direct sum of submatrices.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Essentials.QuantumOperators.matrix-Tuple{SOPT, QuantumLattices.Essentials.Spatials.Bond}","page":"Home","title":"QuantumLattices.Essentials.QuantumOperators.matrix","text":"matrix(sopt::SOPT, bond::Bond) -> SOPTMatrix\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Essentials.QuantumOperators.matrix-Tuple{SOPT}","page":"Home","title":"QuantumLattices.Essentials.QuantumOperators.matrix","text":"matrix(sopt::SOPT) -> Vector\n\nObtain SOPTMatrix on all bonds.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Interfaces.:⊕-Tuple{ProjectState, ProjectState}","page":"Home","title":"QuantumLattices.Interfaces.:⊕","text":"⊕(ps₁::ProjectState, ps₂::ProjectState) -> ProjectState\n\nGet the direct sum of two projected states.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Interfaces.:⊗-Tuple{ProjectState, ProjectState}","page":"Home","title":"QuantumLattices.Interfaces.:⊗","text":"⊗(ps₁::ProjectState, ps₂::ProjectState) -> ProjectState\n\nGet the direct product of two sets of projected states.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Interfaces.dimension-Tuple{ProjectState}","page":"Home","title":"QuantumLattices.Interfaces.dimension","text":"dimension(ps::ProjectState) -> Int\n\nThe dimension of local low-energy hilbert space.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Interfaces.dimension-Union{Tuple{T}, Tuple{ProjectState, Type{T}}} where T<:ExactDiagonalization.TargetSpace","page":"Home","title":"QuantumLattices.Interfaces.dimension","text":"dimension(ps::ProjectState,::Type{T}) where T<:TargetSpace -> Int\n\nThe dimension of target space basis.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Interfaces.expand-Tuple{QuantumLattices.Essentials.Frameworks.Generator, QuantumLattices.Essentials.Spatials.AbstractBond}","page":"Home","title":"QuantumLattices.Interfaces.expand","text":"expand(gen::Generator, bond::AbstractBond) -> Operators\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.:⊠-Tuple{ExactDiagonalization.BinaryBases, ExactDiagonalization.BinaryBases}","page":"Home","title":"SecondOrderPerturbationTheory.:⊠","text":"⊠(bs₁::BinaryBases, bs₂::BinaryBases) -> Tuple{BinaryBases, Vector{Int}}\n\nGet the direct product of two sets of binary bases, and the permutation vector.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.coefficience_project-Tuple{SOPT, Coefficience}","page":"Home","title":"SecondOrderPerturbationTheory.coefficience_project","text":"coefficience_project(sopt::SOPT, coeff::Coefficience; η::Float64=1e-14) -> Generator\n\nOnly support the pseudospin-1/2 case.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.coefficience_project-Union{Tuple{T}, Tuple{Matrix{<:Number}, AbstractVector{T}, Tuple{Int64, Int64}}} where T<:(Matrix{<:Number})","page":"Home","title":"SecondOrderPerturbationTheory.coefficience_project","text":"coefficience_project(m₂::Matrix{<:Number}, gsg::AbstractVector{T}, nshape::Tuple{Int,Int}; η::Float64=1e-12) where T<:Matrix{<:Number} -> Matrix\ncoefficience_project(m₂::Matrix{<:Number}, bond::Bond, coeff::Coefficience) -> Matrix\ncoefficience_project(soptm::SOPTMatrix, coeff::Coefficience) -> Matrix\n\nGet the coefficience of exchange interaction.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.hamiltonianeff-Tuple{ProjectState, ProjectState, QuantumLattices.Essentials.QuantumOperators.Operators, QuantumLattices.Essentials.DegreesOfFreedom.Table}","page":"Home","title":"SecondOrderPerturbationTheory.hamiltonianeff","text":"hamiltonianeff(psp::ProjectState, psq::ProjectState, h1::Operators, table::Table)  ->Tuple{Matrix,Matrix}\n\nGet the effective Hamiltonian, the first and second terms of the result correspond to the zero-th and 2nd perturbations respectively.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.high_configure-Tuple{BinaryConfigure, PickState}","page":"Home","title":"SecondOrderPerturbationTheory.high_configure","text":"high_configure(bc::BinaryConfigure, ls::PickState) -> Tuple\nhigh_configure(bc::BinaryConfigure, ls::PickState, pids::AbstractVector{PID}) -> Tuple\n\nGet the high-energy configure of local space.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.observables_project-Tuple{Tuple{Vararg{QuantumLattices.Essentials.DegreesOfFreedom.Term}}, QuantumLattices.Essentials.Spatials.Point, QuantumLattices.Essentials.DegreesOfFreedom.Hilbert, ProjectState}","page":"Home","title":"SecondOrderPerturbationTheory.observables_project","text":"observables_project(term::Term, point::Point, hilbert::Hilbert, psp::ProjectState) -> Vector\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.projectstate_points-Tuple{BinaryConfigure, PickState, QuantumLattices.Essentials.Frameworks.Generator}","page":"Home","title":"SecondOrderPerturbationTheory.projectstate_points","text":"projectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator) -> Tuple{Dict{PID,ProjectState},Dict{PID,ProjectState}, Dict{PID,ProjectState}}\nprojectstate_points(bc::BinaryConfigure, ls::PickState, H₀::Generator, points::AbstractVector{<:Point}) -> Tuple{Dict{PID,ProjectState},Dict{PID,ProjectState}, Dict{PID,ProjectState}}\n\nConstruct the ProjectState` type of low-energy states within N-particle space, high-energy states with (N+1)-particle space, and high-energy states with (N-1)-particle space.\n\n\n\n\n\n","category":"method"},{"location":"#SecondOrderPerturbationTheory.projectstate_points-Tuple{SOPT}","page":"Home","title":"SecondOrderPerturbationTheory.projectstate_points","text":"projectstate_points(sopt::SOPT) -> Tuple{Dict{PID, ProjectState},Dict{PID, ProjectState},Dict{PID, ProjectState}}\n\nConstruct ProjectState on all points.\n\n\n\n\n\n","category":"method"}]
}
