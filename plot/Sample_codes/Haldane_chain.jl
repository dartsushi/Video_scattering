using MPSKit, TensorKit, MPSKitModels, Plots, ProgressMeter, LaTeXStrings, Test

# Spin 1 Pauli operators
I = id(ℂ^3)
XX = S_xx(spin=1)
YY = S_yy(spin=1)
ZZ = S_zz(spin=1);
X = S_x(spin=1)
Y = S_y(spin=1)
Z = S_z(spin=1)

X_half = to_spin1(σˣ())
Y_half = to_spin1(σʸ())
Z_half = to_spin1(σᶻ())

# Operators 
O_KT = exp(1im*π*(Z⊗X))
expZ = exp(1im*π*Z)
expX = exp(1im*π*X)
Y1 = exp(1im*π*X)*Y
Y2 = Y*exp(1im*π*Z)

# MPO for KT transformation
util = Tensor(ones,ComplexF64,ℂ^1)
@tensor OI[-1 -2;-3 -4] := I[-2;-3] *util[-1] * util'[-4]
@tensor OX[-1 -2;-3 -4] := X[-2;-3] *util[-1] * util'[-4]
@tensor OY[-1 -2;-3 -4] := Y[-2;-3] *util[-1] * util'[-4]
@tensor OZ[-1 -2;-3 -4] := Z[-2;-3] *util[-1] * util'[-4]
@tensor OZexpZ[-1 -2;-3 -4] := (Z*expZ)[-2;-3] *util[-1] * util'[-4]
@tensor OexpZ[-1 -2;-3 -4] := expZ[-2;-3] *util[-1] * util'[-4];

function to_spin1(op)
    box = zeros(ComplexF64,3,3)
    box[[1,3],[1,3]] = op[]
    return TensorMap(box,ℂ^3←ℂ^3)
end


function find_gs_haldane(L,χ;verbosity=1,hz = 1e-4)
    lat = FiniteChain(L)
    h = XX+YY+ZZ
    H = @mpoham sum(h{i,j} for (i,j) in nearest_neighbours(lat))
    Hz = @mpoham -hz*Z{lat[1]}
    H = H+Hz
    ψ = FiniteMPS(L,ℂ^3, ℂ^χ)
    ψ, _ = find_groundstate(ψ,H;verbosity)
    return ψ, H
end

function make_HLR(NL,NR;hz = 1e-4)
    hL = XX+YY+ZZ
    hR = -XX+Y1⊗Y2-ZZ
    hint = X⊗Z_half⊗X  + 1im*Y⊗Y_half⊗Y2 - Z⊗X_half⊗Z;
    
    lat = FiniteChain(NL+1+NR)
    HL = @mpoham sum(hL{lat[i],lat[i+1]} for i=1:NL-1)-hz*Z{lat[1]}
    HR = @mpoham sum(hR{lat[i],lat[i+1]} for i=NL+2:(NL+NR))
    Hint = @mpoham hint{lat[NL],lat[NL+1],lat[NL+2]}
    H = HL+HR+Hint
    return H
end

function find_gs_HLR(NL,NR,χ;verbosity=1,hz = 1e-4)
    L = NL + NR + 1
    lat = FiniteChain(L)
    ψ = FiniteMPS(L,ℂ^3,ℂ^χ)
    H = make_HLR(NL,NR;hz)
    ψ, _ = find_groundstate(ψ,H;verbosity=1)
    return ψ, H
end

function O_profile(ψ,O)
    Xs = []
    lat = FiniteChain(length(ψ))
    X_exp(x) = @mpoham O{lat[x]}
    for i in 1:length(ψ)
        push!(Xs, real(expectation_value(ψ,X_exp(i))))
    end
    return Xs
end

function OO_profile(ψ,OO)
    Xs = []
    lat = FiniteChain(length(ψ))
    X_exp(x) = @mpoham OO{lat[x],lat[x+1]}
    for i in 1:length(ψ)-1
        push!(Xs, real(expectation_value(ψ,X_exp(i))))
    end
    return Xs
end

function string_op(ψ,x)
    Os = vcat(fill(OexpZ,x),fill(OI,length(ψ)-x))
    Os[x] = OZ
    op = FiniteMPO(Os)
    return -real(expectation_value(ψ,op))
end


function plot_spin(ψ)
    Zs = O_profile(ψ,Z)
    Xs = O_profile(ψ,X)
    Z_string = [string_op(ψ,i) for i=1:length(ψ)]
    plot(Zs,label="Z"),
    plot!(Xs,label="X")
    plot!(Z_string,label="Zstring")
    hline!([0],linestyle=:dash,label="")
end


function O_domain(lx,ly;O=OX)
    Os = vcat(fill(OI,lx),fill(OX,ly))
    return FiniteMPO(Os)
end

function gaussian_wave_packet(x, x0, sigma, k)
    return exp(-((x - x0)^2) / (2 * sigma^2)) * exp(1im * k * x)
end

function make_wavepacket(ψ,x0; k=-0.5π, σ=1.5, amplitude=1,operator=X)
    psi_temp = []
    len_psi = length(ψ)
    lat = FiniteChain(len_psi)
    O = @mpoham sum(amplitude*gaussian_wave_packet(i,x0,σ, k)*operator{lat[i]} for i in 1:len_psi)
    # O = @mpoham sum(amplitude*gaussian_wave_packet(i,x0,σ, k)*operator{lat[i]} for i in x0-3:x0+3)
    psi = deepcopy(ψ)
    psi,_ = approximate(psi, (O, ψ),DMRG(;verbosity=0))
    return normalize!(psi)
end

function make_domainwall(ψ,x0; k=-0.5π, σ=1.5, amplitude=1)
    psi_temp = []
    len_psi = length(ψ)
    lat = FiniteChain(len_psi)
    O = sum(amplitude*gaussian_wave_packet(i,x0,σ, k)*O_domain(i,len_psi-i) for i in x0-3:x0+3)
    psi = deepcopy(ψ)
    psi,_ = approximate(psi, (O, ψ),DMRG(;verbosity=0))
    return normalize!(psi)
end


hz = 1e-5
χ = 48
NL,NR = 30,30

ψ_mix,H_mix = find_gs_HLR(NL,NR,χ;hz)
psi = make_domainwall(ψ_mix,NL+15;k=-0.7π);
Ψ = copy(psi);

χ = 40
δt = 0.05 
step = 600
L = NL+NR+1
Xs_list = zeros(step,L)
Zs_list = zeros(step,L)
Z_string_list = zeros(step,L)

@showprogress for i=1:step
    Ψ, _ = timestep(Ψ, H_mix, 0, δt, TDVP2(; trscheme = truncbelow(10^(-8)) & truncdim(χ)))
    Zs_list[i,:] = O_profile(Ψ,Z)
    Xs_list[i,:] = O_profile(Ψ,X)
    Z_string_list[i,:] = [string_op(Ψ,i) for i=1:length(Ψ)]
end