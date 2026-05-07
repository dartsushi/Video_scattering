using MPSKit, TensorKit, MPSKitModels, Plots, ProgressMeter, LaTeXStrings, JLD2

X = σˣ()
XX = σˣˣ()
Z = σᶻ()
ZZ = σᶻᶻ();


function gaussian_wave_packet(x, x0, sigma, k)
    return exp(-((x - x0)^2) / (2 * sigma^2)) * exp(1im * k * x)
end

function make_wavepacket(ψ,x0; k=0.7π, σ=2, amplitude=2)
    psi_temp = []
    len_psi = length(ψ)
    lat = FiniteChain(len_psi)
    O = @mpoham sum(amplitude*gaussian_wave_packet(i,x0,σ, k)*X{lat[i]} for i in 1:len_psi)
    psi = deepcopy(ψ)
    psi,_ = approximate(psi, (O, ψ),DMRG(;verbosity=0))
    normalize!(ψ)
    return normalize!(psi)
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

function simulate_wavepacket(L,g1,g2;χ = 12,δt = 0.05, step = 200, boundary_field = true ,hz = 0,pos=L-5,σ=1.5,return_Zoriginal=false,k=0.7π)
    lat = FiniteChain(2L+1)
    H1 = @mpoham sum(-XX{lat[i],lat[i+1]} for i=1:L-1) + sum(-g1*Z{lat[i]} for i=1:L)
    # magnetic field can be applied on the edge to fix the direction.s
    H2 = @mpoham sum(-g2*ZZ{lat[i],lat[i+1]} for i=L+2:2L) + sum(-X{lat[i]} for i=L+2:2L) - hz*Z{lat[2L+1]}
    H12 = @mpoham -XX{lat[L],lat[L+1]}-g2*ZZ{lat[L+1],lat[L+2]}
    H = H1+H2+H12

    ψ = FiniteMPS(2L+1,ℂ^2,ℂ^χ)
    ψ,_ = find_groundstate(ψ,H;verbosity=0);
    Z_original =  O_profile(ψ,Z)
    ψ = make_wavepacket(ψ,pos;k, σ, amplitude=1);

    Xs_list = zeros(step,2L+1)
    Zs_list = zeros(step,2L+1)
    SA_list = []
    ES_list = []
    @showprogress for i=1:step
        ψ, _ = timestep(ψ, H, 0, δt, TDVP2(; trscheme = truncbelow(10^(-8)) & truncdim(χ)))
        Zs_list[i,:] = O_profile(ψ,Z)
        Xs_list[i,:] = O_profile(ψ,X)
        push!(SA_list,[entropy(ψ,i) for i=1:length(ψ)])
        push!(ES_list,[entanglement_spectrum(ψ,i) for i=1:length(ψ)])
    end
    if return_Zoriginal
        return Z_original,Xs_list,Zs_list,SA_list,ES_list
    else
        return Xs_list,Zs_list,SA_list,ES_list
    end
end