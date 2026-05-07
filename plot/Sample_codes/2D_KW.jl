using TensorKit, MPSKit, MPSKitModels, ProgressMeter, Plots, JLD2

ZZ = σᶻᶻ()
X = σˣ()
Z = σᶻ()

function to_index(x,y,W)
    return y+W*(x-1)
end

function to_cartesian(n,W)
    x = (n+W-1)÷W
    y = mod1(n%W,W)
    return [x,y]
end

function next_y(n,W)
    x,y = to_cartesian(n,W)
    ynew = mod1(y+1,W)
    return to_index(x,ynew,W)
end

function before_y(n,W)
    x,y = to_cartesian(n,W)
    ynew = mod1(y-1,W)
    return to_index(x,ynew,W)
end

# 2d version
function gaussian_wave_packet(x, x0, sigma, k,W)
    return exp(-(norm(x .- x0)^2) / (2 * (sigma)^2)) * exp(1im * k * (x[1]))
end

function make_wavepacket(ψ,x0,W; k=0.3π, σ=1)
    psi_temp = []
    mid = W÷2+1
    len_psi = length(ψ)
    lat = FiniteChain(len_psi)
    O = @mpoham sum(gaussian_wave_packet([x,y], x0, σ, k,W)*Z{lat[to_index(x,y,W)]} 
        for x in (x0[1]-2):(x0[1]+2) for y in (x0[2]-1):(x0[2]+1))
        # for x in (x0[1]-2):(x0[1]+2) for y in (x0[2]-1):(x0[2]+1))
    psi = deepcopy(ψ)
    psi,_ = approximate(psi, (O, ψ),DMRG(;verbosity=0))
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

ZZ = σᶻᶻ()
X = σˣ()
XXXX = σˣˣ()⊗σˣˣ()
ZZZ = σᶻᶻ()⊗σᶻ()

function make_TFIsing(W,L,g)
    Ntot = L*W
    lat = FiniteChain(Ntot)

    HL = @mpoham (sum(-g*X{lat[i]} for i = 1:Ntot) 
    + sum(-ZZ{lat[i],lat[i+W]} for i=1:(Ntot-W))
    + sum(-ZZ{lat[i],lat[next_y(i,W)]} for i = 1:Ntot)
    )
    return HL
end

function make_half_toric(W,L1,g; L2=L1)
    Lx = L1 + 1 + 2*L2+1
    lat = FiniteChain(Lx*W)
    NL = W*L1

    HL = @mpoham (sum(-g*X{lat[i]} for i = 1:NL) 
    + sum(-ZZ{lat[i],lat[i+W]} for i=1:(NL-W))
    + sum(-ZZ{lat[i],lat[next_y(i,W)]} for i = 1:NL)
    )

    HR = @mpoham (sum(-g*XXXX{lat[to_index(L1+2j,y,W)],lat[to_index(L1+2j+2,y,W)],lat[to_index(L1+2j+1,y,W)],lat[before_y(to_index(L1+2j+1,y,W),W)]} for j=1:L2 for y=1:W)
    + sum(-Z{lat[to_index(L1+2j,y,W)]} for j=2:L2 for y=1:W)
    + sum(-Z{lat[to_index(L1+2j+1,y,W)]} for j=1:L2 for y=1:W)
    )

    Hint = @mpoham sum(-ZZZ{lat[to_index(L1,y,W)],lat[to_index(L1+1,y,W)],lat[to_index(L1+2,y,W)]} for y=1:W)
    return HL+HR+Hint
end    