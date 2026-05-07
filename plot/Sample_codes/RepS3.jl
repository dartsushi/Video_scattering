using MPSKit, TensorKit, MPSKitModels, Plots, ProgressMeter 

function O_profile(ψ,O)
    Xs = []
    Xs_h = []
    lat = FiniteChain(length(ψ))
    X_exp(x) = @mpoham O{lat[x]}
    for i in 1:length(ψ)
        push!(Xs, real(expectation_value(ψ,X_exp(i))))
    end
    return Xs
end
function simulate_wavepacket(L,g1,g2;χ = 24,δt = 0.007, step = 600)
    X = σˣ()
    Z = σᶻ()
    id_c = σᶻ()^0
    Y = σʸ()
    ## Make Hamiltonian 
    lat = FiniteChain(2L)
    H1 = @mpoham sum( (X ⊗ (id_c + Z ) ⊗ X ){lat[i-1],lat[i],lat[i+1]} for i=2:L-2)  - sum(g1*Z{lat[i]} for i=1:L-1)
    H2 =  @mpoham sum( (X ⊗ (id_c + Z ) ⊗ X ){lat[i-1],lat[i],lat[i+1]} for i=L+2:2L-1)  - sum(g2*Z{lat[i]} for i=L+1:2L)
    H12 = @mpoham  (0.5*(X ⊗ ( Z + id_c ) ⊗ (X + sqrt(3)* Y ) ⊗ X)){lat[L-2],lat[L-1],lat[L],lat[L+1]} +  (0.5*(X ⊗ (X - sqrt(3)* Y ) ⊗ ( Z + id_c )  ⊗ X)){lat[L-1],lat[L],lat[L+1],lat[L+2]}
    H = H1+H2+H12

    ### Make WavePacket
    ψ = FiniteMPS(2L,ℂ^2,ℂ^χ)
    ψ,_ = find_groundstate(ψ,H;verbosity=0);
    Xs_list = zeros(step+1,2L)
    Zs_list = zeros(step+1,2L)
    Zs_list[1,:] = O_profile(ψ,Z)
    Xs_list[1,:] = O_profile(ψ,X)
    x0 = L-10
    k=-0.7π
    σ=1.5
    O = @mpoham sum(2*exp(-((i - x0)^2) / (2 * σ^2)) * exp(1im * k * i)*(X){lat[i]} for i in 1:2L)
    psi = deepcopy(ψ)
    psi,_ = approximate(psi, (O, ψ),DMRG(;verbosity=0))
    ψ = normalize!(psi)


    ### Run Simulation
    energies = zeros(step,2L)
    SA_list = []
    @showprogress for i=1:step
        ψ, _ = timestep(ψ, H, 0, δt, TDVP2(; trscheme = truncbelow(10^(-8))& truncdim(χ)))
        Zs_list[i,:] = O_profile(ψ,Z)
        Xs_list[i,:] = O_profile(ψ,X)
        push!(SA_list,[entropy(ψ,i) for i=1:length(ψ)])
        E = []
         for i in 1:2L
         if i == 2L-1
            push!(E,+1/3*expectation_value(ψ,(2L-2,2L-1,2L) => (X ⊗ (id_c + Z ) ⊗ X))+1/3*expectation_value(ψ,(2L-3,2L-2,2L-1) => (X ⊗ (id_c + Z ) ⊗ X))-g1 *expectation_value(ψ,i => Z) )
            elseif i == 2L
                 push!(E,+1/3*expectation_value(ψ,(2L-2,2L-1,2L) => (X ⊗ (id_c + Z ) ⊗ X))-g1 *expectation_value(ψ,i => Z) )
            elseif i== 1
            push!(E,+1/3*expectation_value(ψ,(1,2,3) => (X ⊗ (id_c + Z ) ⊗ X))-g1 *expectation_value(ψ,i => Z) )
              elseif i== 2
                push!(E,+1/3*expectation_value(ψ,(1,2,3) => (X ⊗ (id_c + Z ) ⊗ X))+1/3*expectation_value(ψ,(2,3,4) => (X ⊗ (id_c + Z ) ⊗ X))-g1 *expectation_value(ψ,i => Z) )
            elseif i ==L-2
            push!(E,+1/3*expectation_value(ψ,(i-2,i-1,i) => (Z ⊗ (id_c + X) ⊗ Z))-g1*expectation_value(ψ,i => Z)+1/3*expectation_value(ψ,(i-1,i,i+1) => (Z ⊗ (id_c + X) ⊗ Z))  +1/4*0.5*expectation_value(ψ,(i,i+1,i+2,i+3) => (X ⊗ ( Z + id_c ) ⊗ (X + sqrt(3)* Y ) ⊗ X)))
            elseif i ==L-1
            push!(E,+1/3*expectation_value(ψ,(i-2,i-1,i) => (Z ⊗ (id_c + X) ⊗ Z))-g1*expectation_value(ψ,i => Z) +1/4*0.5*expectation_value(ψ,(i-1,i,i+1,i+2) => (X ⊗ ( Z + id_c ) ⊗ (X + sqrt(3)* Y ) ⊗ X))+1/4*0.5*expectation_value(ψ,(i,i+1,i+2,i+3) => (X ⊗ (X - sqrt(3)* Y ) ⊗ ( Z + id_c )  ⊗ X)))
            elseif i == L
            push!(E,+1/3*expectation_value(ψ,(i,i+1,i+2) => (Z ⊗ (id_c + X) ⊗ Z))+1/4*0.5*expectation_value(ψ,(i-2,i-1,i,i+1) => (X ⊗ ( Z + id_c ) ⊗ (X + sqrt(3)* Y ) ⊗ X))+1/4*0.5*expectation_value(ψ,(i-1,i,i+1,i+2) => (X ⊗ (X - sqrt(3)* Y ) ⊗ ( Z + id_c )  ⊗ X)))
             elseif i ==L+1
            push!(E,1/3*expectation_value(ψ,(i,i+1,i+2) => (Z ⊗ (id_c + X) ⊗ Z))+1/4*0.5*expectation_value(ψ,(i-3,i-2,i-1,i) => (X ⊗ ( Z + id_c ) ⊗ (X + sqrt(3)* Y ) ⊗ X))+1/4*0.5*expectation_value(ψ,(i-2,i-1,i,i+1) => (X ⊗ (X - sqrt(3)* Y ) ⊗ ( Z + id_c )  ⊗ X))-g2 *expectation_value(ψ,i => Z))
             elseif i ==L+2
            push!(E,+1/3*expectation_value(ψ,(i-1,i,i+1) => (Z ⊗ (id_c + X) ⊗ Z))+1/3*expectation_value(ψ,(i,i+1,i+2) => (Z ⊗ (id_c + X) ⊗ Z))+1/4*0.5*expectation_value(ψ,(i-3,i-2,i-1,i) => (X ⊗ (X - sqrt(3)* Y ) ⊗ ( Z + id_c )  ⊗ X))-g2 *expectation_value(ψ,i => Z) )
             elseif i < L 
            push!(E,-g1 *expectation_value(ψ,i => Z) + 1/3*expectation_value(ψ,(i-1,i,i+1) => (X ⊗ (id_c + Z ) ⊗ X)) + 1/3*expectation_value(ψ,(i-2,i-1,i) => (X ⊗ (id_c + Z ) ⊗ X)) + 1/3*expectation_value(ψ,(i,i+1,i+2) => (X ⊗ (id_c + Z ) ⊗ X)))
             else  
            push!(E,-g2 *expectation_value(ψ,i => Z) + 1/3*expectation_value(ψ,(i-1,i,i+1) => (X ⊗ (id_c + Z ) ⊗ X)) + 1/3*expectation_value(ψ,(i-2,i-1,i) => (X ⊗ (id_c + Z ) ⊗ X)) + 1/3*expectation_value(ψ,(i,i+1,i+2) => (X ⊗ (id_c + Z ) ⊗ X)))
            end
        end
        energies[i,:] = real(E)
    end
    return Xs_list,Zs_list,energies,SA_list
end

