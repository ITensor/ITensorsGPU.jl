using ITensors,
      ITensorsGPU,
      Test

@testset "CuMPO Basics" begin
  N = 6
  sites = [Index(2,"Site") for n=1:N]
  @test length(cuMPO()) == 0
  O = cuMPO(sites)
  @test length(O) == N

  str = split(sprint(show, O), '\n')
  @test str[1] == "MPO"
  @test length(str) == length(O) + 2

  O[1] = cuITensor(sites[1], prime(sites[1]))
  @test hasindex(O[1],sites[1])
  @test hasindex(O[1],prime(sites[1]))
  P = copy(O)
  @test hasindex(P[1],sites[1])
  @test hasindex(P[1],prime(sites[1]))

  @testset "orthogonalize" begin
    phi = randomCuMPS(sites)
    K = randomCuMPO(sites)
    orthogonalize!(phi, 1)
    orthogonalize!(K, 1)
    orig_inner = inner(phi, K, phi)
    orthogonalize!(phi, div(N, 2))
    orthogonalize!(K, div(N, 2))
    @test inner(phi, K, phi) ≈ orig_inner
  end

  @testset "inner <y|A|x>" begin
    phi = randomCuMPS(sites)
    K = randomCuMPO(sites)
    @test maxLinkDim(K) == 1
    psi = randomCuMPS(sites)
    phidag = dag(phi)
    prime!(phidag)
    phiKpsi = phidag[1]*K[1]*psi[1]
    for j = 2:N
      phiKpsi *= phidag[j]*K[j]*psi[j]
    end
    @test phiKpsi[] ≈ inner(phi,K,psi)

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomCuMPS(badsites)
    @test_throws DimensionMismatch inner(phi,K,badpsi)
    
    # make bigger random MPO...
    for link_dim in 2:5
        mpo_tensors  = ITensor[cuITensor() for ii in 1:N]
        mps_tensors  = ITensor[cuITensor() for ii in 1:N]
        mps_tensors2 = ITensor[cuITensor() for ii in 1:N]
        mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mpo_tensors[1] = randomCuITensor(mpo_link_inds[1], sites[1], sites[1]') 
        mps_tensors[1] = randomCuITensor(mps_link_inds[1], sites[1]) 
        mps_tensors2[1] = randomCuITensor(mps_link_inds[1], sites[1]) 
        for ii in 2:N-1
            mpo_tensors[ii] = randomCuITensor(mpo_link_inds[ii], mpo_link_inds[ii-1], sites[ii], sites[ii]') 
            mps_tensors[ii] = randomCuITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
            mps_tensors2[ii] = randomCuITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
        end
        mpo_tensors[N] = randomCuITensor(mpo_link_inds[N-1], sites[N], sites[N]')
        mps_tensors[N] = randomCuITensor(mps_link_inds[N-1], sites[N])
        mps_tensors2[N] = randomCuITensor(mps_link_inds[N-1], sites[N])
        K   = MPO(N, mpo_tensors, 0, N+1)
        psi = MPS(N, mps_tensors, 0, N+1)
        phi = MPS(N, mps_tensors2, 0, N+1)
        orthogonalize!(psi, 1; maxdim=link_dim)
        orthogonalize!(K,   1; maxdim=link_dim)
        orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
        phidag = dag(phi)
        prime!(phidag)
        phiKpsi = phidag[1]*K[1]*psi[1]
        for j = 2:N
          phiKpsi *= phidag[j]*K[j]*psi[j]
        end
        @test scalar(phiKpsi) ≈ inner(phi,K,psi)
    end
  end

  @testset "applyMPO" begin
    phi = randomCuMPS(sites)
    K   = randomCuMPO(sites)
    @test maxLinkDim(K) == 1
    psi = randomCuMPS(sites)
    psi_out = applyMPO(K, psi,maxdim=1)
    @test inner(phi,psi_out) ≈ inner(phi,K,psi)
    @test_throws ArgumentError applyMPO(K, psi, method="fakemethod")

    badsites = [Index(2,"Site") for n=1:N+1]
    badpsi = randomCuMPS(badsites)
    @test_throws DimensionMismatch applyMPO(K,badpsi)

    # make bigger random MPO...
    for link_dim in 2:5
        mpo_tensors  = ITensor[ITensor() for ii in 1:N]
        mps_tensors  = ITensor[ITensor() for ii in 1:N]
        mps_tensors2 = ITensor[ITensor() for ii in 1:N]
        mpo_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mps_link_inds = [Index(link_dim, "r$ii,Link") for ii in 1:N-1]
        mpo_tensors[1] = randomCuITensor(mpo_link_inds[1], sites[1], sites[1]') 
        mps_tensors[1] = randomCuITensor(mps_link_inds[1], sites[1]) 
        mps_tensors2[1] = randomCuITensor(mps_link_inds[1], sites[1]) 
        for ii in 2:N-1
            mpo_tensors[ii] = randomCuITensor(mpo_link_inds[ii], mpo_link_inds[ii-1], sites[ii], sites[ii]') 
            mps_tensors[ii] = randomCuITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
            mps_tensors2[ii] = randomCuITensor(mps_link_inds[ii], mps_link_inds[ii-1], sites[ii]) 
        end
        mpo_tensors[N] = randomCuITensor(mpo_link_inds[N-1], sites[N], sites[N]')
        mps_tensors[N] = randomCuITensor(mps_link_inds[N-1], sites[N])
        mps_tensors2[N] = randomCuITensor(mps_link_inds[N-1], sites[N])
        K   = MPO(N, mpo_tensors, 0, N+1)
        psi = MPS(N, mps_tensors, 0, N+1)
        phi = MPS(N, mps_tensors2, 0, N+1)
        orthogonalize!(psi, 1; maxdim=link_dim)
        orthogonalize!(K,   1; maxdim=link_dim)
        orthogonalize!(phi, 1; normalize=true, maxdim=link_dim)
        psi_out = applyMPO(deepcopy(K), deepcopy(psi); maxdim=10*link_dim, cutoff=0.0)
        @test inner(phi, psi_out) ≈ inner(phi, K, psi)
    end
  end
  @testset "add" begin
    shsites = spinHalfSites(N)
    K = randomCuMPO(shsites)
    L = randomCuMPO(shsites)
    M = sum(K, L)
    @test length(M) == N
    psi = randomCuMPS(shsites)
    k_psi = applyMPO(K, psi, maxdim=1)
    l_psi = applyMPO(L, psi, maxdim=1)
    @test inner(psi, sum(k_psi, l_psi)) ≈ inner(psi, M, psi) atol=5e-3
  end

  @testset "nmultMPO" begin
    psi = randomCuMPS(sites)
    K   = randomCuMPO(sites)
    L   = randomCuMPO(sites)
    @test maxLinkDim(K) == 1
    @test maxLinkDim(L) == 1
    KL = nmultMPO(K, L, maxdim=1)
    psi_kl_out = applyMPO(K, applyMPO(L, psi, maxdim=1), maxdim=1)
    @test inner(psi, KL, psi) ≈ inner(psi, psi_kl_out) atol=5e-3

    badsites = [Index(2,"Site") for n=1:N+1]
    badL = randomMPO(badsites)
    @test_throws DimensionMismatch nmultMPO(K,badL)
  end
end
