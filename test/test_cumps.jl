using ITensors,
      ITensorsGPU,
      Test

@testset "cuMPS Basics" begin

  N = 10
  sites = [Index(2,"Site") for n=1:N]
  psi = cuMPS(sites)
  @test length(psi) == N
  @test length(cuMPS()) == 0

  str = split(sprint(show, psi), '\n')
  @test str[1] == "MPS"
  @test length(str) == length(psi) + 2

  @test siteindex(psi,2) == sites[2]
  @test hasindex(psi[3],linkindex(psi,2))
  @test hasindex(psi[3],linkindex(psi,3))

  psi[1] = cuITensor(sites[1])
  @test hasindex(psi[1],sites[1])

  @testset "cuproductMPS" begin
    @testset "vector of string input" begin
      sites = spinHalfSites(N)
      state = fill("",N)
      for j=1:N
        state[j] = isodd(j) ? "Up" : "Dn"
      end
      psi  = productCuMPS(sites,state)
      for j=1:N
        sign = isodd(j) ? +1.0 : -1.0
        ops = cuITensor(op(sites,"Sz",j))
        psip = prime(psi[j], "Site")
        res = psi[j]*ops*dag(psip)
        @test res[] ≈ sign/2
      end
      @test_throws DimensionMismatch productCuMPS(sites, fill("", N - 1))
    end

    @testset "vector of int input" begin
      sites = spinHalfSites(N)
      state = fill(0,N)
      for j=1:N
        state[j] = isodd(j) ? 1 : 2
      end
      psi = productCuMPS(sites,state)
      for j=1:N
        sign = isodd(j) ? +1.0 : -1.0
        ops = cuITensor(op(sites,"Sz",j))
        psip = prime(psi[j], "Site")
        @test (psi[j]*ops*dag(psip))[] ≈ sign/2
      end
    end

  end

  @testset "randomMPS" begin
    phi = randomCuMPS(sites)
    @test hasindex(phi[1],sites[1])
    @test norm(phi[1])≈1.0
    @test hasindex(phi[4],sites[4])
    @test norm(phi[4])≈1.0
  end

  @testset "inner different MPS" begin
    phi = randomCuMPS(sites)
    psi = randomCuMPS(sites)
    phipsi = dag(phi[1])*psi[1]
    for j = 2:N
      phipsi *= dag(phi[j])*psi[j]
    end
    @test phipsi[] ≈ inner(phi,psi)
 
    badsites = [Index(2) for n=1:N+1]
    badpsi = randomCuMPS(badsites)
    @test_throws DimensionMismatch inner(phi,badpsi)
  end

  @testset "inner same MPS" begin
    psi = randomCuMPS(sites)
    psidag = dag(psi)
    primelinks!(psidag)
    psipsi = psidag[1]*psi[1]
    for j = 2:N
      psipsi *= psidag[j]*psi[j]
    end
    @test psipsi[] ≈ inner(psi,psi)
  end

  @testset "add MPS" begin
    psi    = randomCuMPS(sites)
    phi    = similar(psi)
    phi.A_ = deepcopy(psi.A_)
    xi     = sum(psi, phi)
    @test inner(xi, xi) ≈ 4.0 * inner(psi, psi) 
  end

  sites = spinHalfSites(N)
  psi = cuMPS(sites)
  @test length(psi) == N # just make sure this works
  @test length(siteinds(psi)) == N

  psi = randomCuMPS(sites)
  orthogonalize!(psi, N-1)
  @test ITensors.leftLim(psi) == N-2
  @test ITensors.rightLim(psi) == N
  orthogonalize!(psi, 2)
  @test ITensors.leftLim(psi) == 1
  @test ITensors.rightLim(psi) == 3
  psi = randomCuMPS(sites)
  psi.rlim_ = N+1 # do this to test qr from rightmost tensor
  orthogonalize!(psi, div(N, 2))
  @test ITensors.leftLim(psi) == div(N, 2) - 1
  @test ITensors.rightLim(psi) == div(N, 2) + 1

  @test_throws ErrorException linkindex(MPS(N, fill(cuITensor(), N), 0, N + 1), 1)

  @testset "replaceBond!" begin
  # make sure factorization preserves the bond index tags
    psi = randomCuMPS(sites)
    phi = psi[1]*psi[2]
    bondindtags = tags(linkindex(psi,1))
    replaceBond!(psi,1,phi)
    @test tags(linkindex(psi,1)) == bondindtags

    # check that replaceBond! updates llim_ and rlim_ properly
    orthogonalize!(psi,5)
    phi = psi[5]*psi[6]
    replaceBond!(psi,5,phi, dir="fromleft")
    @test ITensors.leftLim(psi)==5
    @test ITensors.rightLim(psi)==7

    phi = psi[5]*psi[6]
    replaceBond!(psi,5,phi,dir="fromright")
    @test ITensors.leftLim(psi)==4
    @test ITensors.rightLim(psi)==6

    psi.llim_ = 3
    psi.rlim_ = 7
    phi = psi[5]*psi[6]
    replaceBond!(psi,5,phi,dir="fromleft")
    @test ITensors.leftLim(psi)==3
    @test ITensors.rightLim(psi)==7
  end

end

# Helper function for making MPS
function basicRandomCuMPS(N::Int;dim=4)
  sites  = [Index(2,"Site") for n=1:N]
  M      = MPS(sites)
  links  = [Index(dim,"n=$(n-1),Link") for n=1:N+1]
  M[1]   = randomCuITensor(sites[1],links[2])
  for n=2:N-1
    M[n] = randomCuITensor(links[n],sites[n],links[n+1])
  end
  M[N]   = randomCuITensor(links[N],sites[N])
  M[1]  /= sqrt(inner(M,M))
  return M
end

@testset "MPS gauging and truncation" begin

  N = 30

  @testset "orthogonalize! method" begin
    c = 12
    M = basicRandomCuMPS(N)
    orthogonalize!(M,c)

    @test leftLim(M) == c-1
    @test rightLim(M) == c+1

    # Test for left-orthogonality
    L = M[1]*prime(M[1],"Link")
    l = linkindex(M,1)
    @test collect(L) ≈ delta(l,l') rtol=1E-12
    for j=2:c-1
      L = L*M[j]*prime(M[j],"Link")
      l = linkindex(M,j)
      @test collect(L) ≈ delta(l,l') rtol=1E-12
    end

    # Test for right-orthogonality
    R = M[N]*prime(M[N],"Link")
    r = linkindex(M,N-1)
    @test collect(R) ≈ delta(r,r') rtol=1E-12
    for j in reverse(c+1:N-1)
      R = R*M[j]*prime(M[j],"Link")
      r = linkindex(M,j-1)
      @test collect(R) ≈ delta(r,r') rtol=1E-12
    end

    @test norm(M[c]) ≈ 1.0
  end

  @testset "truncate! method" begin
    M  = basicRandomCuMPS(N;dim=10)
    M0 = copy(M)
    truncate!(M;maxdim=5)
    @test rightLim(M) == 2
    # Test for right-orthogonality
    R = M[N]*prime(M[N],"Link")
    r = linkindex(M,N-1)
    @test collect(R) ≈ delta(r,r') rtol=1E-12
    for j in reverse(2:N-1)
      R = R*M[j]*prime(M[j],"Link")
      r = linkindex(M,j-1)
      @test collect(R) ≈ delta(r,r') rtol=1E-12
    end
    @test inner(M0, M) > 0.1
  end


end

#=@testset "Other MPS methods" begin

  @testset "sample! method" begin
    N = 10
    sites = [Index(3,"Site,n=$n") for n=1:N]
    psi = makeRandomCuMPS(sites,chi=3)
    nrm2 = inner(psi,psi)
    psi[1] *= (1.0/sqrt(nrm2))

    s = sample!(psi)

    @test length(s) == N
    for n=1:N
      @test 1 <= s[n] <= 3
    end

    # Throws becase not orthogonalized to site 1:
    orthogonalize!(psi,3)
    @test_throws ErrorException sample(psi)

    # Throws becase not normalized
    orthogonalize!(psi,1)
    psi[1] *= (5.0/norm(psi[1]))
    @test_throws ErrorException sample(psi)

    # Works when ortho & normalized:
    orthogonalize!(psi,1)
    psi[1] *= (1.0/norm(psi[1]))
    s = sample(psi)
    @test length(s) == N
  end

end=#
