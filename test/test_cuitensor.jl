using ITensors,
      ITensorsGPU,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      CuArrays,
      Test

      # gpu tests!
@testset "cuITensor, Dense{$SType} storage" for SType ∈ (Float64,)#,ComplexF64)
  mi,mj,mk,ml,ma = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  a = Index(ma,"a") 
  @testset "Constructor" begin
      A = cuITensor(one(SType),i,j,k)
      @test collect(CuArray(A, i, j, k)) == ones(SType, dim(i), dim(j), dim(k))
      A = randomCuITensor(IndexSet(i,j,k))
      @test inds(A) == IndexSet(i, j, k)
      @test ITensorsGPU.store(A) isa ITensorsGPU.CuDense
      Aarr = rand(SType, dim(i)*dim(j)*dim(k))
      @test collect(ITensor(Aarr, IndexSet(i, j, k))) == collect(cuITensor(Aarr, IndexSet(i, j, k))) 
  end
  @testset "Test permute(cuITensor,Index...)" begin
    CA = randomCuITensor(SType,i,k,j)
    permCA = permute(CA,k,j,i)
    permA = collect(permCA)
    @test k==inds(permA)[1]
    @test j==inds(permA)[2]
    @test i==inds(permA)[3]
    A = collect(CA)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[k(kk),i(ii),j(jj)]==permA[i(ii),j(jj),k(kk)]
    end
  end
  #=@testset "Set and get values with Ints" begin
    A = ITensor(SType,i,j,k)
    A = permute(A,k,i,j)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      A[kk,ii,jj] = digits(SType,ii,jj,kk)
    end
    CA = cuITensor(A)
    CA = permute(CA,i,j,k)
    A = collect(CA)
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test A[ii,jj,kk]==digits(SType,ii,jj,kk)
    end
  end=#
  #=@testset "Test scalar(cuITensor)" begin
    x = SType(34)
    A = randomCuITensor(a)
    @test x==scalar(A)
  end=#
  @testset "Test CuVector(cuITensor)" begin
      v = CuVector(ones(Float64, dim(a)))
      A = cuITensor(v, a)
      @test v==CuVector(A)
  end
  @testset "Test CuMatrix(cuITensor)" begin
      v = CuMatrix(ones(Float64, dim(a), dim(l)))
      A = cuITensor(vec(v), a, l)
      @test v==CuMatrix(A, a, l)
      A = cuITensor(vec(v), a, l)
      @test v==CuMatrix(A)
      A = cuITensor(vec(v), a, l)
      @test v==CuArray(A, a, l)
      @test v==CuArray(A)
  end
  @testset "Test norm(cuITensor)" begin
    A = randomCuITensor(SType,i,j,k)
    B = dag(A)*A
    @test norm(A)≈sqrt(scalar(B))
  end
  @testset "Test complex(cuITensor)" begin
    A  = randomCuITensor(SType,i,j,k)
    cA = complex(A)
    @test complex.(CuArray(A)) == CuArray(cA)
  end
  #@testset "Test exp(cuITensor)" begin
  #  A  = randomCuITensor(SType,i,i')
  #  @test CuArray(exp(A,i,i')) ≈ exp(CuArray(A))
  #end
  @testset "Test add cuITensors" begin
    dA = randomCuITensor(SType,i,j,k)
    dB = randomCuITensor(SType,k,i,j)
    A = collect(dA)
    B = collect(dB)
    C = collect(dA+dB)
    @test CuArray(permute(C,i,j,k))==CuArray(permute(A,i,j,k))+CuArray(permute(B,i,j,k))
    for ii ∈ 1:dim(i), jj ∈ 1:dim(j), kk ∈ 1:dim(k)
      @test C[i(ii),j(jj),k(kk)]==A[j(jj),i(ii),k(kk)]+B[i(ii),k(kk),j(jj)]
    end
  end 

  @testset "Test factorizations of a cuITensor" begin

    A = randomCuITensor(SType,i,j,k,l)

    @testset "Test SVD of a cuITensor" begin
      U,S,V = svd(A,(j,l))
      u = commonind(U,S)
      v = commonind(S,V)
      @test collect(A)≈collect(U*S*V)
      @test collect(U*dag(prime(U,u)))≈δ(SType,u,u') rtol=1e-14
      @test collect(V*dag(prime(V,v)))≈δ(SType,v,v') rtol=1e-14
    end

    #=@testset "Test SVD truncation" begin 
        M = randn(4,4)
        (U,s,V) = svd(M)
        ii = Index(4)
        jj = Index(4)
        S = Diagonal(s)
        T = cuITensor(vec(CuArray(U*S*V')),IndexSet(ii,jj))
        (U,S,V) = svd(T,ii;maxm=2)
        @test norm(U*S*V-T)≈sqrt(s[3]^2+s[4]^2)
    end=#

    @testset "Test QR decomposition of a cuITensor" begin
      Q,R = qr(A,(i,l))
      q = commonind(Q,R)
      @test collect(A)≈collect(Q*R)
      @test collect(Q*dag(prime(Q,q)))≈δ(SType,q,q') atol=1e-14
    end

    @testset "Test polar decomposition of a cuITensor" begin
      U,P = polar(A,(k,l))
      @test collect(A)≈collect(U*P)
    end

  end # End ITensor factorization testset
end # End Dense storage test
