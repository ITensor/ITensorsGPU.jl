using ITensors,
      ITensors.NDTensors,
      ITensorsGPU,
      LinearAlgebra, # For tr()
      Combinatorics, # For permutations()
      CuArrays,
      Test

@testset "cuITensor $T Contractions" for T ∈ (Float64,ComplexF64)
  mi,mj,mk,ml,ma = 2,3,4,5,6,7
  i = Index(mi,"i")
  j = Index(mj,"j")
  k = Index(mk,"k")
  l = Index(ml,"l")
  a = Index(ma,"a") 
  @testset "Test contract cuITensors" begin
      Aij = cuITensor(randomITensor(T,i,j))
      Aji = cuITensor(randomITensor(T,j,i))
      Bij = cuITensor(randomITensor(T,i,j))
      Aik = cuITensor(randomITensor(T,i,k))
      Ajk = cuITensor(randomITensor(T,j,k))
      Ajl = cuITensor(randomITensor(T,j,l))
      Akl = cuITensor(randomITensor(T,k,l))
      Aijk = cuITensor(randomITensor(T,i,j,k))
      Ajkl = cuITensor(randomITensor(T,j,k,l))
      Aikl = cuITensor(randomITensor(T,i,k,l))
      Akla = cuITensor(randomITensor(T,k,l,a))
      Aijkl = cuITensor(randomITensor(T,i,j,k,l))
      Dv = rand(mi)
      D = itensor(ITensors.tensor(NDTensors.Diag(CuVector(Dv)), IndexSet(i, i')))
      Ev = rand(mi)
      E = itensor(ITensors.tensor(NDTensors.Diag(CuVector(Ev)), IndexSet(i, i'')))
    @testset "Test contract cuITensors (Matrix*Diag -> Matrix)" begin
      C = Aij*D
      @test collect(CuArray(C))≈collect(CuMatrix(Aij, j, i))*diagm(0=>Dv)
    end
    @testset "Test contract cuDiagITensors (Diag*Diag -> Diag)" begin
      C = E*D
      @test collect(CuArray(C))≈diagm(0=>Ev)*diagm(0=>Dv)
    end
    @testset "Test contract cuDiagITensors (UniformDiag*Diag -> Diag)" begin
      scal = itensor(ITensors.tensor(NDTensors.Diag(2.0), IndexSet(i, i'')))
      C = scal*D
      @test collect(CuArray(C))≈2.0 .* diagm(0=>Dv)
      C = D*scal
      @test collect(CuArray(C))≈2.0 .* diagm(0=>Dv)
    end
  end # End contraction testset
end
