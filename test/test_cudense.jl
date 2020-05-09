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
  @testset "Test add CuDense" begin
    A  = [1.0 for ii in 1:dim(i), jj in 1:dim(j)]
    dA = ITensorsGPU.CuDense{SType, CuVector{SType}}(1.0, dim(i)*dim(j))
    B  = [2.0 for ii in 1:dim(i), jj in 1:dim(j)]
    dB = ITensorsGPU.CuDense{SType, CuVector{SType}}(2.0, dim(i)*dim(j))
    dC = +(dA, IndexSet(i, j), dB, IndexSet(j, i))
    hC = collect(dC)
    @test collect(A + B) ≈ hC
  end 
end # End Dense storage test
