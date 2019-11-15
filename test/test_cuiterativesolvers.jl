using ITensorsGPU
using Test

# Wrap an ITensor with pairs of primed and
# unprimed indices to pass to davidson
struct ITensorMap
  A::ITensor
end
Base.eltype(M::ITensorMap)  = eltype(M.A)
Base.size(M::ITensorMap)    = dim(findinds(M.A,("",0)))
(M::ITensorMap)(v::ITensor) = noprime(M.A*v)

@testset "Complex davidson" begin
  d = 10
  i = Index(d,"i")
  A = cuITensor(randomITensor(Complex,i,prime(i)))
  A = mapprime(A*mapprime(dag(A),0,2),2,1)
  M = ITensorMap(A)
    
  v = randomCuITensor(i)
  λ,v = davidson(M,v;maxiter=10)
  @test M(v)≈λ*v
    
  v = cuITensor(randomITensor(Complex, i))
  λ,v = davidson(M,v;maxiter=10)
  @test M(v)≈λ*v
    
end

