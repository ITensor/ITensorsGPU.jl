using ITensors,
      ITensorsGPU,
      LinearAlgebra, # For tr()
      CuArrays,
      Test

      # gpu tests!
@testset "cutrunctate" begin
    @test ITensorsGPU.truncate!(CuArrays.zeros(Float64, 10)) == (0., 0., CuArrays.zeros(Float64, 1))
    trunc = ITensorsGPU.truncate!(CuArray([1.0, 0.5, 0.1, 0.05]); absoluteCutoff=true, cutoff=0.2)
    @test trunc[1] ≈ 0.15
    @test trunc[2] ≈ 0.3
    @test trunc[3] == CuArray([1.0, 0.5])
    trunc = ITensorsGPU.truncate!(CuArray([0.5, 0.4, 0.1]); relativeCutoff=true, cutoff=0.2)
    @test trunc[1] ≈ 0.1
    @test trunc[2] ≈ 0.45
    @test trunc[3] == CuArray([0.5])
end # End truncate test
