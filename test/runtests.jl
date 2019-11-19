using ITensorsGPU, Test, CuArrays

@testset "ITensorsGPU.jl" begin
    @testset "$filename" for filename in (
        "test_cuitensor.jl",
        "test_cucontract.jl",
        "test_cumpo.jl",
        "test_cumps.jl",
        "test_cuiterativesolvers.jl"
    )
      println("Running $filename")
      include(filename)
    end
end
