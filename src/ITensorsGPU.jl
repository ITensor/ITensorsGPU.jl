module ITensorsGPU

using CuArrays
using CuArrays.CUTENSOR
using CuArrays.CUBLAS
using CuArrays.CUSOLVER
using LinearAlgebra
using Random
using TimerOutputs
using StaticArrays
using ITensors
using CUDAdrv
using CUDAdrv.Mem
import CuArrays: CuArray, CuMatrix, CuVector
import ITensors: randn!, compute_contraction_labels,
                 plussers, DenseTensor, eigenHermitian,
                 TensorStorage, similar_type,
                 scale!, getperm, unioninds, array, matrix, vector,
                 polar, tensors, truncate!
import ITensors.Tensors: ContractionProperties, contract!!, _contract!!, _contract!, contract!, contract,
                         contraction_output, UniformDiagTensor, CombinerTensor, contraction_output_type,
                         UniformDiag, Diag, DiagTensor, NonuniformDiag, NonuniformDiagTensor, zero_contraction_output,
                         outer!, outer!!, is_trivial_permutation, ind, permutedims!!
import Base.*
include("tensor/cudense.jl")
include("tensor/culinearalgebra.jl")
include("tensor/cutruncate.jl")
include("tensor/cucombiner.jl")
include("tensor/cudiag.jl")
include("cuitensor.jl")
include("mps/cumps.jl")
include("mps/cumpo.jl")

export cuITensor,
       randomCuITensor,
       cuMPS,
       randomCuMPS,
       productCuMPS,
       randomCuMPO,
       cuMPO

end #module
