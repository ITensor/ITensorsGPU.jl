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
using ITensors: ContractionProperties, Atrans, Btrans, Ctrans, truncate!
import CuArrays: CuArray, CuMatrix, CuVector
import ITensors: randn!, contract!!, _contract!!, _contract!, contract,
                 compute_contraction_labels, contract_inds, truncate!,
                 plussers, DenseTensor, eigenHermitian, permutedims!!,
                 TensorStorage, contraction_output, similar_type,
                 is_trivial_permutation, scale!, contraction_output_type,
                 DiagTensor, getperm, unioninds, array, matrix, vector,
                 polar, tensors, zero_contraction_output, UniformDiagTensor,
                 UniformDiag, Diag, NonuniformDiag, NonuniformDiagTensor,
                 CombinerTensor, outer!, outer!!

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
