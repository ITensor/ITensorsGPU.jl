# ITensorsGPU: Intelligent Tensors with GPU acceleration


[![codecov](https://codecov.io/gh/ITensor/ITensorsGPU.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ITensor/ITensorsGPU.jl)

[![gitlab-ci](https://gitlab.com/JuliaGPU/ITensorsGPU-jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/ITensorsGPU-jl/commits/master)

This package is meant to extend the functionality of [ITensors.jl](https://github.com/ITensor/ITensors.jl) to make use of CUDA-enabled GPUs in a way that's simple enough that any user of ITensors.jl can take advantage of. It sits on top of the wonderful [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl) package and uses NVIDIA's CUTENSOR library for high-performance tensor operations. It includes a GPU-enabled implementation of the [DMRG-PEPS algorithm](https://arxiv.org/abs/1908.08833).

## Getting ITensorsGPU.jl and PEPS up and running

What you'll need:
  - Julia 1.x -- I use 1.4-dev but anything 1.1 or after should work
  - the [ITensors.jl](https://github.com/ITensor/ITensors.jl) library: `add ITensors` to install
  - CUDA 10.1
  - CUTENSOR v1.0.0 -- `libcutensor.so` needs to be on your `LD_LIBRARY_PATH` so that `CuArrays.jl` will be able to find it.
  - A copy of this repo that is available to Julia. `ITensorsGPU.jl` is presently *not* registered in the main Julia package registry. The easiest way to acquire it is to do an `add` or `dev` using its URL:
  
    ```
    julia> ]
    
    pkg> add https://github.com/ITensor/ITensorsGPU.jl.git#master
    
    pkg> dev https://github.com/ITensor/ITensorsGPU.jl.git
    ```
    For a bit more explanation of what's going on here, check out the [Julia Pkg docs](https://docs.julialang.org/en/v1/stdlib/Pkg/).
  - You need a specific version of some dependencies, because updates to `CuArrays.jl` and friends aren't playing nicely with `ITensorsGPU.jl` yet:
  
    ```
    julia> cd("YOUR_PATH_HERE/ITensorsGPU.jl")

    julia> ]

    pkg> activate .
    ...

    ITensors> add CuArrays#4e73de2f1

    ITensors> build
    ```

To check if this has all worked, you can run the package tests using:

    ```
    julia> cd("YOUR_PATH_HERE/ITensorsGPU.jl")

    julia> ]

    pkg> activate .
    
    pkg> test
    ```

Scripts included:
- `prof/`: Really basic time profiling of algorithms that exist in "basic" `ITensors.jl`: 1-D and 2-D DMRG 
- `src/peps/`: Implementation of the PEPS algorithm in [this article](https://arxiv.org/abs/1908.08833), along with profiling and timing info.

Probably the most interesting file in `peps/` is `full_peps_run.jl`. This runs identical (or, extremely similar) simulations on the CPU (using BLAS) and GPU (using CUTENSOR) and does time profiling of them.
Because of the way Julia compilation works, only the "main" simulation is timed (this is the vast majority of the walltime) -- for more information on this, see [here](https://docs.julialang.org/en/v1/manual/profile/).
There are two command line arguments you need to provide: the system size `L`, which creates an `L x L` lattice, and `chi`, which controls the internal matrix size. `chi` has the name "bond dimension" in tensor network
algorithms, but essentially it sets the size of the `chi x chi x chi x chi x chi` tensors that make up the PEPS. Right now the C++ code handles `chi` in the range `[3 .. 7]`. You might see OOM issues if you try to run
a big lattice with big `chi` on the GPU, even one with a lot of memory -- `CuArrays.jl` and the Julia GC are working "as intended" but maybe not as we might like, see [here](https://github.com/JuliaGPU/CuArrays.jl/issues/323) for more.

An example of running this file:

`julia-1.4 full_peps_run.jl 4 3`  

Here, `L=4` and `chi=3`. If you find this annoying it's easy to hardcode these variables in the script itself, or load them from JSON files.

Right now, the code is single GPU and single node only.
I've checked that the CPU run (the first one in the file) involves no GPU activity (besides phoning it up to say "hi, I'm here!") using `nvprof`. There is a bit of transfer from the host to device
in the GPU code, but last time I profiled the code with `nvprof` and Julia's inbuilt profiler it was not contributing significantly to runtime -- if you see something different, let me know, as I'm
pretty sure I know where most of it's coming from and it could be fixed.

The code outputs to `STDOUT` the energy at each sweep and, at the end of run, the total walltime needed to do ten sweeps. We found in the Julia and C++ simulations that 10 sweeps was realistic to get the
energy to converge, and for larger systems with larger `chi` you will be waiting a *long* time for the CPU to finish (it's a big reason we're so excited about the GPU results!).

All the simulations I've conducted were on the rusty cluster at the Simons Foundation. The nodes I ran the code on have 32GB V100 GPUs with 36 core Skylake CPUs (I can get more detailed information if it would be helpful).

To use `nvprof` on the PEPS code, you can run `prof_run.jl` (and mess around with parameters in that file):

`nvprof julia-1.4 -e 'using Pkg; Pkg.activate("."); include("prof_run.jl")'`

# DMRG

The DMRG code can be found in the `prof/` folder. These two example DMRG codes were modified from CPU only ones in `benchmark/`. Examples of how to run them are in the `runner` scripts. For example, here is the output of the `2ddmrg_gpu_runner`
on our cluster:

```
Activating environment at `~/projects/ITensors/Project.toml`
After sweep 1 energy=-36.122895185750 maxLinkDim=4 time=8.241
After sweep 2 energy=-41.011003343895 maxLinkDim=16 time=2.779
After sweep 3 energy=-44.048843621536 maxLinkDim=64 time=17.389
After sweep 4 energy=-44.484327192948 maxLinkDim=256 time=190.736
After sweep 5 energy=-44.559161373677 maxLinkDim=400 time=764.369
After sweep 6 energy=-44.563240537317 maxLinkDim=500 time=1144.768
Time to do DMRG on CPU: 2147.086442509
After sweep 1 energy=-40.552049646455 maxLinkDim=10 time=40.198
After sweep 2 energy=-44.208348798560 maxLinkDim=100 time=8.371
After sweep 3 energy=-44.484373071219 maxLinkDim=200 time=20.851
After sweep 4 energy=-44.549936218533 maxLinkDim=300 time=62.029
After sweep 5 energy=-44.560654498554 maxLinkDim=400 time=116.575
After sweep 6 energy=-44.563248632726 maxLinkDim=500 time=182.301
Time to do DMRG on GPU: 434.903995371
```

We can see that the GPU gets good energies *and* is much faster. This `maxLinkDim` is another name for `chi` from the PEPS code.
