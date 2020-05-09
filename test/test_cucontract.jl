using ITensors,
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
      A = cuITensor(randomITensor(T))
      B = cuITensor(randomITensor(T))
      Ai = cuITensor(randomITensor(T,i))
      Bi = cuITensor(randomITensor(T,i))
      Aj = cuITensor(randomITensor(T,j))
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
    @testset "Test contract cuITensor (Scalar*Scalar -> Scalar)" begin
      C = A*B
      @test scalar(C)≈scalar(A)*scalar(B)
    end
    @testset "Test contract cuITensor (Scalar*Vector -> Vector)" begin
      C = A*Ai
      @test collect(C)≈scalar(A)*collect(Ai)
    end
    @testset "Test contract cuITensor (Vector*Scalar -> Vector)" begin
      C = Aj*A
      @test collect(C)≈scalar(A)*collect(Aj)
    end
    @testset "Test contract cuITensors (Vectorᵀ*Vector -> Scalar)" begin
      C = Ai*Bi
      Ccollect = collect(Ai) * collect(Bi) 
      @test scalar(Ccollect)≈scalar(C)
    end
    @testset "Test contract cuITensors (Vector*Vectorᵀ -> Matrix)" begin
      C = Ai*Aj
      Ccollect = collect(Ai)*collect(Aj)
      @test Ccollect≈collect(permute(C,i,j))
    end
    @testset "Test contract cuITensors (Matrix*Scalar -> Matrix)" begin
      Aij = permute(Aij,i,j)
      C = Aij*A
      @test collect(permute(C,i,j))≈scalar(A)*collect(Aij)
    end
    @testset "Test contract cuITensors (Matrixᵀ*Vector -> Vector)" begin
      cAij = permute(copy(Aij),j,i)
      Ccollect = collect(Aij)*collect(Aj)
      C = cAij*Aj
      @test Ccollect ≈ collect(C)
    end
    @testset "Test contract cuITensors (Matrix*Vector -> Vector)" begin
      cpAij = permute(copy(Aij),i,j)
      Ccollect = collect(cpAij)*collect(Aj)
      C   = copy(cpAij)*copy(Aj)
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Vector*Matrix -> Vector)" begin
      Aij = permute(Aij,i,j)
      C = Ai*Aij
      Ccollect = collect(Ai) * collect(Aij) 
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Vector*Matrixᵀ -> Vector)" begin
      C = Ai*Aji
      Ccollect = collect(Ai) * collect(Aji) 
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Matrix*Matrix -> Scalar)" begin
      Aij = permute(Aij,i,j)
      Bij = permute(Bij,i,j)
      C = Aij*Bij
      Ccollect = collect(Aij) * collect(Bij) 
      @test scalar(Ccollect)≈scalar(C)
    end
    @testset "Test contract cuITensors (Matrix*Matrix -> Matrix)" begin
      Aij = permute(Aij,i,j)
      Ajk = permute(Ajk,j,k)
      C = Aij*Ajk
      Ccollect = collect(Aij)*collect(Ajk)
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Matrixᵀ*Matrix -> Matrix)" begin
      Aij = permute(Aij,j,i)
      Ajk = permute(Ajk,j,k)
      C = Aij*Ajk
      Ccollect = collect(Aij) * collect(Ajk) 
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Matrix*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij,i,j)
      Ajk = permute(Ajk,k,j)
      C = Aij*Ajk
      Ccollect = collect(Aij) * collect(Ajk) 
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Matrixᵀ*Matrixᵀ -> Matrix)" begin
      Aij = permute(Aij,j,i)
      Ajk = permute(Ajk,k,j)
      C = Aij*Ajk
      Ccollect = collect(Aij) * collect(Ajk) 
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (3-Tensor*Scalar -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aijk*A
      @test collect(permute(C,i,j,k))≈scalar(A)*collect(Aijk)
    end
    @testset "Test contract cuITensors (3-Tensor*Vector -> Matrix)" begin
      cAijk = permute(copy(Aijk),i,j,k)
      C = cAijk*Ai
      Ccollect = collect(cAijk) * collect(Ai) 
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (Vector*3-Tensor -> Matrix)" begin
      Aijk = permute(Aijk,i,j,k)
      C = Aj*Aijk
      Ccollect = collect(Aj)*collect(Aijk)
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (3-Tensor*Matrix -> Vector)" begin
      Aijk = permute(Aijk,i,j,k)
      Aik = permute(Aik,i,k)
      C = Aijk*Aik
      Ccollect = collect(Aijk)*collect(Aik)
      @test Ccollect≈collect(C)
    end
    @testset "Test contract cuITensors (3-Tensor*Matrix -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Ajl = permute(Ajl,j,l)
      C = Aijk*Ajl
      Ccollect = collect(Aijk)*collect(Ajl)
      @test Ccollect≈collect(permute(C,i,k,l))
    end
    @testset "Test contract cuITensors (Matrix*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Akl  = permute(Akl,k,l)
      C = Akl*Aijk
      Ccollect = collect(Aijk)*collect(Akl)
      @test Ccollect≈collect(permute(C,l,i,j))
    end
    @testset "Test contract cuITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      Aijk = permute(Aijk,i,j,k)
      Ajkl = permute(Ajkl,j,k,l)
      C = Aijk*Ajkl
      Ccollect = collect(Aijk)*collect(Ajkl)
      @test Ccollect≈collect(permute(C,i,l))
    end
    @testset "Test contract cuITensors (3-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijk ∈ permutations([i,j,k]), inds_jkl ∈ permutations([j,k,l])
        Aijk = permute(Aijk,inds_ijk...)
        Ajkl = permute(Ajkl,inds_jkl...)
        C = Ajkl*Aijk
        Ccollect = collect(Ajkl) * collect(Aijk)
        @test Ccollect≈collect(C)
      end
    end
    @testset "Test contract cuITensors (4-Tensor*3-Tensor -> 3-Tensor)" begin
      for inds_ijkl ∈ permutations([i,j,k,l]), inds_kla ∈ permutations([k,l,a])
        Aijkl = permute(Aijkl,inds_ijkl...)
        Akla = permute(Akla,inds_kla...)
        C = Akla*Aijkl
        Ccollect = collect(Akla) * collect(Aijkl) 
        @test Ccollect≈collect(C)
      end
    end
    @testset "Test contract cuITensors (4-Tensor*3-Tensor -> 1-Tensor)" begin
      for inds_ijkl ∈ permutations([i,j,k,l]), inds_jkl ∈ permutations([j,k,l])
        Aijkl = permute(Aijkl,inds_ijkl...)
        Ajkl = permute(Ajkl,inds_jkl...) 
        C = Ajkl*Aijkl
        Ccollect = collect(Ajkl)*collect(Aijkl) 
        @test Ccollect≈collect(C)
      end
    end
  end # End contraction testset
end
