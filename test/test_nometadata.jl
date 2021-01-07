using MetaDataArrays, LinearAlgebra, CUDA, Flux, Test

# Initialize
A = NoMetaDataArray(randn(ComplexF32, 5, 5))
B = NoMetaDataArray(randn(ComplexF32, 5, 5))

# Operations
A+B
A-B
A.*B
A./B
-A
randn(Float32)*A
A*randn(Float32)
randn(Float32).*A
A.*randn(Float32)
A./randn(Float32)
randn(Float32)./A
conj(A)
dot(A,B)
norm(A,1)
sum(A; dims=2)
fill!(A, 1f0)
similar(A)

# Gpu
A = A |> gpu
B = B |> gpu
A+B
A-B
A.*B
A./B
-A
randn(Float32)*A
A*randn(Float32)
randn(Float32).*A
A.*randn(Float32)
A./randn(Float32)
randn(Float32)./A
conj(A)
dot(A,B)
norm(A,1)
sum(A; dims=2)
fill!(A, 1f0)
similar(A)

# Cpu
A = A |> cpu