using MetaDataArrays, LinearAlgebra, Test

# Custom concrete type
module CustomMetaDataArray

    using MetaDataArrays
    import MetaDataArrays: raw_data, meta_data, join_metadata
    export NamedArray

    struct NamedArray{T,N} <: AbstractMetaDataArray{Array{T,N},String,T,N}
        array::Array{T,N}
        name::Array{String,1}
    end
    raw_data(A::NamedArray) = A.array
    meta_data(A::NamedArray) = A.name
    join_metadata(s1::Array{String,1}, s2::Array{String,1}) = [s1; s2]

end
using .CustomMetaDataArray

# Initialize
A = NamedArray{ComplexF32,2}(randn(ComplexF32, 5, 5), ["A"])
B = NamedArray{ComplexF32,2}(randn(ComplexF32, 5, 5), ["B"])

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
A .= B
A .= randn(Float32)