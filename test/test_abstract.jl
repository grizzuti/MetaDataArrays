using MetaDataArrays, LinearAlgebra, Test

# Custom concrete type
module CustomMetaDataArray

    using MetaDataArrays
    import MetaDataArrays: raw_data, meta_data, join_metadata, metadata_array
    export NamedArray

    struct NamedArray{T,N} <: AbstractMetaDataArray{String,T,N}
        name::String
        array::Array{T,N}
    end
    raw_data(A::NamedArray) = A.array
    meta_data(A::NamedArray) = A.name
    join_metadata(A::NamedArray, B::NamedArray) = string(A.name, B.name)
    metadata_array(array::Array{T,N}, name::String) where {T,N} = NamedArray{T,N}(name, array)

end
using .CustomMetaDataArray

# Initialize
A = NamedArray{ComplexF32,2}("A", randn(ComplexF32, 5, 5))
B = NamedArray{ComplexF32,2}("B", randn(ComplexF32, 5, 5))

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