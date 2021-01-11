#: Abstract type: AbstractMetaDataArray


export AbstractMetaDataArray


# Abstract type

"""
AbstractMetaDataArray is simply a wrapper for arrays + metadata.
Expected behavior for AbstractMetaDataArray:
    - raw_data(x::AbstractMetaDataArray{ADT,MDT,T,N})::ADT<:AbstractArray{T,N} -> Return raw data
    - meta_data(x::AbstractMetaDataArray{ADT,MDT,T,N})::MDT -> Return meta data
    - join_metadata(meta_data1::MDT, meta_data2::MDT)::MDT
    - a constructor CMDT(raw_data::ADT, meta_data::MDT)::CMDT<:AbstractMetaDataArray{MDT,T,N}
"""
abstract type AbstractMetaDataArray{ADT,MDT,T<:Number,N}<:AbstractArray{T,N} end


# Size/dimension

Base.size(x::AbstractMetaDataArray) = size(raw_data(x))
Base.ndims(x::AbstractMetaDataArray) = ndims(raw_data(x))


# Copying

Base.copy!(x::TF, y::TF) where {TF<:AbstractMetaDataArray} = (copy!(meta_data(x), meta_data(y)); copy!(raw_data(x), raw_data(y)); return x)
Base.copy(x::TF) where {TF<:AbstractMetaDataArray} = TF(copy(raw_data(x)), copy(meta_data(x)))


# Indexing

Base.IndexStyle(::Type{<:AbstractMetaDataArray}) = IndexLinear()
Base.getindex(x::AbstractMetaDataArray, i::Int64) = getindex(raw_data(x), i)
Base.setindex!(x::AbstractMetaDataArray{ADT,MDT,T,N}, val::T, i::Int64) where {ADT,MDT,T,N} = (setindex!(raw_data(x), val, i); return x)


# Show

Base.show(io::IO, x::AbstractMetaDataArray) = show(io, raw_data(x))
Base.show(io::IO, mime::MIME"text/plain", x::AbstractMetaDataArray) = show(io, mime, raw_data(x))


# Linear algebra

## x1+x2
+(x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = TF(raw_data(x1)+raw_data(x2), join_metadata(meta_data(x1), meta_data(x2)))

## x1.+c, c.+x1
Base.broadcasted(::typeof(+), x::TF, c::Number) where {TF<:AbstractMetaDataArray} = TF(raw_data(x).+c, meta_data(x))
Base.broadcasted(::typeof(+), c::Number, x::AbstractMetaDataArray) = x.+c

## x1-x2
-(x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = TF(raw_data(x1)-raw_data(x2), join_metadata(meta_data(x1), meta_data(x2)))

## -x
-(x::TF) where {TF<:AbstractMetaDataArray} = TF(-raw_data(x), meta_data(x))

## x.-c, c.-x
Base.broadcasted(::typeof(-), x::TF, c::Number) where {TF<:AbstractMetaDataArray} = TF(raw_data(x).-c, meta_data(x))
Base.broadcasted(::typeof(-), c::Number, x::AbstractMetaDataArray) = x.-c

## x1.*x2
Base.broadcasted(::typeof(*), x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = TF(raw_data(x1).*raw_data(x2), join_metadata(meta_data(x1), meta_data(x2)))

## c*x, x*c
*(c::Number, x::TF) where {TF<:AbstractMetaDataArray} = TF(c*raw_data(x), meta_data(x))
*(x::AbstractMetaDataArray, c::Number) = c*x
LinearAlgebra.lmul!(c::Number, x::AbstractMetaDataArray) = (lmul!(c, raw_data(x)); return x)
LinearAlgebra.rmul!(x::AbstractMetaDataArray, c::Number) = (rmul!(raw_data(x), c); return x)

## c.*x, x.*c
Base.broadcasted(::typeof(*), c::Number, x::TF) where {TF<:AbstractMetaDataArray} = TF(c.*raw_data(x), meta_data(x))
Base.broadcasted(::typeof(*), x::AbstractMetaDataArray, c::Number) = c.*x

## x1./x2
Base.broadcasted(::typeof(/), x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = TF(raw_data(x1)./raw_data(x2), join_metadata(meta_data(x1), meta_data(x2)))

## x/c
/(x::TF, c::Number) where {TF<:AbstractMetaDataArray} = TF(raw_data(x)/c, meta_data(x))
LinearAlgebra.ldiv!(c::Number, x::AbstractMetaDataArray) = (ldiv!(c, raw_data(x)); return x)
LinearAlgebra.rdiv!(x::AbstractMetaDataArray, c::Number) = (rdiv!(raw_data(x), c); return x)

## x./c, c./x
Base.broadcasted(::typeof(/), x::TF, c::Number) where {TF<:AbstractMetaDataArray} = TF(raw_data(x)./c, meta_data(x))
Base.broadcasted(::typeof(/), c::Number, x::TF) where {TF<:AbstractMetaDataArray} = TF(c./raw_data(x), meta_data(x))

## conj(x)
Base.conj(x::TF) where {TF<:AbstractMetaDataArray} = TF(conj(raw_data(x)), meta_data(x))

## dot(x1,x2)
LinearAlgebra.dot(x1::TF, x2::TF) where {TF<:AbstractMetaDataArray} = dot(raw_data(x1), raw_data(x2))

## norm(x)
LinearAlgebra.norm(x::AbstractMetaDataArray, p::Real) = norm(raw_data(x), p)

## abs.(x)
Base.broadcasted(::typeof(abs), x::TF) where {TF<:AbstractMetaDataArray} = TF(abs.(raw_data(x)), meta_data(x))

## fill!
Base.fill!(x::AbstractMetaDataArray, c::Number) = (fill!(raw_data(x), c); return x)

## sum(x)
Base.sum(x::TF; dims) where {TF<:AbstractMetaDataArray} = TF(sum(raw_data(x); dims=dims), meta_data(x))


# Utils

function Base.isapprox(x::TF, y::TF; rtol::Real=sqrt(eps()), atol::Real=0) where {TF<:AbstractMetaDataArray}
    join_metadata(meta_data(x),meta_data(y))
    return isapprox(raw_data(x), raw_data(y); rtol=rtol, atol=atol)
end

Base.similar(x::TF) where {TF<:AbstractMetaDataArray} = TF(similar(raw_data(x)), meta_data(x))


# CUDA utils

Flux.gpu(x::TF) where {TF<:AbstractMetaDataArray{<:AbstractArray,Nothing,<:Number,<:Any}} = TF(gpu(raw_data(x)), nothing)
Flux.cpu(x::TF) where {TF<:AbstractMetaDataArray{<:AbstractArray,Nothing,<:Number,<:Any}} = TF(cpu(raw_data(x)), nothing)