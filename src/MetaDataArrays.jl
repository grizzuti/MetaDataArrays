module MetaDataArrays

using LinearAlgebra, CUDA, Flux
import Base: +,-,*,/

include("abstracttype.jl")
include("nometadata_type.jl")

end
