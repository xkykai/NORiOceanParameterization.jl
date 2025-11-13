module Implementation

export 
    NORiBaseVerticalDiffusivity, NORiNNFluxClosure,
    NORiClosureTuple, NORiBaseClosureOnly, NORiClosureWithNN

include("NORiBaseVerticalDiffusivity.jl")
include("NORiNNFluxClosure.jl")
include("closure_builder.jl")

end # module
