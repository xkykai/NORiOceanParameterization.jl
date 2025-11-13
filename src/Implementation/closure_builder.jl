"""
    NORiClosureTuple(args...; kwargs...)

Helper function to build a properly ordered tuple of closures for Oceananigans.

When using `NORiNNFluxClosure`, it must be accompanied by `NORiBaseVerticalDiffusivity`,
and the base closure must come first in the tuple. This function ensures correct ordering
and allows users to add additional closures.

# Arguments
- `args...`: Closures to include. Can be:
  - `NORiBaseVerticalDiffusivity` instance
  - `NORiNNFluxClosure` instance
  - Any other Oceananigans closure instances
  - Keywords `:base` or `:nn` to create default instances

# Keyword Arguments
- `arch`: Architecture for NN closure (default: `CPU()`). Used when creating default NN closure.
- `nn_model_path`: Optional path to NN model file (default: `nothing`)
- Any other kwargs are passed to `NORiBaseVerticalDiffusivity` constructor if creating default base closure

# Returns
A tuple of closures in the correct order:
1. `NORiBaseVerticalDiffusivity` (if present)
2. `NORiNNFluxClosure` (if present) - **must be immediately after base**
3. Other closures (in order provided)

# Examples

```julia
# Create both closures with defaults
closures = NORiClosureTuple(:base, :nn)

# Create with custom architecture
closures = NORiClosureTuple(:base, :nn, arch=GPU())

# Create base with custom parameters and default NN
closures = NORiClosureTuple(:base, :nn, νˢʰ=0.1, Riᶜ=0.5)

# Use existing closure instances
base = NORiBaseVerticalDiffusivity(νˢʰ=0.1)
nn = NORiNNFluxClosure(CPU())
closures = NORiClosureTuple(base, nn)

# Add additional closures (they go between base and nn)
using Oceananigans.TurbulenceClosures
base = NORiBaseVerticalDiffusivity()
nn = NORiNNFluxClosure(CPU())
scalar_diff = ScalarDiffusivity(ν=1e-4, κ=1e-5)
closures = NORiClosureTuple(base, scalar_diff, nn)
# Result order: (base, nn, scalar_diff)

# Just base closure (no NN)
closures = NORiClosureTuple(:base)

# Error: NN without base will throw informative error
closures = NORiClosureTuple(:nn)  # Error!
```

# Notes
- If `NORiNNFluxClosure` is provided without `NORiBaseVerticalDiffusivity`, an error is thrown
- The function automatically handles ordering, so you don't need to worry about the order of arguments
- Additional closures are placed **after** base and NN closures (positions 3+)
"""
function NORiClosureTuple(args...; arch=CPU(), nn_model_path=nothing, kwargs...)
    base_closure = nothing
    nn_closure = nothing
    other_closures = []

    # Parse arguments
    for arg in args
        if arg === :base
            # Create default base closure
            base_closure = NORiBaseVerticalDiffusivity(; kwargs...)
        elseif arg === :nn
            # Create default NN closure
            nn_closure = NORiNNFluxClosure(arch; model_path=nn_model_path)
        elseif isa(arg, NORiBaseVerticalDiffusivity)
            base_closure = arg
        elseif isa(arg, NORiNNFluxClosure)
            nn_closure = arg
        else
            # Any other closure type
            push!(other_closures, arg)
        end
    end

    # Validation: NN requires base
    if !isnothing(nn_closure) && isnothing(base_closure)
        error("""
        NORiNNFluxClosure requires NORiBaseVerticalDiffusivity!

        The NN closure needs the Richardson number field computed by the base closure.

        Usage:
            NORiClosureTuple(:base, :nn)  # Both closures
            # or
            base = NORiBaseVerticalDiffusivity()
            nn = NORiNNFluxClosure(CPU())
            NORiClosureTuple(base, nn)
        """)
    end

    # Build closure tuple in correct order
    closures = []

    # 1. Base closure goes first (if present)
    if !isnothing(base_closure)
        push!(closures, base_closure)
    end

    # 2. NN closure goes second (if present)
    if !isnothing(nn_closure)
        push!(closures, nn_closure)
    end

    # 3. Other closures go after base and NN
    append!(closures, other_closures)

    # Return as tuple
    if length(closures) == 0
        error("No closures provided! At least one closure must be specified.")
    elseif length(closures) == 1
        return closures[1]  # Return single closure (not tuple)
    else
        return Tuple(closures)
    end
end

"""
    NORiBaseClosureOnly(; kwargs...)

Convenience function to create just the base closure.

Equivalent to `NORiClosureTuple(:base; kwargs...)`.

# Examples
```julia
closure = NORiBaseClosureOnly()
closure = NORiBaseClosureOnly(νˢʰ=0.1, Riᶜ=0.5)
```
"""
NORiBaseClosureOnly(; kwargs...) = NORiBaseVerticalDiffusivity(; kwargs...)

"""
    NORiClosureWithNN(; arch=CPU(), nn_model_path=nothing, kwargs...)

Convenience function to create base + NN closure tuple.

Equivalent to `NORiClosureTuple(:base, :nn; arch=arch, nn_model_path=nn_model_path, kwargs...)`.

# Examples
```julia
closures = NORiClosureWithNN()
closures = NORiClosureWithNN(arch=GPU())
closures = NORiClosureWithNN(νˢʰ=0.1, arch=GPU())
```
"""
NORiClosureWithNN(; arch=CPU(), nn_model_path=nothing, kwargs...) =
    NORiClosureTuple(:base, :nn; arch=arch, nn_model_path=nn_model_path, kwargs...)
