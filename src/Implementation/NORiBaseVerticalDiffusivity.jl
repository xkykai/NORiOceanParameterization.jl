"""
    NORiBaseVerticalDiffusivity

Richardson number-based vertical diffusivity parameterization for ocean turbulence.

This closure implements a local parameterization that adjusts vertical diffusivities
based on the Richardson number (Ri), distinguishing between convective (Ri < 0) and
shear-driven (Ri > 0) regimes.

# Physics
- **Convective regime (Ri < 0)**: Uses tanh transition between shear and convective diffusivities
- **Stable regime (Ri > 0)**: Linear interpolation between background and shear diffusivities based on Ri/Riᶜ
- **Critical Richardson number (Riᶜ)**: Threshold for transition to background diffusivity
- **Separate Prandtl numbers**: Different values for convective vs shear regimes

# Parameters
- `ν₀`: Background viscosity (m²/s)
- `νˢʰ`: Shear-driven viscosity (m²/s)
- `νᶜⁿ`: Convective viscosity (m²/s)
- `Pr_convₜ`: Prandtl number for convective regime
- `Pr_shearₜ`: Prandtl number for shear regime
- `Riᶜ`: Critical Richardson number threshold
- `δRi`: Richardson number scale for tanh transition

# Usage
```julia
using NORiOceanParameterization.Implementation

# Create closure with default parameters (trained values)
base_closure = NORiBaseVerticalDiffusivity()

# Or specify custom parameters
base_closure = NORiBaseVerticalDiffusivity(
    ν₀ = 1e-5,
    νˢʰ = 0.062,
    νᶜⁿ = 0.761,
    Pr_convₜ = 0.175,
    Pr_shearₜ = 1.084,
    Riᶜ = 0.437,
    δRi = 0.0097
)

# Use in Oceananigans model
using Oceananigans
model = HydrostaticFreeSurfaceModel(
    grid = grid,
    closure = base_closure,
    ...
)
```

# Implementation Details
- Computes Richardson number: Ri = N²/S² where N² is buoyancy frequency and S² is shear
- Applies horizontal filtering to Richardson number for stability
- Uses vertically implicit time discretization for numerical stability
"""

using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b
using Oceananigans.Grids: inactive_node, total_size
using Oceananigans.Operators
using Oceananigans.Operators: ℑxyᶠᶠᵃ, ℑxyᶜᶜᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures:
    AbstractScalarDiffusivity,
    VerticalFormulation,
    VerticallyImplicitTimeDiscretization,
    getclosure

# Import (not using) functions we will extend
import Oceananigans.TurbulenceClosures: viscosity, diffusivity, compute_diffusivities!, DiffusivityFields
using Oceananigans.Utils: KernelParameters, launch!, prettysummary

using Adapt
using KernelAbstractions: @index, @kernel

#####
##### NORi Base Vertical Diffusivity Closure
#####

"""
    NORiBaseVerticalDiffusivity{TD, FT}

Richardson number-based vertical diffusivity closure.

# Fields
- `ν₀::FT`: Background viscosity
- `νˢʰ::FT`: Shear-driven viscosity
- `νᶜⁿ::FT`: Convective viscosity
- `Pr_convₜ::FT`: Prandtl number for convective regime
- `Pr_shearₜ::FT`: Prandtl number for shear regime
- `Riᶜ::FT`: Critical Richardson number
- `δRi::FT`: Richardson number transition scale
"""
struct NORiBaseVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    ν₀        :: FT
    νˢʰ       :: FT
    νᶜⁿ       :: FT
    Pr_convₜ  :: FT
    Pr_shearₜ :: FT
    Riᶜ       :: FT
    δRi       :: FT
end

"""
    NORiBaseVerticalDiffusivity(time_discretization, FT; parameter_file=nothing, kwargs...)

Construct a NORi base vertical diffusivity closure.

# Arguments
- `time_discretization`: Time discretization method (default: VerticallyImplicitTimeDiscretization())
- `FT`: Floating point type (default: Float64)

# Keyword Arguments
- `parameter_file`: Path to JLD2 file containing calibrated parameters (default: "calibrated_parameters/baseclosure_final.jld2")
  - If file exists and contains "u" key → Load complete parameter set from file
  - If file missing or no "u" key → Use default kwargs below
  - Set to `nothing` to disable file loading and use kwargs
- `ν₀`: Background viscosity (default: 1e-5)
- `νˢʰ`: Shear-driven viscosity (default: 0.0615914063656973)
- `νᶜⁿ`: Convective viscosity (default: 0.7612393837759673)
- `Pr_convₜ`: Prandtl number for convective regime (default: 0.1749433627329692)
- `Pr_shearₜ`: Prandtl number for shear regime (default: 1.0842017486284887)
- `Riᶜ`: Critical Richardson number (default: 0.4366901962987793)
- `δRi`: Richardson number transition scale (default: 0.009695724988589002)

# Parameter File Format
The JLD2 file should contain a key `"u"` with a ComponentArray containing:
- `ν_shear`: Shear-driven viscosity (maps to νˢʰ)
- `ν_conv`: Convective viscosity (maps to νᶜⁿ)
- `Pr_shear`: Prandtl number for shear regime (maps to Pr_shearₜ)
- `Pr_conv`: Prandtl number for convective regime (maps to Pr_convₜ)
- `Riᶜ`: Critical Richardson number
- `ΔRi`: Richardson number transition scale (maps to δRi)

Alternatively, the file can contain direct keys: "ν₀", "νˢʰ", "νᶜⁿ", "Pr_convₜ", "Pr_shearₜ", "Riᶜ", "δRi"

# Examples
```julia
# Load parameters from default file (calibrated_parameters/baseclosure_final.jld2)
closure = NORiBaseVerticalDiffusivity()

# Load from custom file
closure = NORiBaseVerticalDiffusivity(parameter_file="path/to/params.jld2")

# Use manual parameters (disable file loading)
closure = NORiBaseVerticalDiffusivity(parameter_file=nothing, νˢʰ=0.1, Riᶜ=0.5)

# Create file from training results
using JLD2
ps = ComponentArray(ν_shear=0.06, ν_conv=0.76, Pr_shear=1.08, Pr_conv=0.17, Riᶜ=0.44, ΔRi=0.01)
jldsave("my_params.jld2", u=ps)
closure = NORiBaseVerticalDiffusivity(parameter_file="my_params.jld2")
```
"""
function NORiBaseVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                     FT = Float64;
                                     parameter_file = nothing,
                                     ν₀        = 1e-5,
                                     νˢʰ       = 0.0615914063656973,
                                     νᶜⁿ       = 0.7612393837759673,
                                     Pr_convₜ  = 0.1749433627329692,
                                     Pr_shearₜ = 1.0842017486284887,
                                     Riᶜ       = 0.4366901962987793,
                                     δRi       = 0.009695724988589002)

    TD = typeof(time_discretization)

    # Try to load parameters from file if it exists
    # If file has "u" key, load complete parameter set from there
    # Otherwise use the default kwargs
    if !isnothing(parameter_file) && isfile(parameter_file)
        @info "Loading NORi base closure parameters from file: $parameter_file"
        params = jldopen(parameter_file, "r") do file
            if haskey(file, "u")
                # Load complete parameter set from training results
                u = file["u"]
                (
                    ν₀        = ν₀,  # ν₀ not trained, use default
                    νˢʰ       = u.ν_shear,
                    νᶜⁿ       = u.ν_conv,
                    Pr_convₜ  = u.Pr_conv,
                    Pr_shearₜ = u.Pr_shear,
                    Riᶜ       = u.Riᶜ,
                    δRi       = u.ΔRi
                )
            else
                # No "u" key - use default kwargs
                (ν₀=ν₀, νˢʰ=νˢʰ, νᶜⁿ=νᶜⁿ, Pr_convₜ=Pr_convₜ, Pr_shearₜ=Pr_shearₜ, Riᶜ=Riᶜ, δRi=δRi)
            end
        end
    elseif !isnothing(parameter_file)
        @warn "Parameter file not found: $parameter_file. Using default parameters."
        params = (ν₀=ν₀, νˢʰ=νˢʰ, νᶜⁿ=νᶜⁿ, Pr_convₜ=Pr_convₜ, Pr_shearₜ=Pr_shearₜ, Riᶜ=Riᶜ, δRi=δRi)
    else
        # parameter_file explicitly set to nothing - use kwargs
        params = (ν₀=ν₀, νˢʰ=νˢʰ, νᶜⁿ=νᶜⁿ, Pr_convₜ=Pr_convₜ, Pr_shearₜ=Pr_shearₜ, Riᶜ=Riᶜ, δRi=δRi)
    end

    return NORiBaseVerticalDiffusivity{TD, FT}(
        convert(FT, params.ν₀),
        convert(FT, params.νˢʰ),
        convert(FT, params.νᶜⁿ),
        convert(FT, params.Pr_convₜ),
        convert(FT, params.Pr_shearₜ),
        convert(FT, params.Riᶜ),
        convert(FT, params.δRi)
    )
end

NORiBaseVerticalDiffusivity(FT::DataType; kw...) =
    NORiBaseVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Adapt.adapt_structure(to, clo::NORiBaseVerticalDiffusivity{TD, FT}) where {TD, FT} =
    NORiBaseVerticalDiffusivity{TD, FT}(clo.ν₀, clo.νˢʰ, clo.νᶜⁿ, clo.Pr_convₜ, clo.Pr_shearₜ, clo.Riᶜ, clo.δRi)

#####
##### Diffusivity field utilities
#####

const NBVD = NORiBaseVerticalDiffusivity
const NBVDArray = AbstractArray{<:NBVD}
const FlavorOfNBVD = Union{NBVD, NBVDArray}
const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfNBVD)   = (c, c, f)
@inline diffusivity_location(::FlavorOfNBVD) = (c, c, f)

@inline viscosity(::FlavorOfNBVD, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfNBVD, diffusivities, id) = diffusivities.κᶜ

with_tracers(tracers, closure::FlavorOfNBVD) = closure

"""
    DiffusivityFields(grid, tracer_names, bcs, closure::NORiBaseVerticalDiffusivity)

Create diffusivity fields for the NORi base closure.

Returns a NamedTuple with:
- `κᶜ`: Tracer diffusivity field
- `κᵘ`: Momentum diffusivity (viscosity) field
- `Ri`: Richardson number field
"""
function DiffusivityFields(grid, tracer_names, bcs, closure::FlavorOfNBVD)
    κᶜ = Field((Center, Center, Face), grid)
    κᵘ = Field((Center, Center, Face), grid)
    Ri = Field((Center, Center, Face), grid)
    return (; κᶜ, κᵘ, Ri)
end

#####
##### Compute diffusivities
#####

"""
    compute_diffusivities!(diffusivities, closure::NORiBaseVerticalDiffusivity, model; parameters = :xyz)

Compute Richardson number-based diffusivities.

# Steps
1. Compute Richardson number at all grid points
2. Fill halo regions for Richardson number field
3. Compute diffusivities based on Richardson number and regime
"""
function compute_diffusivities!(diffusivities, closure::FlavorOfNBVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    Nx_in, Ny_in, Nz_in = total_size(diffusivities.κᶜ)
    ox_in, oy_in, oz_in = diffusivities.κᶜ.data.offsets

    kp = KernelParameters((Nx_in, Ny_in, Nz_in), (ox_in, oy_in, oz_in))

    # Step 1: Compute Richardson number
    launch!(arch, grid, kp, compute_ri_number!,
            diffusivities, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock)

    # Step 2: Fill halos (use only_local_halos to avoid communication)
    fill_halo_regions!(diffusivities.Ri; only_local_halos=true)

    # Step 3: Compute diffusivities based on Richardson number
    launch!(arch, grid, kp, compute_NORi_diffusivities!,
            diffusivities, grid, closure, velocities, tracers, buoyancy, top_tracer_bcs, clock)

    return nothing
end

#####
##### Richardson number calculation
#####

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

"""
    shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)

Compute squared vertical shear S² = (∂u/∂z)² + (∂v/∂z)² at cell center, face location.
"""
@inline function shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return ∂z_u² + ∂z_v²
end

"""
    Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)

Compute Richardson number Ri = N²/S² at cell center, face location.

Returns zero if N² = 0 to avoid NaN.
"""
@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    Ri = N² / (S² + 1e-11)

    # Avoid NaN when N² = 0
    return ifelse(N² == 0, zero(grid), Ri)
end

"""
    compute_ri_number!(diffusivities, grid, closure, velocities, tracers, buoyancy, tracer_bcs, clock)

Kernel to compute Richardson number at all grid points.
"""
@kernel function compute_ri_number!(diffusivities, grid, closure::FlavorOfNBVD,
                                    velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, buoyancy, tracers)
end

#####
##### Diffusivity computation kernel
#####

"""
    compute_NORi_diffusivities!(diffusivities, grid, closure, velocities, tracers, buoyancy, tracer_bcs, clock)

Kernel to compute diffusivities based on Richardson number.
"""
@kernel function compute_NORi_diffusivities!(diffusivities, grid, closure::FlavorOfNBVD,
                                             velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    _compute_NORi_diffusivities!(i, j, k, diffusivities, grid, closure,
                                 velocities, tracers, buoyancy, tracer_bcs, clock)
end

"""
    _compute_NORi_diffusivities!(i, j, k, diffusivities, grid, closure, velocities, tracers, buoyancy, tracer_bcs, clock)

Compute local diffusivities at grid point (i, j, k).

# Physics
- **Convective (Ri < 0)**: ν = (νˢʰ - νᶜⁿ) * tanh(Ri/δRi) + νˢʰ
- **Stable (Ri > 0)**: ν = (ν₀ - νˢʰ) * Ri/Riᶜ + νˢʰ, clamped to [ν₀, νˢʰ]
- Diffusivities computed from viscosities using Prandtl numbers
- Boundary values (k=1 and k=Nz+1) set to zero
"""
@inline function _compute_NORi_diffusivities!(i, j, k, diffusivities, grid, closure,
                                              velocities, tracers, buoyancy, tracer_bcs, clock)
    # Get closure parameters (works with ensembles of closures)
    closure_ij = getclosure(i, j, closure)

    ν₀        = closure_ij.ν₀
    νˢʰ       = closure_ij.νˢʰ
    νᶜⁿ       = closure_ij.νᶜⁿ
    Pr_convₜ  = closure_ij.Pr_convₜ
    Pr_shearₜ = closure_ij.Pr_shearₜ
    Riᶜ       = closure_ij.Riᶜ
    δRi       = closure_ij.δRi

    # Compute diffusivities from viscosities using Prandtl numbers
    κ₀  = ν₀ / Pr_shearₜ
    κˢʰ = νˢʰ / Pr_shearₜ
    κᶜⁿ = νᶜⁿ / Pr_convₜ

    # Apply horizontal filter to Richardson number (9-point stencil)
    Ri = ℑxyᶜᶜᵃ(i, j, k, grid, ℑxyᶠᶠᵃ, diffusivities.Ri)

    # Determine regime
    convecting = Ri < 0

    # Compute local diffusivities based on regime
    if convecting
        # Convective regime: tanh transition
        ν_local = (νˢʰ - νᶜⁿ) * tanh(Ri / δRi) + νˢʰ
        κ_local = (κˢʰ - κᶜⁿ) * tanh(Ri / δRi) + κˢʰ
    else
        # Stable regime: linear interpolation, clamped
        ν_local = clamp((ν₀ - νˢʰ) * Ri / Riᶜ + νˢʰ, ν₀, νˢʰ)
        κ_local = clamp((κ₀ - κˢʰ) * Ri / Riᶜ + κˢʰ, κ₀, κˢʰ)
    end

    # Set boundary values to zero
    is_interior = k > 1 && k < grid.Nz + 1

    @inbounds diffusivities.κᵘ[i, j, k] = ifelse(is_interior, ν_local, 0)
    @inbounds diffusivities.κᶜ[i, j, k] = ifelse(is_interior, κ_local, 0)

    return nothing
end

#####
##### Show
#####

@inline time_discretization(::NORiBaseVerticalDiffusivity{TD}) where TD = TD()

function Base.summary(closure::NBVD)
    TD = nameof(typeof(time_discretization(closure)))
    return string("NORiBaseVerticalDiffusivity{$TD}")
end

function Base.show(io::IO, closure::NBVD)
    print(io, summary(closure))
    print(io, '\n')
    print(io, "├── ν₀: ", prettysummary(closure.ν₀), '\n',
              "├── νˢʰ: ", prettysummary(closure.νˢʰ), '\n',
              "├── νᶜⁿ: ", prettysummary(closure.νᶜⁿ), '\n',
              "├── Pr_convₜ: ", prettysummary(closure.Pr_convₜ), '\n',
              "├── Pr_shearₜ: ", prettysummary(closure.Pr_shearₜ), '\n',
              "├── Riᶜ: ", prettysummary(closure.Riᶜ), '\n',
              "└── δRi: ", prettysummary(closure.δRi))
end