"""
    NORiNNFluxClosure

Neural network-based turbulence closure for ocean parameterization in Oceananigans.

This closure combines a base Richardson number-dependent parameterization with learned
neural network corrections for temperature and salinity fluxes. The neural networks are
trained to predict residual fluxes in convective regions based on local stratification
and Richardson number profiles.

# Architecture
- Separate neural networks for temperature (wT) and salinity (wS) fluxes
- Inputs: arctan(Ri), ∂T/∂z, ∂S/∂z, ∂ρ/∂z at multiple vertical levels + surface buoyancy flux
- Sliding window approach for local vertical structure
- Active only in convecting regions (positive surface buoyancy flux)

# Usage
```julia
using NORiOceanParameterization.Implementation

# Create closure (automatically loads trained model and scaling parameters)
nn_closure = NORiNNFluxClosure(GPU())  # or CPU()

# Use in Oceananigans model with base closure
using Oceananigans
base_closure = NORiBaseVerticalDiffusivity()
model = HydrostaticFreeSurfaceModel(
    grid = grid,
    closure = (base_closure, nn_closure),
    ...
)
```

# Model Files
The closure loads a pre-trained model file. Update the `nn_path` in the constructor
to use different trained models.
"""

import Oceananigans.TurbulenceClosures:
        compute_diffusivities!,
        DiffusivityFields,
        viscosity,
        diffusivity,
        diffusive_flux_x,
        diffusive_flux_y,
        diffusive_flux_z,
        viscous_flux_ux,
        viscous_flux_uy,
        viscous_flux_uz,
        viscous_flux_vx,
        viscous_flux_vy,
        viscous_flux_vz,
        viscous_flux_wx,
        viscous_flux_wy,
        viscous_flux_wz

using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b, g_Earth, top_buoyancy_flux
using Oceananigans.Coriolis
using Oceananigans.Grids: φnode, total_size
using Oceananigans.Utils: KernelParameters, launch!
using Oceananigans: architecture, on_architecture
using Oceananigans.Operators: ∂zᶜᶜᶠ
using Oceananigans.Fields: Field, ZFaceField

using Lux, LuxCUDA
using JLD2
using ComponentArrays
using SeawaterPolynomials.TEOS10

using KernelAbstractions: @index, @kernel, @private

import Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, ExplicitTimeDiscretization

using Adapt

include("../DataWrangling/feature_scaling.jl")

@inline hack_sind(φ) = sin(φ * π / 180)

@inline fᶜᶜᵃ(i, j, k, grid, coriolis::HydrostaticSphericalCoriolis) = 2 * coriolis.rotation_rate * hack_sind(φnode(i, j, k, grid, Center(), Center(), Center()))
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::FPlane)    = coriolis.f
@inline fᶜᶜᵃ(i, j, k, grid, coriolis::BetaPlane) = coriolis.f₀ + coriolis.β * ynode(i, j, k, grid, Center(), Center(), Center())

"""
    NN{M, P, S}

Wrapper for Lux neural network model with parameters and state.
"""
struct NN{M, P, S}
    model   :: M
    ps      :: P
    st      :: S
end

@inline (neural_network::NN)(input) = first(neural_network.model(input, neural_network.ps, neural_network.st))
@inline tosarray(x::AbstractArray) = SArray{Tuple{size(x)...}}(x)

"""
    NORiNNFluxClosure{A <: NN, S, G} <: AbstractTurbulenceClosure

Neural network closure that predicts residual temperature and salinity fluxes.

# Fields
- `wT::NN`: Neural network for temperature flux
- `wS::NN`: Neural network for salinity flux
- `scaling::S`: Zero-mean unit-variance scaling parameters
- `grid_point_above::G`: Number of grid points above Richardson number threshold
- `grid_point_below::G`: Number of grid points below Richardson number threshold
"""
struct NORiNNFluxClosure{A <: NN, S, G} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 3}
    wT               :: A
    wS               :: A
    scaling          :: S
    grid_point_above :: G
    grid_point_below :: G
end

Adapt.adapt_structure(to, nn :: NORiNNFluxClosure) =
    NORiNNFluxClosure(Adapt.adapt(to, nn.wT),
                      Adapt.adapt(to, nn.wS),
                      Adapt.adapt(to, nn.scaling),
                      Adapt.adapt(to, nn.grid_point_above),
                      Adapt.adapt(to, nn.grid_point_below))

Adapt.adapt_structure(to, nn :: NN) =
    NN(Adapt.adapt(to, nn.model),
       Adapt.adapt(to, nn.ps),
       Adapt.adapt(to, nn.st))

"""
    NORiNNFluxClosure(arch; model_path=nothing)

Construct a neural network flux closure for the given architecture.

# Arguments
- `arch`: Architecture (CPU() or GPU())
- `model_path`: Optional path to trained model file. If not provided, uses default model.

# Returns
- `NORiNNFluxClosure` instance ready for use in Oceananigans
"""
function NORiNNFluxClosure(arch; model_path=nothing)
    dev = ifelse(arch == GPU(), gpu_device(), cpu_device())

    # Default model path - update this to use different trained models
    if isnothing(model_path)
        model_path = joinpath(pwd(), "calibrated_parameters", "NNclosure_weights.jld2")
    end

    # Load trained model parameters
    ps, sts, scaling_params, wT_model, wS_model = jldopen(model_path, "r") do file
        ps = file["u"] |> dev |> f64
        sts = file["sts"] |> dev |> f64
        scaling_params = file["scaling"]
        wT_model = file["model"].wT
        wS_model = file["model"].wS
        return ps, sts, scaling_params, wT_model, wS_model
    end

    # Construct scaling transformation
    scaling = construct_zeromeanunitvariance_scaling(scaling_params)

    # Create neural network wrappers
    wT_NN = NN(wT_model, ps.wT, sts.wT)
    wS_NN = NN(wS_model, ps.wS, sts.wS)

    # Vertical extent of NN active region around Richardson number threshold
    grid_point_above = 10
    grid_point_below = 5

    return NORiNNFluxClosure(wT_NN, wS_NN, scaling, grid_point_above, grid_point_below)
end

"""
    DiffusivityFields(grid, tracer_names, bcs, closure::NORiNNFluxClosure)

Create work arrays and fields for neural network flux computation.
"""
function DiffusivityFields(grid, tracer_names, bcs, closure::NORiNNFluxClosure)
    arch = architecture(grid)
    wT = ZFaceField(grid)
    wS = ZFaceField(grid)
    first_index = Field((Center, Center, Nothing), grid, Int32)
    last_index = Field((Center, Center, Nothing), grid, Int32)

    N_input = closure.wT.model.layers.layer_1.in_dims
    N_levels = closure.grid_point_above + closure.grid_point_below

    Nx_in, Ny_in, _ = size(wT)
    wrk_in = zeros(N_input, Nx_in, Ny_in, N_levels)
    wrk_in = on_architecture(arch, wrk_in)

    wrk_wT = zeros(Nx_in, Ny_in, 15)
    wrk_wS = zeros(Nx_in, Ny_in, 15)
    wrk_wT = on_architecture(arch, wrk_wT)
    wrk_wS = on_architecture(arch, wrk_wS)

    return (; wrk_in, wrk_wT, wrk_wS, wT, wS, first_index, last_index)
end

"""
    compute_diffusivities!(diffusivities, closure::NORiNNFluxClosure, model; parameters = :xyz)

Compute neural network flux corrections.

# Steps
1. Find active region based on Richardson number threshold
2. Populate NN inputs with local stratification and Ri profiles
3. Evaluate neural networks to get flux predictions
4. Apply scaling and boundary conditions
"""
function compute_diffusivities!(diffusivities, closure::NORiNNFluxClosure, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers    = model.tracers
    buoyancy   = model.buoyancy
    coriolis   = model.coriolis
    clock      = model.clock
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    # Get Richardson number from base closure
    Riᶜ = model.closure[1].Riᶜ
    Ri = model.diffusivity_fields[1].Ri

    wrk_in = diffusivities.wrk_in
    wrk_wT = diffusivities.wrk_wT
    wrk_wS = diffusivities.wrk_wS
    wT = diffusivities.wT

    first_index = diffusivities.first_index
    last_index = diffusivities.last_index

    Nx_in, Ny_in, Nz_in = total_size(wT)
    ox_in, oy_in, oz_in = wT.data.offsets
    kp = KernelParameters((Nx_in, Ny_in, Nz_in), (ox_in, oy_in, oz_in))
    kp_2D = KernelParameters((Nx_in, Ny_in), (ox_in, oy_in))

    N_levels = closure.grid_point_above + closure.grid_point_below

    Nx_wrk, Ny_wrk, _ = size(wT)
    # kp_wrk = KernelParameters((Nx_in, Ny_in, N_levels), (0, 0, 0))
    kp_wrk = KernelParameters((Nx_wrk, Ny_wrk, N_levels), (0, 0, 0))

    # Step 1: Find active region based on Ri threshold
    launch!(arch, grid, kp_2D, _find_NN_active_region!, Ri, grid, Riᶜ, first_index, last_index, closure)

    # Step 2: Populate NN inputs
    launch!(arch, grid, kp_wrk,
            _populate_input!, wrk_in, first_index, last_index, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)

    # Step 3: Evaluate neural networks
    wrk_wT .= dropdims(closure.wT(wrk_in), dims=1)
    wrk_wS .= dropdims(closure.wS(wrk_in), dims=1)

    # Step 4: Apply scaling and fill output fields
    launch!(arch, grid, kp, _fill_adjust_nn_fluxes!, diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)

    return nothing
end

"""
    _populate_input!(input, first_index, last_index, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)

Kernel to populate neural network input arrays with local vertical profiles.

Inputs at each vertical level include:
- arctan(Ri) at 5 vertical points (sliding window)
- ∂T/∂z at 5 vertical points
- ∂S/∂z at 5 vertical points
- ∂ρ/∂z at 5 vertical points
- Surface buoyancy flux (scalar)
"""
@kernel function _populate_input!(input, first_index, last_index, grid, closure::NORiNNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, Ri, clock)
    i, j, k = @index(Global, NTuple)

    scaling = closure.scaling

    ρ₀ = TEOS10EquationOfState().reference_density
    g = g_Earth

    @inbounds k_first = first_index[i, j, 1]
    @inbounds k_last = last_index[i, j, 1]

    quiescent = quiescent_condition(k_first, k_last)

    @inbounds k_tracer = clamp_k_interior(k_first + k - 1, grid)

    # 5-point stencil centered at k_tracer
    @inbounds k₋₂ = clamp_k_interior(k_tracer - 2, grid)
    @inbounds k₋₁ = clamp_k_interior(k_tracer - 1, grid)
    @inbounds k₀ = clamp_k_interior(k_tracer, grid)
    @inbounds k₊₁ = clamp_k_interior(k_tracer + 1, grid)
    @inbounds k₊₂ = clamp_k_interior(k_tracer + 2, grid)

    T, S = tracers.T, tracers.S

    # Richardson number inputs (arctan transformed)
    @inbounds input[1, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₋₂]))
    @inbounds input[2, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₋₁]))
    @inbounds input[3, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₀]))
    @inbounds input[4, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₊₁]))
    @inbounds input[5, i, j, k] = ifelse(quiescent, 0, atan(Ri[i, j, k₊₂]))

    # Temperature gradient inputs (scaled)
    @inbounds input[6, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₂, grid, T)))
    @inbounds input[7, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₁, grid, T)))
    @inbounds input[8, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₀, grid, T)))
    @inbounds input[9, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₁, grid, T)))
    @inbounds input[10, i, j, k] = scaling.∂T∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₂, grid, T)))

    # Salinity gradient inputs (scaled)
    @inbounds input[11, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₂, grid, S)))
    @inbounds input[12, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₋₁, grid, S)))
    @inbounds input[13, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₀, grid, S)))
    @inbounds input[14, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₁, grid, S)))
    @inbounds input[15, i, j, k] = scaling.∂S∂z(ifelse(quiescent, 0, ∂zᶜᶜᶠ(i, j, k₊₂, grid, S)))

    # Density gradient inputs (scaled)
    @inbounds input[16, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₋₂, grid, buoyancy, tracers) / g))
    @inbounds input[17, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₋₁, grid, buoyancy, tracers) / g))
    @inbounds input[18, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₀, grid, buoyancy, tracers) / g))
    @inbounds input[19, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₊₁, grid, buoyancy, tracers) / g))
    @inbounds input[20, i, j, k] = scaling.∂ρ∂z(ifelse(quiescent, 0, -ρ₀ * ∂z_b(i, j, k₊₂, grid, buoyancy, tracers) / g))

    # Surface buoyancy flux (scaled)
    @inbounds input[21, i, j, k] = scaling.wb(top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)))

end

"""
    _find_NN_active_region!(Ri, grid, Riᶜ, first_index, last_index, closure)

Kernel to find the vertical extent of the NN active region based on Richardson number.

Identifies the base of the mixed layer by tracking the deepest unstable point (Ri < Riᶜ)
that satisfies mixed layer criteria.

# Algorithm
The function descends from the surface (k = grid.Nz) downward, accumulating counts of
stable and unstable points. The variable `index_mixed_layer_base` continuously updates
to track the deepest unstable point that meets mixed layer criteria, stopping when:
1. Flow becomes stable (Ri >= Riᶜ), OR
2. Mixed layer criteria are no longer satisfied

# Mixed Layer Criteria
Two criteria are applied depending on depth:

**Above cutoff depth** (k > stability_threshold_cutoff, within 24 points of surface):
- Relaxed criteria: requires n_unstable >= n_stable
- Allows NN activation in typical mixed layer depths

**Below cutoff depth** (k <= stability_threshold_cutoff, deeper than 24 points):
- Strict criteria: requires n_stable == 0 OR n_unstable/n_stable >= stability_threshold
- Prevents spurious NN activation in deep, weakly unstable regions

The `|` (OR) operator in `column_above_largely_unstable` handles the edge case where
n_stable = 0 (fully unstable column), avoiding division by zero while correctly
identifying a well-mixed column.

# Output
`first_index` and `last_index` define the vertical extent of the NN active region:
- Centered around `background_κ_index` (one point below the mixed layer base)
- Extended by `grid_point_above_kappa` points upward
- Extended by `grid_point_below_kappa` points downward
- Set to empty range if no mixed layer detected
"""
@kernel function _find_NN_active_region!(Ri, grid, Riᶜ, first_index, last_index, closure::NORiNNFluxClosure)
    i, j = @index(Global, NTuple)
    top_index = grid.Nz + 1
    grid_point_above_kappa = closure.grid_point_above
    grid_point_below_kappa = closure.grid_point_below

    n_stable = 0
    n_unstable = 0

    # Initialize to surface - will update as we descend to find base of mixed layer
    index_mixed_layer_base = grid.Nz+1

    # Depth cutoff: 24 grid points below surface which is 25 * 8 m = 200 m
    # Prevents NN from activating in deep, weakly unstable regions
    stability_threshold_cutoff = grid.Nz+1 - 25
    
    # Ratio threshold for deep regions: requires 4:1 unstable:stable ratio
    stability_threshold = 4

    # Descend from surface, tracking deepest point that is both unstable and
    # satisfies mixed layer criteria
    @inbounds for k in grid.Nz:-1:2
        unstable = Ri[i, j, k] < Riᶜ

        # Accumulate counts as we descend
        n_stable = ifelse(unstable, n_stable, n_stable + 1)
        n_unstable = ifelse(unstable, n_unstable + 1, n_unstable)

        # Check if current point should be considered near the surface
        is_near_surface = k > stability_threshold_cutoff
        
        # Use OR to handle n_stable=0 case (fully unstable column) without division
        column_above_largely_unstable = (n_stable == 0) | (n_unstable/n_stable >= stability_threshold)
        
        # Near surface: relaxed criteria (just need more unstable than stable)
        # Deep: strict criteria (need strong ratio OR fully unstable column)
        meets_stability_criteria = is_near_surface | column_above_largely_unstable

        # Update if: unstable AND enough unstable points AND depth criteria met
        should_update_mixed_layer_depth = unstable & (n_unstable >= n_stable) & meets_stability_criteria
        index_mixed_layer_base = ifelse(should_update_mixed_layer_depth, k, index_mixed_layer_base)
    end

    # Define NN active region around the point just above mixed layer base
    background_κ_index = index_mixed_layer_base - 1
    no_mixed_layer = index_mixed_layer_base == top_index

    # If no mixed layer: set empty range (last < first)
    # Otherwise: extend region above and below background_κ_index
    @inbounds last_index[i, j, 1] = ifelse(no_mixed_layer, top_index - 1, clamp_k_interior(background_κ_index + grid_point_above_kappa, grid))
    @inbounds first_index[i, j, 1] = ifelse(no_mixed_layer, top_index, clamp_k_interior(background_κ_index - grid_point_below_kappa + 1, grid))

end

@inline function quiescent_condition(lo, hi)
    return hi < lo
end

@inline function within_zone_condition(k, lo, hi)
    return (k >= lo) & (k <= hi)
end

@inline function clamp_k_interior(k, grid)
    kmax = grid.Nz
    kmin = 2

    return clamp(k, kmin, kmax)
end

"""
    _fill_adjust_nn_fluxes!(diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure, tracers, velocities, buoyancy, top_tracer_bcs, clock)

Kernel to apply NN predictions to flux fields with proper scaling and boundary conditions.

NN is only active when:
- Surface buoyancy flux is positive (convecting)
- Region is not quiescent (valid active region found)
- Grid point is within the active zone
"""
@kernel function _fill_adjust_nn_fluxes!(diffusivities, first_index, last_index, wrk_wT, wrk_wS, grid, closure::NORiNNFluxClosure, tracers, velocities, buoyancy, top_tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    scaling = closure.scaling

    @inbounds k_first = first_index[i, j, 1]
    @inbounds k_last = last_index[i, j, 1]

    convecting = top_buoyancy_flux(i, j, grid, buoyancy, top_tracer_bcs, clock, merge(velocities, tracers)) > 0
    quiescent = quiescent_condition(k_first, k_last)
    within_zone = within_zone_condition(k, k_first, k_last)

    N_levels = closure.grid_point_above + closure.grid_point_below
    @inbounds k_wrk = clamp(k - k_first + 1, 1, N_levels)

    NN_active = convecting & !quiescent & within_zone

    # Apply inverse scaling to get physical fluxes
    @inbounds diffusivities.wT[i, j, k] = ifelse(NN_active, scaling.wT.σ * wrk_wT[i, j, k_wrk], 0)
    @inbounds diffusivities.wS[i, j, k] = ifelse(NN_active, scaling.wS.σ * wrk_wS[i, j, k_wrk], 0)
end

const NNC = NORiNNFluxClosure

#####
##### Flux functions for Oceananigans interface
#####

# Horizontal fluxes are zero
@inline viscous_flux_wz( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_wx( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_wy( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_ux( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_vx( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_uy( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_vy( i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline diffusive_flux_x(i, j, k, grid, clo::NNC, K, ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index = zero(grid)
@inline diffusive_flux_y(i, j, k, grid, clo::NNC, K, ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index = zero(grid)

# Viscous fluxes are zero (for now)
@inline viscous_flux_uz(i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)
@inline viscous_flux_vz(i, j, k, grid, clo::NNC, K, clk, fields, b) = zero(grid)

# Vertical diffusive fluxes (the only non-zero fluxes)
@inline diffusive_flux_z(i, j, k, grid, clo::NNC, K, ::Val{1}, c, clock, fields, buoyancy) = @inbounds K.wT[i, j, k]
@inline diffusive_flux_z(i, j, k, grid, clo::NNC, K, ::Val{2}, c, clock, fields, buoyancy) = @inbounds K.wS[i, j, k]


#####
##### Show
#####

function Base.summary(closure::NORiNNFluxClosure)
    return "NORiNNFluxClosure"
end

function Base.show(io::IO, closure::NORiNNFluxClosure)
    print(io, summary(closure))
    print(io, '\n')
    
    # Get layer information for wT network
    wT_layers = closure.wT.model.layers
    wT_layer_names = propertynames(wT_layers)
    wT_total_params = length(closure.wT.ps)
    
    # Get layer information for wS network
    wS_layers = closure.wS.model.layers
    wS_layer_names = propertynames(wS_layers)
    wS_total_params = length(closure.wS.ps)
    
    # Get parameter precision
    param_type = eltype(closure.wT.ps)
    
    print(io, "├── grid_point_above: ", prettysummary(closure.grid_point_above), '\n',
              "├── grid_point_below: ", prettysummary(closure.grid_point_below), '\n',
              "├── parameter type: ", param_type, '\n',
              "├── wT neural network (", wT_total_params, " parameters): ", '\n')
    
    # Display wT layers with activation functions
    for (idx, name) in enumerate(wT_layer_names)
        layer = getproperty(wT_layers, name)
        in_dim = layer.in_dims
        out_dim = layer.out_dims
        activation = hasproperty(layer, :activation) ? nameof(typeof(layer.activation)) : "none"
        
        if idx < length(wT_layer_names)
            print(io, "│   ├── layer ", idx, ": ", in_dim, " → ", out_dim, " (", activation, ")", '\n')
        else
            print(io, "│   └── layer ", idx, ": ", in_dim, " → ", out_dim, " (", activation, ")", '\n')
        end
    end
    
    print(io, "└── wS neural network (", wS_total_params, " parameters): ", '\n')
    
    # Display wS layers with activation functions
    for (idx, name) in enumerate(wS_layer_names)
        layer = getproperty(wS_layers, name)
        in_dim = layer.in_dims
        out_dim = layer.out_dims
        activation = hasproperty(layer, :activation) ? nameof(typeof(layer.activation)) : "none"
        
        if idx < length(wS_layer_names)
            print(io, "    ├── layer ", idx, ": ", in_dim, " → ", out_dim, " (", activation, ")", '\n')
        else
            print(io, "    └── layer ", idx, ": ", in_dim, " → ", out_dim, " (", activation, ")", '\n')
        end
    end
end