using NORiOceanParameterization.Operators: Dᶜ, Dᶠ, Iᶠ

"""
    ODEParam{F, FS, T, NT, ST, Z, L, G, CS, DC, DF, IF, DCH, DFH, UW, VW, WT, WS, SC}

Parameters for ocean ODE simulations derived from LES data.

Contains physical parameters, grid information, differential operators, boundary fluxes,
and scaling transformations for neural differential equation models.

# Key Fields
- `f, f_scaled`: Coriolis parameter (unscaled [rad/s] and scaled)
- `τ, N_timesteps, scaled_time`: Time duration [s], number of steps, normalized time [0,1]
- `zC, zF, H`: Vertical coords at centers/faces [m], domain depth [m]
- `g`: Gravitational acceleration [m/s²]
- `Dᶜ, Dᶠ, Iᶠ`: Derivative and interpolation operators
- `Dᶜ_hat, Dᶠ_hat`: Nondimensional derivatives
- `uw, vw, wT, wS`: Momentum and tracer fluxes (scaled/unscaled, top/bottom)
- `scaling`: Feature scaling object

# Example
```julia
julia> params = ODEParam(data, scaling);
julia> params.f, params.coarse_size
(1.0e-4, 32)
```
"""
struct ODEParam{F, FS, T, NT, ST, Z, L, G, CS, DC, DF, IF, DCH, DFH, UW, VW, WT, WS, SC}
                       f :: F
                f_scaled :: FS
                       τ :: T
             N_timesteps :: NT
             scaled_time :: ST
                      zC :: Z
                      zF :: Z
                       H :: L
                       g :: G
             coarse_size :: CS
                      Dᶜ :: DC
                      Dᶠ :: DF
                      Iᶠ :: IF
                  Dᶜ_hat :: DCH
                  Dᶠ_hat :: DFH
                      uw :: UW
                      vw :: VW
                      wT :: WT
                      wS :: WS
                 scaling :: SC
end

"""
    ODEParam(data::LESData, scaling; abs_f=false)

Construct ODE parameters from LES data with scaling transformations.

# Arguments
- `data::LESData`: LES simulation data
- `scaling`: Scaling object for physical quantities
- `abs_f=false`: Use absolute value of Coriolis parameter

# Returns
`ODEParam` with grid info, operators, and scaled/unscaled boundary fluxes.

# Example
```julia
julia> params = ODEParam(data, scaling);
julia> params_abs = ODEParam(data, scaling, abs_f=true);  # hemisphere-invariant
julia> params.wT.scaled.top  # scaled surface heat flux
```
"""
function ODEParam(data::LESData, scaling; abs_f=false)
    coarse_size = data.metadata["Nz"]
    if abs_f
        f_scaled = scaling.f(abs(data.coriolis.unscaled))
    else
        f_scaled = scaling.f(data.coriolis.unscaled)
    end

    return ODEParam(data.coriolis.unscaled,
                    f_scaled,
                    data.times[end] - data.times[1],
                    length(data.times),
                    (data.times .- data.times[1]) ./ (data.times[end] - data.times[1]),
                    data.metadata["zC"],
                    data.metadata["zF"],
                    data.metadata["original_grid"].Lz,
                    data.metadata["gravitational_acceleration"],
                    coarse_size,
                    Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]),
                    Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]),
                    Iᶠ(coarse_size),
                    Dᶜ(coarse_size, data.metadata["zC"][2] - data.metadata["zC"][1]) .* data.metadata["original_grid"].Lz,
                    Dᶠ(coarse_size, data.metadata["zF"][3] - data.metadata["zF"][2]) .* data.metadata["original_grid"].Lz,
                    (scaled = (top=scaling.uw(data.flux.uw.surface.unscaled), bottom=scaling.uw(data.flux.uw.bottom.unscaled)),
                     unscaled = (top=data.flux.uw.surface.unscaled, bottom=data.flux.uw.bottom.unscaled)),
                    (scaled = (top=scaling.vw(data.flux.vw.surface.unscaled), bottom=scaling.vw(data.flux.vw.bottom.unscaled)),
                     unscaled = (top=data.flux.vw.surface.unscaled, bottom=data.flux.vw.bottom.unscaled)),
                    (scaled = (top=scaling.wT(data.flux.wT.surface.unscaled), bottom=scaling.wT(data.flux.wT.bottom.unscaled)),
                     unscaled = (top=data.flux.wT.surface.unscaled, bottom=data.flux.wT.bottom.unscaled)),
                    (scaled = (top=scaling.wS(data.flux.wS.surface.unscaled), bottom=scaling.wS(data.flux.wS.bottom.unscaled)),
                     unscaled = (top=data.flux.wS.surface.unscaled, bottom=data.flux.wS.bottom.unscaled)),
                    scaling)
end

"""
    ODEParams(dataset::LESDatasets, scaling=dataset.scaling; abs_f=false)

Create ODE parameters for multiple LES datasets.

# Arguments
- `dataset::LESDatasets`: Collection of LES simulations
- `scaling=dataset.scaling`: Scaling object (default: dataset's scaling)
- `abs_f=false`: Use absolute Coriolis parameter

# Returns
`Vector{ODEParam}` with one parameter set per simulation.

# Example
```julia
julia> params_array = ODEParams(dataset);
julia> params_array = ODEParams(dataset, custom_scaling, abs_f=true);
```
"""
function ODEParams(dataset::LESDatasets, scaling=dataset.scaling; abs_f=false)
    return [ODEParam(data, scaling, abs_f=abs_f) for data in dataset.data]
end