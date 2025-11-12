using SeawaterPolynomials.TEOS10
using NORiOceanParameterization.Operators: Dᶠ!
using Statistics

"""
    find_min(a...)

Find the global minimum value across multiple arrays.

# Arguments
- `a...`: Variable number of arrays to search through.

# Returns
- The minimum value found across all input arrays.

# Examples
```julia
julia> find_min([1, 2, 3], [4, 5, 6])
1
```
"""
function find_min(a...)
    return minimum(minimum.([a...]))
end

"""
    find_max(a...)

Find the global maximum value across multiple arrays.

# Arguments
- `a...`: Variable number of arrays to search through.

# Returns
- The maximum value found across all input arrays.

# Examples
```julia
julia> find_max([1, 2, 3], [4, 5, 6])
6
```
"""
function find_max(a...)
    return maximum(maximum.([a...]))
end

"""
    calculate_Ri(u, v, T, S, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf), ϵ=1e-11)

Calculate the Richardson number from velocity, temperature, and salinity profiles.

The Richardson number is defined as Ri = (∂b/∂z) / (∂u/∂z)² + (∂v/∂z)²), where
b is buoyancy.

# Arguments
- `u`: Zonal velocity profile.
- `v`: Meridional velocity profile.
- `T`: Temperature profile.
- `S`: Salinity profile.
- `Dᶠ`: Finite difference operator for vertical derivatives.
- `g`: Gravitational acceleration (m/s²).
- `ρ₀`: Reference density (kg/m³).

# Keyword Arguments
- `clamp_lims=(-Inf, Inf)`: Tuple specifying (min, max) limits for clamping the Richardson number.
- `ϵ=1e-11`: Small number added to denominator to prevent division by zero.

# Returns
- Array of Richardson numbers clamped to the specified limits.
"""
function calculate_Ri(u, v, T, S, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf), ϵ=1e-11)
    eos = TEOS10EquationOfState()
    ρ′ = TEOS10.ρ′.(T, S, 0, Ref(eos))
    ∂ρ∂z = Dᶠ * ρ′
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g ./ ρ₀ .* ∂ρ∂z

    return clamp.(∂b∂z ./ (∂u∂z.^2 .+ ∂v∂z.^2 .+ ϵ), clamp_lims[1], clamp_lims[2])
end

"""
    calculate_Ri(u, v, ρ, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf), ϵ=1e-11)

Calculate the Richardson number from velocity and density profiles.

The Richardson number is defined as Ri = (∂b/∂z) / (∂u/∂z)² + (∂v/∂z)²), where
b is buoyancy.

# Arguments
- `u`: Zonal velocity profile.
- `v`: Meridional velocity profile.
- `ρ`: Density profile.
- `Dᶠ`: Finite difference operator for vertical derivatives.
- `g`: Gravitational acceleration (m/s²).
- `ρ₀`: Reference density (kg/m³).

# Keyword Arguments
- `clamp_lims=(-Inf, Inf)`: Tuple specifying (min, max) limits for clamping the Richardson number.
- `ϵ=1e-11`: Small number added to denominator to prevent division by zero.

# Returns
- Array of Richardson numbers clamped to the specified limits.
"""
function calculate_Ri(u, v, ρ, Dᶠ, g, ρ₀; clamp_lims=(-Inf, Inf), ϵ=1e-11)
    ∂ρ∂z = Dᶠ * ρ
    ∂u∂z = Dᶠ * u
    ∂v∂z = Dᶠ * v
    ∂b∂z = -g ./ ρ₀ .* ∂ρ∂z

    return clamp.(∂b∂z ./ (∂u∂z.^2 .+ ∂v∂z.^2 .+ ϵ), clamp_lims[1], clamp_lims[2])
end

"""
    calculate_Ri!(Ri, u, v, ρ, ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀, Δ; clamp_lims=(-Inf, Inf), ϵ=1e-11)

In-place calculation of the Richardson number from velocity and density profiles.

This is a mutating version that writes results directly to pre-allocated arrays,
avoiding memory allocations.

# Arguments
- `Ri`: Pre-allocated array to store the Richardson number (output).
- `u`: Zonal velocity profile.
- `v`: Meridional velocity profile.
- `ρ`: Density profile.
- `∂b∂z`: Pre-allocated array for buoyancy gradient (working array).
- `∂u∂z`: Pre-allocated array for zonal velocity gradient (working array).
- `∂v∂z`: Pre-allocated array for meridional velocity gradient (working array).
- `g`: Gravitational acceleration (m/s²).
- `ρ₀`: Reference density (kg/m³).
- `Δ`: Grid spacing for finite difference calculation.

# Keyword Arguments
- `clamp_lims=(-Inf, Inf)`: Tuple specifying (min, max) limits for clamping the Richardson number.
- `ϵ=1e-11`: Small number added to denominator to prevent division by zero.

# Note
This function modifies `Ri`, `∂b∂z`, `∂u∂z`, and `∂v∂z` in place.
"""
function calculate_Ri!(Ri, u, v, ρ, ∂b∂z, ∂u∂z, ∂v∂z, g, ρ₀, Δ; clamp_lims=(-Inf, Inf), ϵ=1e-11)
    Dᶠ!(∂u∂z, u, Δ)
    Dᶠ!(∂v∂z, v, Δ)

    Dᶠ!(∂b∂z, ρ, Δ)
    @. ∂b∂z = -g / ρ₀ * ∂b∂z

    @. Ri = clamp(∂b∂z / (∂u∂z^2 + ∂v∂z^2 + ϵ), clamp_lims[1], clamp_lims[2])
end