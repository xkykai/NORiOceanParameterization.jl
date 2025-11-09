using SeawaterPolynomials.TEOS10
using NNlib: tanh_fast

"""
    local_Ri_ν(Ri, ν_conv, ν_shear, Riᶜ, ΔRi)

Compute local viscosity based on Richardson number across three regimes:
- Ri ≥ Riᶜ: Stable stratification, linearly decreases to background `ν₀`
- 0 ≤ Ri < Riᶜ: Shear-driven mixing, linearly interpolates between `ν₀` and `ν_shear`
- Ri < 0: Convective regime, tanh transition from `ν_shear` to `ν_conv`
"""
function local_Ri_ν(Ri, ν_conv, ν_shear, Riᶜ, ΔRi)
    Riᶜ <= 0 && return NaN
    ν₀ = 1e-5

    if Ri >= 0
        ν = clamp((ν₀ - ν_shear) * Ri / Riᶜ + ν_shear, ν₀, ν_shear)
    else
        ν = (ν_shear - ν_conv) * tanh_fast(Ri / ΔRi) + ν_shear
    end

    return ν
end

"""
    local_Ri_κ(Ri, ν_conv, ν_shear, Riᶜ, ΔRi, Pr_conv, Pr_shear)

Compute local diffusivity based on Richardson number across three regimes:
- Ri ≥ Riᶜ: Stable stratification, linearly decreases to background `κ₀`
- 0 ≤ Ri < Riᶜ: Shear-driven mixing, linearly interpolates between `κ₀` and `κ_shear`
- Ri < 0: Convective regime, tanh transition from `κ_shear` to `κ_conv`

Uses Prandtl numbers to convert viscosity to diffusivity.
"""
function local_Ri_κ(Ri, ν_conv, ν_shear, Riᶜ, ΔRi, Pr_conv, Pr_shear)
    Riᶜ <= 0 && return NaN
    κ₀ = 1e-5 / Pr_shear
    κ_shear = ν_shear / Pr_shear
    κ_conv = ν_conv / Pr_conv

    if Ri >= 0
        κ = clamp((κ₀ - κ_shear) * Ri / Riᶜ + κ_shear, κ₀, κ_shear)
    else
        κ = (κ_shear - κ_conv) * tanh_fast(Ri / ΔRi) + κ_shear
    end

    return κ
end

"""
    nonlocal_Ri_ν(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, Riᶜ, ΔRi, C_en, x₀, Δx, ϵ=1e-11)

Compute nonlocal viscosity including entrainment effects.
Adds entrainment contribution when transitioning from convective to stable layer with negative buoyancy flux.
"""
function nonlocal_Ri_ν(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, Riᶜ, ΔRi, C_en, x₀, Δx, ϵ=1e-11)
    ν_local = local_Ri_ν(Ri, ν_conv, ν_shear, Riᶜ, ΔRi)
    entrainment = Ri > 0 && Ri_above < 0 && Qρ < 0
    x = Qρ / (∂ρ∂z + ϵ)
    ν_nonlocal = entrainment * C_en * ν_conv * 0.5 * (tanh((x - x₀) / Δx) + 1)

    return ν_local + ν_nonlocal
end

"""
    nonlocal_Ri_κ(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, Riᶜ, ΔRi, Pr_conv, Pr_shear, C_en, x₀, Δx, ϵ=1e-11)

Compute nonlocal diffusivity including entrainment effects.
Converts nonlocal viscosity to diffusivity using appropriate Prandtl number.
"""
function nonlocal_Ri_κ(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, Riᶜ, ΔRi, Pr_conv, Pr_shear, C_en, x₀, Δx, ϵ=1e-11)
    ν = nonlocal_Ri_ν(Ri, Ri_above, ∂ρ∂z, Qρ, ν_conv, ν_shear, Riᶜ, ΔRi, C_en, x₀, Δx, ϵ)
    return ifelse(Ri >= 0, ν / Pr_shear, ν / Pr_conv)
end


