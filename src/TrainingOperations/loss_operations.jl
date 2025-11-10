using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using NORiOceanParameterization.DataWrangling: LESData
using Statistics

"""
    compute_density_contribution(data::LESData)

Compute density contributions from temperature and salinity variations at initial conditions.
Returns a NamedTuple with T, S contributions (in density units) and total density range ρ.
"""
function compute_density_contribution(data::LESData)
    eos = TEOS10EquationOfState()
    ρ = data.profile.ρ.unscaled[:, 1]
    T = data.profile.T.unscaled[:, 1]
    S = data.profile.S.unscaled[:, 1]

    Δρ = maximum(ρ) - minimum(ρ)
    ΔT = maximum(T) - minimum(T)
    ΔS = maximum(S) - minimum(S)

    α = mean(SeawaterPolynomials.thermal_expansion.(T, S, 0, Ref(eos)))
    β = mean(SeawaterPolynomials.haline_contraction.(T, S, 0, Ref(eos)))
    ρ₀ = eos.reference_density

    T_contribution = α * ΔT * ρ₀
    S_contribution = β * ΔS * ρ₀

    return (; T=T_contribution, S=S_contribution, ρ=Δρ)
end

"""
    compute_density_contribution(data::LESData, index)

Compute density contributions from temperature and salinity variations at specified time index.
Returns a NamedTuple with T, S contributions (in density units) and total density range ρ.
"""
function compute_density_contribution(data::LESData, index)
    eos = TEOS10EquationOfState()
    ρ = data.profile.ρ.unscaled[:, index]
    T = data.profile.T.unscaled[:, index]
    S = data.profile.S.unscaled[:, index]

    Δρ = maximum(ρ) - minimum(ρ)
    ΔT = maximum(T) - minimum(T)
    ΔS = maximum(S) - minimum(S)

    α = mean(SeawaterPolynomials.thermal_expansion.(T, S, 0, Ref(eos)))
    β = mean(SeawaterPolynomials.haline_contraction.(T, S, 0, Ref(eos)))
    ρ₀ = eos.reference_density

    T_contribution = α * ΔT * ρ₀
    S_contribution = β * ΔS * ρ₀

    return (; T=T_contribution, S=S_contribution, ρ=Δρ)
end