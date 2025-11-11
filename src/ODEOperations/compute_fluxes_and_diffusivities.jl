using NORiOceanParameterization.Closures: local_Ri_ν, local_Ri_κ

"""
    predict_boundary_flux(params)

Predict boundary fluxes by filling interior points with bottom boundary values
and setting the last point to top boundary values.

Returns `uw, vw, wT, wS` flux arrays.
"""
function predict_boundary_flux(params)
    uw = vcat(fill(params.uw.scaled.bottom, params.coarse_size), params.uw.scaled.top)
    vw = vcat(fill(params.vw.scaled.bottom, params.coarse_size), params.vw.scaled.top)
    wT = vcat(fill(params.wT.scaled.bottom, params.coarse_size), params.wT.scaled.top)
    wS = vcat(fill(params.wS.scaled.bottom, params.coarse_size), params.wS.scaled.top)

    return uw, vw, wT, wS
end

"""
    predict_boundary_flux!(uw, vw, wT, wS, params)

In-place version of `predict_boundary_flux`. Modifies flux arrays directly.
"""
function predict_boundary_flux!(uw, vw, wT, wS, params)
    uw[1:end-1] .= params.uw.scaled.bottom
    vw[1:end-1] .= params.vw.scaled.bottom
    wT[1:end-1] .= params.wT.scaled.bottom
    wS[1:end-1] .= params.wS.scaled.bottom

    uw[end] = params.uw.scaled.top
    vw[end] = params.vw.scaled.top
    wT[end] = params.wT.scaled.top
    wS[end] = params.wS.scaled.top

    return nothing
end

"""
    predict_diffusivities(Ris, ps)

Compute momentum (`νs`) and tracer (`κs`) diffusivities from Richardson numbers
using local Richardson number closures.

Returns `νs, κs`.
"""
function predict_diffusivities(Ris, ps)
    νs = local_Ri_ν.(Ris, ps.ν_conv, ps.ν_shear, ps.Riᶜ, ps.ΔRi)
    κs = local_Ri_κ.(Ris, ps.ν_conv, ps.ν_shear, ps.Riᶜ, ps.ΔRi, ps.Pr_conv, ps.Pr_shear)
    return νs, κs
end

"""
    predict_diffusivities!(νs, κs, Ris, ps)

In-place version of `predict_diffusivities`. Modifies diffusivity arrays directly.
"""
function predict_diffusivities!(νs, κs, Ris, ps)
    νs .= local_Ri_ν.(Ris, ps.ν_conv, ps.ν_shear, ps.Riᶜ, ps.ΔRi)
    κs .= local_Ri_κ.(Ris, ps.ν_conv, ps.ν_shear, ps.Riᶜ, ps.ΔRi, ps.Pr_conv, ps.Pr_shear)
    return nothing
end