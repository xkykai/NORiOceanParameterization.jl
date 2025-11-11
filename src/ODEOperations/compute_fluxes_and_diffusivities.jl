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

"""
    predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps, params)

Compute diffusive fluxes from Richardson numbers and normalized fields.
Returns diffusive components of `uw, vw, wT, wS`.
"""
function predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps, params)
    νs, κs = predict_diffusivities(Ris, ps)

    ∂u∂z_hat = params.Dᶠ_hat * u_hat
    ∂v∂z_hat = params.Dᶠ_hat * v_hat
    ∂T∂z_hat = params.Dᶠ_hat * T_hat
    ∂S∂z_hat = params.Dᶠ_hat * S_hat

    uw_diffusive = -νs .* ∂u∂z_hat
    vw_diffusive = -νs .* ∂v∂z_hat
    wT_diffusive = -κs .* ∂T∂z_hat
    wS_diffusive = -κs .* ∂S∂z_hat
    return uw_diffusive, vw_diffusive, wT_diffusive, wS_diffusive
end

"""
    predict_diffusive_boundary_flux_dimensional(Ris, u_hat, v_hat, T_hat, S_hat, ps, params)

Compute total dimensional fluxes by combining diffusive and boundary flux contributions.
Returns dimensional `uw, vw, wT, wS` flux arrays.
"""
function predict_diffusive_boundary_flux_dimensional(Ris, u_hat, v_hat, T_hat, S_hat, ps, params)
    _uw_diffusive, _vw_diffusive, _wT_diffusive, _wS_diffusive = predict_diffusive_flux(Ris, u_hat, v_hat, T_hat, S_hat, ps, params)
    _uw_boundary, _vw_boundary, _wT_boundary, _wS_boundary = predict_boundary_flux(params)

    uw_diffusive = params.scaling.u.σ / params.H .* _uw_diffusive
    vw_diffusive = params.scaling.v.σ / params.H .* _vw_diffusive
    wT_diffusive = params.scaling.T.σ / params.H .* _wT_diffusive
    wS_diffusive = params.scaling.S.σ / params.H .* _wS_diffusive

    uw_boundary = inv(params.scaling.uw).(_uw_boundary)
    vw_boundary = inv(params.scaling.vw).(_vw_boundary)
    wT_boundary = inv(params.scaling.wT).(_wT_boundary)
    wS_boundary = inv(params.scaling.wS).(_wS_boundary)

    uw = uw_diffusive .+ uw_boundary
    vw = vw_diffusive .+ vw_boundary
    wT = wT_diffusive .+ wT_boundary
    wS = wS_diffusive .+ wS_boundary

    return uw, vw, wT, wS
end