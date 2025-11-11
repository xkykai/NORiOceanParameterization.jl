module ODEOperations

export 
    predict_boundary_flux, predict_boundary_flux!,
    predict_diffusivities, predict_diffusivities!,
    predict_diffusive_flux, predict_diffusive_boundary_flux_dimensional

include("compute_fluxes_and_diffusivities.jl")
end