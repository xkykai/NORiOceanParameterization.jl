module ODEOperations

export 
    predict_boundary_flux, predict_boundary_flux!,
    predict_diffusivities, predict_diffusivities!

include("compute_fluxes_and_diffusivities.jl")
end