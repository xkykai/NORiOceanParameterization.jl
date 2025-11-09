module ODEOperations
export predict_boundary_flux, predict_boundary_flux!, predict_diffusive_flux, predict_diffusive_boundary_flux_dimensional
export solve_NDE

include("compute_fluxes.jl")
end