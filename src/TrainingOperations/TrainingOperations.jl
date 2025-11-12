module TrainingOperations

export
    compute_density_contribution,
    BaseClosureMode, NNMode,
    solve_NDE, diagnose_fields,
    individual_loss, loss, compute_loss_prefactor_density_contribution, loss_multipleics,
    predict_residual_flux, predict_residual_flux_dimensional

abstract type TrainingMode end

struct BaseClosureMode <: TrainingMode end
struct NNMode <: TrainingMode end

include("loss_operations.jl")
include("baseclosure_NDE_and_losses.jl")
include("NN_NDE_and_losses.jl")

end