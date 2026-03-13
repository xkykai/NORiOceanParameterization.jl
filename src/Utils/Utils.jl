module Utils

export 
    find_min, find_max, calculate_Ri, calculate_Ri!,
    save_nn_weights, load_nn_weights, make_model_config,
    save_baseclosure_params, load_baseclosure_params

    include("utilities.jl")
    include("model_io.jl")
end