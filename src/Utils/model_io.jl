"""
    Version-agnostic save/load utilities for neural network weights.

Saves and loads NN weights as plain `Dict{String, Array}` to avoid
serializing Lux-internal types that break across Lux version upgrades.
Model architecture is stored as a simple metadata dictionary and
reconstructed from code on load.
"""

using Lux
using JLD2
using NNlib: relu, tanh_fast, sigmoid, gelu
using Random
using ComponentArrays

# Map activation function names to functions and vice versa
const ACTIVATION_REGISTRY = Dict{String, Any}(
    "relu"      => relu,
    "tanh_fast" => tanh_fast,
    "tanh"      => tanh,
    "sigmoid"   => sigmoid,
    "gelu"      => gelu,
    "identity"  => identity,
)

const ACTIVATION_NAME = Dict{Any, String}(v => k for (k, v) in ACTIVATION_REGISTRY)

"""
    save_nn_weights(path; ps, scaling_params, model_config, sts_data=nothing)

Save neural network weights in a version-agnostic format.

All data is stored as plain `Dict`s and `Array`s — no Lux types are serialized.

# Arguments
- `path`: Output file path (JLD2)
- `ps`: ComponentArray or NamedTuple of parameters (will be flattened to Dict{String, Array})
- `scaling_params`: Scaling parameters (NamedTuple of NamedTuples with μ, σ fields)
- `model_config`: Dict describing the NN architecture
- `sts_data`: Optional Lux states (will be converted to plain Dict)
"""
function save_nn_weights(path; ps, scaling_params, model_config, sts_data=nothing)
    # Convert parameters to Dict{String, Dict{String, Array}}
    ps_dict = _params_to_dict(ps)

    # Convert scaling to plain Dict
    scaling_dict = _scaling_to_dict(scaling_params)

    # Convert sts to plain Dict (if provided)
    sts_dict = isnothing(sts_data) ? Dict{String, Any}() : _sts_to_dict(sts_data)

    jldsave(path;
        ps          = ps_dict,
        scaling     = scaling_dict,
        model_config = model_config,
        sts         = sts_dict,
        format_version = 2,  # Version tag so we can distinguish old vs new files
    )
end

"""
    load_nn_weights(path, dev)

Load neural network weights from a version-agnostic JLD2 file.

Reconstructs Lux models from the stored `model_config` and populates them
with the stored weight arrays. Returns the same types expected by
`NORiNNFluxClosure`: parameters as ComponentArray, Lux model objects, scaling, and states.

# Arguments
- `path`: Path to JLD2 file
- `dev`: Lux device (e.g., `cpu_device()` or `gpu_device()`)

# Returns
`(ps, sts, scaling_params, wT_model, wS_model)` — same signature as the old loader.
"""
function load_nn_weights(path, dev)
    file = jldopen(path, "r")
    try
        format_version = haskey(file, "format_version") ? file["format_version"] : 1

        if format_version >= 2
            return _load_v2(file, dev)
        else
            return _load_v1(file, dev)
        end
    finally
        close(file)
    end
end

"""
    _load_v1(file, dev)

Load from the legacy format (Lux objects serialized directly).
This is the fallback for old files that haven't been migrated yet.
"""
function _load_v1(file, dev)
    ps = file["u"] |> dev |> Lux.f64
    sts = file["sts"] |> dev |> Lux.f64
    scaling_params = file["scaling"]
    wT_model = file["model"].wT
    wS_model = file["model"].wS
    return ps, sts, scaling_params, wT_model, wS_model
end

"""
    _load_v2(file, dev)

Load from the version-agnostic format (plain Dicts + model_config).
"""
function _load_v2(file, dev)
    ps_dict = file["ps"]
    scaling_dict = file["scaling"]
    model_config = file["model_config"]
    sts_dict = file["sts"]

    # Reconstruct Lux models from config
    wT_model = _build_nn_from_config(model_config)
    wS_model = _build_nn_from_config(model_config)

    # Reconstruct ComponentArray parameters
    ps = _dict_to_params(ps_dict, wT_model, wS_model) |> dev |> Lux.f64

    # Reconstruct states
    sts = _dict_to_sts(sts_dict, wT_model, wS_model) |> dev |> Lux.f64

    # Reconstruct scaling (returns NamedTuple of NamedTuples with μ, σ)
    scaling_params = _dict_to_scaling(scaling_dict)

    return ps, sts, scaling_params, wT_model, wS_model
end

"""
    _build_nn_from_config(config)

Reconstruct a Lux Chain from a plain configuration dictionary.
"""
function _build_nn_from_config(config)
    activation = get(ACTIVATION_REGISTRY, config["activation"], nothing)
    isnothing(activation) && error("Unknown activation function: $(config["activation"]). " *
        "Register it in ACTIVATION_REGISTRY.")

    input_size = config["input_size"]
    hidden_size = config["hidden_layer_size"]
    n_hidden = config["n_hidden_layers"]
    output_size = config["output_size"]

    layers = vcat(
        Dense(input_size, hidden_size, activation),
        [Dense(hidden_size, hidden_size, activation) for _ in 1:n_hidden-1]...,
        Dense(hidden_size, output_size),
    )
    return Chain(layers...)
end

"""
    make_model_config(; input_size, hidden_layer_size, n_hidden_layers, output_size, activation)

Create a model configuration dictionary for saving.

# Example
```julia
config = make_model_config(
    input_size       = 21,
    hidden_layer_size = 128,
    n_hidden_layers  = 3,
    output_size      = 1,
    activation       = relu,
)
```
"""
function make_model_config(; input_size, hidden_layer_size, n_hidden_layers, output_size, activation)
    activation_name = get(ACTIVATION_NAME, activation, nothing)
    isnothing(activation_name) && error("Unknown activation function. Register it in ACTIVATION_NAME.")

    return Dict{String, Any}(
        "input_size"        => input_size,
        "hidden_layer_size" => hidden_layer_size,
        "n_hidden_layers"   => n_hidden_layers,
        "output_size"       => output_size,
        "activation"        => activation_name,
    )
end

# ── Internal conversion helpers ──────────────────────────────────────────────

"""Convert ComponentArray ps (with .wT, .wS sub-arrays) to Dict of Dicts of plain Arrays."""
function _params_to_dict(ps)
    result = Dict{String, Any}()
    for net_name in (:wT, :wS)
        net_ps = getproperty(ps, net_name)
        layer_dict = Dict{String, Any}()
        for layer_name in propertynames(net_ps)
            layer_ps = getproperty(net_ps, layer_name)
            param_dict = Dict{String, Any}()
            for param_name in propertynames(layer_ps)
                param_dict[string(param_name)] = Array(getproperty(layer_ps, param_name))
            end
            layer_dict[string(layer_name)] = param_dict
        end
        result[string(net_name)] = layer_dict
    end
    return result
end

"""Convert Dict back to ComponentArray with the same structure Lux expects."""
function _dict_to_params(ps_dict, wT_model, wS_model)
    rng = Random.default_rng()

    # Use Lux.setup to get the correct parameter structure, then fill in values
    ps_wT_template, _ = Lux.setup(rng, wT_model)
    ps_wS_template, _ = Lux.setup(rng, wS_model)

    ps_wT = _fill_params_from_dict(ps_wT_template, ps_dict["wT"])
    ps_wS = _fill_params_from_dict(ps_wS_template, ps_dict["wS"])

    ps_wT_ca = ps_wT |> ComponentArray .|> Float64
    ps_wS_ca = ps_wS |> ComponentArray .|> Float64

    return ComponentArray(; wT=ps_wT_ca, wS=ps_wS_ca)
end

"""Recursively fill a Lux parameter NamedTuple from a Dict of Arrays."""
function _fill_params_from_dict(template::NamedTuple, dict::Dict)
    pairs = []
    for key in keys(template)
        key_str = string(key)
        val = template[key]
        if val isa NamedTuple
            push!(pairs, key => _fill_params_from_dict(val, dict[key_str]))
        else
            push!(pairs, key => reshape(collect(dict[key_str]), size(val)))
        end
    end
    return NamedTuple(pairs)
end

"""Convert Lux states NamedTuple to plain Dict."""
function _sts_to_dict(sts)
    result = Dict{String, Any}()
    for net_name in (:wT, :wS)
        net_st = getproperty(sts, net_name)
        layer_dict = Dict{String, Any}()
        for layer_name in propertynames(net_st)
            layer_st = getproperty(net_st, layer_name)
            # Each layer state is typically a NamedTuple (possibly empty)
            param_dict = Dict{String, Any}()
            if layer_st isa NamedTuple
                for pname in propertynames(layer_st)
                    val = getproperty(layer_st, pname)
                    param_dict[string(pname)] = val isa AbstractArray ? Array(val) : val
                end
            end
            layer_dict[string(layer_name)] = param_dict
        end
        result[string(net_name)] = layer_dict
    end
    return result
end

"""Convert Dict back to Lux states NamedTuple."""
function _dict_to_sts(sts_dict, wT_model, wS_model)
    rng = Random.default_rng()
    _, st_wT_template = Lux.setup(rng, wT_model)
    _, st_wS_template = Lux.setup(rng, wS_model)

    # For Dense layers, states are typically empty NamedTuples, so the template is fine
    # But we fill from dict in case there are non-empty states
    st_wT = isempty(sts_dict) ? st_wT_template : _fill_params_from_dict(st_wT_template, get(sts_dict, "wT", Dict()))
    st_wS = isempty(sts_dict) ? st_wS_template : _fill_params_from_dict(st_wS_template, get(sts_dict, "wS", Dict()))

    return (wT=st_wT, wS=st_wS)
end

"""Convert scaling NamedTuple to plain Dict."""
function _scaling_to_dict(scaling_params)
    result = Dict{String, Any}()
    for key in keys(scaling_params)
        inner = scaling_params[key]
        inner_dict = Dict{String, Any}()
        for field in propertynames(inner)
            inner_dict[string(field)] = getproperty(inner, field)
        end
        result[string(key)] = inner_dict
    end
    return result
end

"""Convert plain Dict back to scaling NamedTuple (same format as write_scaling_params output)."""
function _dict_to_scaling(scaling_dict)
    pairs = []
    for (key, inner_dict) in scaling_dict
        inner_pairs = [Symbol(k) => v for (k, v) in inner_dict]
        push!(pairs, Symbol(key) => NamedTuple(inner_pairs))
    end
    return NamedTuple(pairs)
end

"""
    save_baseclosure_params(path; ps)

Save base closure parameters in a version-agnostic format.

# Arguments
- `path`: Output file path (JLD2)
- `ps`: ComponentArray of base closure parameters
"""
function save_baseclosure_params(path; ps)
    params_dict = Dict{String, Float64}()
    for key in propertynames(ps)
        params_dict[string(key)] = Float64(getproperty(ps, key))
    end
    jldsave(path; u=params_dict, format_version=2)
end

"""
    load_baseclosure_params(path)

Load base closure parameters from JLD2 (supports both old and new format).

Returns a NamedTuple with the parameter values.
"""
function load_baseclosure_params(path)
    file = jldopen(path, "r")
    try
        format_version = haskey(file, "format_version") ? file["format_version"] : 1
        u = file["u"]

        if format_version >= 2
            # New format: Dict{String, Float64}
            return NamedTuple(Symbol(k) => v for (k, v) in u)
        else
            # Old format: ComponentArray — just access fields directly
            return u
        end
    finally
        close(file)
    end
end
