using DataDeps
using JSON

"""
Register and download LES dataset from Zenodo using DataDeps.
"""
function __init__()
    register(DataDep(
        "SOBLLES",
        """
        A Salty Ocean Boundary Layer Large-Eddy Simulations Dataset
        """,
        "https://zenodo.org/api/records/17677802/files/SOBLLES_jld2.tar/content",
        post_fetch_method = unpack
    ))
end

"""
    get_SOBLLES_data_path()

Get the path to the downloaded LES data. Downloads automatically if not present.
"""
function get_SOBLLES_data_path()
    return datadep"SOBLLES"
end

"""
    load_dataset_split(filepath::String)

Load and parse a dataset split JSON file.

Merges array of dictionaries from JSON into a single dictionary mapping
split names to file lists.

# Arguments
- `filepath::String`: Path to JSON file containing dataset splits

# Returns
Dictionary mapping split names (e.g., "training", "validation") to file lists
"""
function load_dataset_split(filepath::String)
    data = JSON.parsefile(filepath)
    
    # The JSON structure is an array of dictionaries
    # Merge them into a single dictionary
    merged = Dict{String, Vector{String}}()
    for item in data
        merge!(merged, item)
    end
    
    return merged
end

"""
    get_nn_datasets(filepath::String)

Get training and validation datasets from nn_dataset_split.json

# Arguments
- `filepath::String`: Path to nn_dataset_split.json

# Returns
NamedTuple with `training` and `validation` file lists
"""
function get_nn_datasets(filepath::String)
    splits = load_dataset_split(filepath)
    
    training = get(splits, "training", String[])
    validation = get(splits, "validation", String[])
    
    return (training=training, validation=validation)
end

"""
    get_baseclosure_datasets(filepath::String)

Get training and test datasets from baseclosure_dataset_split.json

# Arguments
- `filepath::String`: Path to baseclosure_dataset_split.json

# Returns
NamedTuple with:
- `training`: Combined training files
- `test`: Test files
- `training_shear`: Shear-driven training subset
- `training_convection`: Convection-driven training subset
"""
function get_baseclosure_datasets(filepath::String)
    splits = load_dataset_split(filepath)
    
    training_shear = get(splits, "training_shear", String[])
    training_convection = get(splits, "training_convection", String[])
    test = get(splits, "test", String[])
    
    # Combine all training datasets
    training = vcat(training_shear, training_convection)
    
    return (training=training, test=test, 
            training_shear=training_shear, 
            training_convection=training_convection)
end

