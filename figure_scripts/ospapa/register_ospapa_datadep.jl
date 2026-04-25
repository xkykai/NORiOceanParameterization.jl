using DataDeps

function __init__()
    # If it doesn't exist, register it

    try
        datadep"ospapa_observations"
    catch
        register(DataDep(
            "ospapa_observations",
            """
            Pre-computed observational data from Ocean Station Papa used in figure plotting
            """,
            "https://zenodo.org/api/records/19753206/files/ospapa_observations.tar.gz/content",

            post_fetch_method = unpack
        ))
    end

    try
        datadep"ospapa_inference"
    catch
        register(DataDep(
            "ospapa_inference",
            """
            Pre-computed inference results for Ocean Station Papa used in figure plotting
            """,
            "https://zenodo.org/api/records/19753206/files/ospapa_inference.tar.gz/content",

            post_fetch_method = unpack
        ))
    end
end

# Call immediately when included
__init__()

# Helper function to get the data path
get_ospapa_observations_data_path() = joinpath(datadep"ospapa_observations", "ospapa")
get_ospapa_inference_data_path() = joinpath(datadep"ospapa_inference", "ospapa")