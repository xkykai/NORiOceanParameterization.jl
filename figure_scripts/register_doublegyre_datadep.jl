using DataDeps

function __init__()
    # Only register if not already registered
    try
        # Try to access the datadep - if it doesn't exist, it will error
        datadep"doublegyre"
    catch
        # If it doesn't exist, register it
        register(DataDep(
            "doublegyre",
            """
            Pre-computed double-gyre data for use in figure plotting
            """,
            "https://zenodo.org/api/records/17605195/files/doublegyre.tar.gz/content",

            post_fetch_method = unpack
        ))
    end
end

# Call immediately when included
__init__()

# Helper function to get the data path
get_doublegyre_data_path() = joinpath(datadep"doublegyre", "doublegyre")