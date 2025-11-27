using DataDeps

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

    register(DataDep(
        "LES_3D_fields",
        """
        Pre-computed 3D snapshots from LES simulations used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/LES_3D_fields.jld2.gz/content",

        post_fetch_method = unpack
    ))

    register(DataDep(
        "losses",
        """
        Training and validation losses from neural network training of NORi used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/losses.tar.gz/content",

        post_fetch_method = unpack
    ))

    register(DataDep(
        "ODE_inference",
        """
        Pre-computed column model inference results run natively within training framework used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/ODE_inference.tar.gz/content",

        post_fetch_method = unpack
    ))

    register(DataDep(
        "oceananigans_column",
        """
        Pre-computed column model inference results run with Oceananigans.jl used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/oceananigans_column.tar.gz/content",

        post_fetch_method = unpack
    ))

    register(DataDep(
        "long_integration",
        """
        Pre-computed long integration data of column models used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/long_integration.tar.gz/content",

        post_fetch_method = unpack
    ))

    register(DataDep(
        "timestep",
        """
        Pre-computed timestep dependence data of column models used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/timestep.tar.gz/content",

        post_fetch_method = unpack
    ))

    register(DataDep(
        "doublegyre",
        """
        Pre-computed double-gyre data used in figure plotting
        """,
        "https://zenodo.org/api/records/17605195/files/doublegyre.tar.gz/content",

        post_fetch_method = unpack
    ))
end

get_SOBLLES_data_path() = datadep"SOBLLES"
get_LES_3D_fields_data_path() = datadep"LES_3D_fields"
get_losses_data_path() = joinpath(datadep"losses", "losses")
get_ODE_inference_data_path() = joinpath(datadep"ODE_inference", "ODE_inference")
get_oceananigans_column_data_path() = joinpath(datadep"oceananigans_column", "oceananigans_column")
get_long_integration_data_path() = joinpath(datadep"long_integration", "long_integration")
get_timestep_data_path() = joinpath(datadep"timestep", "timestep")
get_doublegyre_data_path() = joinpath(datadep"doublegyre", "doublegyre")



