using TOML

const FIGURE_SCRIPTS_ROOT = joinpath(REPO_ROOT, "figure_scripts")
const PREPARED_FIGURE_PROJECTS = Set{String}()

function discover_figure_scripts(root::AbstractString=FIGURE_SCRIPTS_ROOT)
    scripts = String[]

    for (dir, _, files) in walkdir(root)
        for file in files
            if startswith(file, "plot_") && endswith(file, ".jl")
                push!(scripts, normpath(joinpath(dir, file)))
            end
        end
    end

    return sort(scripts)
end

function nearest_project_dir(path::AbstractString)
    dir = isdir(path) ? normpath(path) : dirname(normpath(path))

    while true
        isfile(joinpath(dir, "Project.toml")) && return dir

        parent = dirname(dir)
        parent == dir && error("No Project.toml found for $path")
        dir = parent
    end
end

function local_develop_paths(project::AbstractString; repo_root::AbstractString=REPO_ROOT)
    project_toml = joinpath(project, "Project.toml")
    project_data = TOML.parsefile(project_toml)
    dependencies = get(project_data, "deps", Dict{String, Any}())

    paths = String[]
    if haskey(dependencies, "NORiOceanParameterization")
        push!(paths, repo_root)
    end
    if haskey(dependencies, "OceanStationPapa")
        push!(paths, joinpath(repo_root, "OceanStationPapa"))
    end

    return paths
end

function prepare_figure_project(project::AbstractString; repo_root::AbstractString=REPO_ROOT)
    project = normpath(project)
    project in PREPARED_FIGURE_PROJECTS && return nothing

    develop_paths = local_develop_paths(project; repo_root)
    prepare_script = """
    using Pkg
    develop_paths = ARGS
    if !isempty(develop_paths)
        Pkg.develop([Pkg.PackageSpec(path=path) for path in develop_paths])
    end
    Pkg.instantiate()
    """

    command = `$(Base.julia_cmd()) --project=$project -e $prepare_script -- $(develop_paths...)`
    success(command) || error("Failed to prepare plotting project: $project")

    push!(PREPARED_FIGURE_PROJECTS, project)
    return nothing
end

function run_figure_script(script::AbstractString; env=Pair{String, String}[])
    project = nearest_project_dir(script)
    prepare_figure_project(project)

    child_env = copy(ENV)
    child_env["JULIA_NUM_THREADS"] = get(ENV, "JULIA_NUM_THREADS", "1")
    child_env["DATADEPS_ALWAYS_ACCEPT"] = get(ENV, "DATADEPS_ALWAYS_ACCEPT", "true")
    for (key, value) in env
        child_env[key] = value
    end

    command = `$(Base.julia_cmd()) --project=$project $script`
    return success(setenv(command, child_env))
end
