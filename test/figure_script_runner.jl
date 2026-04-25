const FIGURE_SCRIPTS_ROOT = joinpath(REPO_ROOT, "figure_scripts")

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

function run_figure_script(script::AbstractString; env=Pair{String, String}[])
    child_env = copy(ENV)
    child_env["JULIA_NUM_THREADS"] = get(ENV, "JULIA_NUM_THREADS", "1")
    child_env["DATADEPS_ALWAYS_ACCEPT"] = get(ENV, "DATADEPS_ALWAYS_ACCEPT", "true")
    for (key, value) in env
        child_env[key] = value
    end

    command = `$(Base.julia_cmd()) $script`
    @info "Running figure script" script command
    return success(pipeline(setenv(command, child_env); stdout=stdout, stderr=stderr))
end
