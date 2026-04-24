using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function run_test_in_project(script, project; env=Pair{String, String}[])
    command = `$(Base.julia_cmd()) --project=$project $script`
    return success(setenv(command, env...))
end

# Get list of tests to run from environment variable (all by default)
const TEST_FILTER = split(get(ENV, "TEST_FILTER", "all"), ",")
const RUN_ALL_TESTS = "all" in TEST_FILTER

if RUN_ALL_TESTS || "oceanstationpapa" in TEST_FILTER
    @testset "OceanStationPapa" begin
        project = joinpath(REPO_ROOT, "OceanStationPapa")
        test_file = joinpath(@__DIR__, "OceanStationPapa", "runtests.jl")
        @test run_test_in_project(test_file, project)
    end
end

if RUN_ALL_TESTS || "columnmodel" in TEST_FILTER
    @testset "Inference: column model" begin
        inference_project = joinpath(REPO_ROOT, "NORiImplementation")
        column_script = joinpath(REPO_ROOT, "inference", "column_model_nori_closures_example.jl")

        common_env = [
            "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", "1"),
        ]
        function run_test_in_project(script, project; env=Pair{String, String}[])
            # Instantiate the project first
            instantiate_cmd = `$(Base.julia_cmd()) --project=$project -e 'using Pkg; Pkg.instantiate()'`
            !success(instantiate_cmd) && error("Failed to instantiate project: $project")
    
            # Then run the script (which may handle its own package development)
            command = `$(Base.julia_cmd()) --project=$project $script`
            return success(setenv(command, env...))
        end

if RUN_ALL_TESTS || "doublegyre" in TEST_FILTER
    @testset "Inference: double gyre" begin
        inference_project = joinpath(REPO_ROOT, "NORiImplementation")
        doublegyre_script = joinpath(REPO_ROOT, "inference", "doublegyre_nori_closures_example.jl")
        doublegyre_config = joinpath(@__DIR__, "configs", "doublegyre_cpu_smoke.toml")

        common_env = [
            "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", "1"),
        ]

        @test run_test_in_project(doublegyre_script, inference_project; env=[
            common_env...,
            "NORI_DOUBLEGYRE_TEST" => doublegyre_config,
        ])
    end
end
