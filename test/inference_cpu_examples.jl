using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const INFERENCE_PROJECT = joinpath(REPO_ROOT, "NORiImplementation")

function run_inference_example(script; env=Pair{String, String}[])
    command = `$(Base.julia_cmd()) --project=$INFERENCE_PROJECT $script`
    return success(setenv(command, env...))
end

@testset "Inference examples run on CPU" begin
    column_script = joinpath(REPO_ROOT, "inference", "column_model_nori_closures_example.jl")
    doublegyre_script = joinpath(REPO_ROOT, "inference", "doublegyre_nori_closures_example.jl")
    doublegyre_config = joinpath(REPO_ROOT, "test", "configs", "doublegyre_cpu_smoke.toml")

    common_env = [
        "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", "1"),
    ]

    @testset "column model" begin
        @test run_inference_example(column_script; env=common_env)
    end

    @testset "double gyre" begin
        @test run_inference_example(doublegyre_script; env=[
            common_env...,
            "NORI_DOUBLEGYRE_TEST" => doublegyre_config,
        ])
    end
end
