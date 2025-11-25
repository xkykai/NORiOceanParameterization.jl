using CairoMakie
using JLD2
using Makie
using NORiOceanParameterization
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using SeawaterPolynomials.TEOS10

#####
##### Physical constants
#####

const ρ₀ = TEOS10EquationOfState().reference_density
colors = Makie.wong_colors()

#####
##### Load LES data and scaling parameters
#####

SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
cases = readdir(data_path)
field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in cases]

nn_model_path = joinpath(pwd(), "calibrated_parameters", "NNclosure_weights.jld2")
scaling_params = jldopen(nn_model_path, "r") do file
    return file["scaling"]
end

scaling = construct_zeromeanunitvariance_scaling(scaling_params)
full_timeframes = [25:length(data["ubar"].times) for data in field_datasets]
dataset = LESDatasets(field_datasets, scaling, full_timeframes)

#####
##### Load inference results
#####

FILE_DIR = "./figure_data/NN_inference_results.jld2"
results = jldopen(FILE_DIR, "r")

#####
##### Select cases for visualization
#####

# Training cases - used during neural network training
result_names = ["windsandconvection_03", "freeconvection_15", "windsandconvection_15"]
result_indices = [findfirst(x -> x == name, cases) for name in result_names]
result_titles = ["Strong wind strong cooling", "Free convection", "Weak wind strong evaporation"]
result_subtitles = ["Midlatitude Atlantic", "Southern Ocean", "Equatorial Pacific"]

# Validation cases - held out during training
result_names_validation = ["windsandconvection_38", "windsandconvection_51", "windsandconvection_40"]
result_indices_validation = [findfirst(x -> x == name, cases) for name in result_names_validation]
result_titles_validation = ["Weak wind strong convection", "Strong wind weak convection", "Moderate wind weak convection"]
result_subtitles_validation = ["Cooling + evaporation", "Cooling + evaporation", "Cooling + evaporation"]

#####
##### Plotting configuration
#####

coarse_size = 32
zC = -252:8:0

initial_linestyle = :dash
initial_linewidth = 5
NDE_linestyle = :solid
LES_linewidth = 10
LES_color = (colors[3], 0.5)
plot_titlesize = 20
plot_subtitlesize = 20

# Helper functions
b_from_ρ(ρ) = -g_Earth * (ρ - ρ₀) / ρ₀

#####
##### Extract data at specific timeframe
#####

timeframe = 255

# Training cases - buoyancy profiles normalized by surface value
bs_initial = [b_from_ρ.(results["training/$(name)"].sols_dimensional.ρ[:, 1]) for name in result_names]
bs_top = [b[end] for b in bs_initial]
bs_initial = [bs_initial[i] .- bs_top[i] for i in 1:length(bs_initial)]

bs = [b_from_ρ.(results["training/$(name)"].sols_dimensional.ρ[:, timeframe]) for name in result_names]
bs = [bs[i] .- bs_top[i] for i in 1:length(bs)]

bs_noNN = [b_from_ρ.(results["training/$(name)"].sols_dimensional_noNN.ρ[:, timeframe]) for name in result_names]
bs_noNN = [bs_noNN[i] .- bs_top[i] for i in 1:length(bs_noNN)]

bs_LES = [b_from_ρ.(dataset.data[index].profile.ρ.unscaled[:, timeframe]) for index in result_indices]
bs_LES = [bs_LES[i] .- bs_top[i] for i in 1:length(bs_LES)]

# Validation cases - buoyancy profiles normalized by surface value
bs_initial_validation = [b_from_ρ.(results["validation/$(name)"].sols_dimensional.ρ[:, 1]) for name in result_names_validation]
bs_top_validation = [b[end] for b in bs_initial_validation]
bs_initial_validation = [bs_initial_validation[i] .- bs_top_validation[i] for i in 1:length(bs_initial_validation)]

bs_validation = [b_from_ρ.(results["validation/$(name)"].sols_dimensional.ρ[:, timeframe]) for name in result_names_validation]
bs_validation = [bs_validation[i] .- bs_top_validation[i] for i in 1:length(bs_validation)]

bs_noNN_validation = [b_from_ρ.(results["validation/$(name)"].sols_dimensional_noNN.ρ[:, timeframe]) for name in result_names_validation]
bs_noNN_validation = [bs_noNN_validation[i] .- bs_top_validation[i] for i in 1:length(bs_noNN_validation)]

bs_LES_validation = [b_from_ρ.(dataset.data[index].profile.ρ.unscaled[:, timeframe]) for index in result_indices_validation]
bs_LES_validation = [bs_LES_validation[i] .- bs_top_validation[i] for i in 1:length(bs_LES_validation)]

#####
##### Create figure
#####

with_theme(theme_latexfonts()) do
    fig = Figure(size = (2000, 1600), fontsize=25)
    
    # Create 3x6 grid of axes (T, S, b rows; 3 training + 3 validation columns)
    axT1 = Axis(fig[1, 1], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[1], titlesize=plot_titlesize, subtitle=result_subtitles[1], subtitlesize=plot_subtitlesize)
    axT2 = Axis(fig[1, 2], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[2], titlesize=plot_titlesize, subtitle=result_subtitles[2], subtitlesize=plot_subtitlesize)
    axT3 = Axis(fig[1, 3], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[3], titlesize=plot_titlesize, subtitle=result_subtitles[3], subtitlesize=plot_subtitlesize)
    axT4 = Axis(fig[1, 4], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles_validation[1], titlesize=plot_titlesize, subtitle=result_subtitles_validation[1], subtitlesize=plot_subtitlesize)
    axT5 = Axis(fig[1, 5], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles_validation[2], titlesize=plot_titlesize, subtitle=result_subtitles_validation[2], subtitlesize=plot_subtitlesize)
    axT6 = Axis(fig[1, 6], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles_validation[3], titlesize=plot_titlesize, subtitle=result_subtitles_validation[3], subtitlesize=plot_subtitlesize)

    Label(fig[2, :], L"Temperature ($\degree$C)")

    axS1 = Axis(fig[3, 1], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS2 = Axis(fig[3, 2], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS3 = Axis(fig[3, 3], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS4 = Axis(fig[3, 4], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS5 = Axis(fig[3, 5], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS6 = Axis(fig[3, 6], xlabel = "Salinity (psu)", ylabel = "z (m)", xticks=LinearTicks(4))

    Label(fig[4, :], "Salinity (psu)")

    axb1 = Axis(fig[5, 1], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(2))
    axb2 = Axis(fig[5, 2], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(2))
    axb3 = Axis(fig[5, 3], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(2))
    axb4 = Axis(fig[5, 4], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(2))
    axb5 = Axis(fig[5, 5], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(2))
    axb6 = Axis(fig[5, 6], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(2))

    Label(fig[6, :], L"Buoyancy (m s$^{-2}$)")

    #####
    ##### Plot initial conditions
    #####
    
    lines!(axT1, results["training/$(result_names[1])"].sols_dimensional.T[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth, label="Initial stratification")
    lines!(axT2, results["training/$(result_names[2])"].sols_dimensional.T[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axT3, results["training/$(result_names[3])"].sols_dimensional.T[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axT4, results["validation/$(result_names_validation[1])"].sols_dimensional.T[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axT5, results["validation/$(result_names_validation[2])"].sols_dimensional.T[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axT6, results["validation/$(result_names_validation[3])"].sols_dimensional.T[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)

    lines!(axS1, results["training/$(result_names[1])"].sols_dimensional.S[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axS2, results["training/$(result_names[2])"].sols_dimensional.S[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axS3, results["training/$(result_names[3])"].sols_dimensional.S[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axS4, results["validation/$(result_names_validation[1])"].sols_dimensional.S[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axS5, results["validation/$(result_names_validation[2])"].sols_dimensional.S[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axS6, results["validation/$(result_names_validation[3])"].sols_dimensional.S[:, 1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)

    lines!(axb1, bs_initial[1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axb2, bs_initial[2], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axb3, bs_initial[3], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axb4, bs_initial_validation[1], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axb5, bs_initial_validation[2], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    lines!(axb6, bs_initial_validation[3], zC, color = colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)

    #####
    ##### Plot LES reference solutions
    #####
    
    lines!(axT1, dataset.data[result_indices[1]].profile.T.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth, label="Large eddy simulation")
    lines!(axT2, dataset.data[result_indices[2]].profile.T.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axT3, dataset.data[result_indices[3]].profile.T.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axT4, dataset.data[result_indices_validation[1]].profile.T.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axT5, dataset.data[result_indices_validation[2]].profile.T.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axT6, dataset.data[result_indices_validation[3]].profile.T.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)

    lines!(axS1, dataset.data[result_indices[1]].profile.S.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axS2, dataset.data[result_indices[2]].profile.S.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axS3, dataset.data[result_indices[3]].profile.S.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axS4, dataset.data[result_indices_validation[1]].profile.S.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axS5, dataset.data[result_indices_validation[2]].profile.S.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axS6, dataset.data[result_indices_validation[3]].profile.S.unscaled[:, timeframe], zC, color=LES_color, linewidth=LES_linewidth)

    lines!(axb1, bs_LES[1], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axb2, bs_LES[2], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axb3, bs_LES[3], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axb4, bs_LES_validation[1], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axb5, bs_LES_validation[2], zC, color=LES_color, linewidth=LES_linewidth)
    lines!(axb6, bs_LES_validation[3], zC, color=LES_color, linewidth=LES_linewidth)

    #####
    ##### Plot base closure only results
    #####
    
    lines!(axT1, results["training/$(result_names[1])"].sols_dimensional_noNN.T[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth, label="Base closure only")
    lines!(axT2, results["training/$(result_names[2])"].sols_dimensional_noNN.T[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axT3, results["training/$(result_names[3])"].sols_dimensional_noNN.T[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axT4, results["validation/$(result_names_validation[1])"].sols_dimensional_noNN.T[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axT5, results["validation/$(result_names_validation[2])"].sols_dimensional_noNN.T[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axT6, results["validation/$(result_names_validation[3])"].sols_dimensional_noNN.T[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)

    lines!(axS1, results["training/$(result_names[1])"].sols_dimensional_noNN.S[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axS2, results["training/$(result_names[2])"].sols_dimensional_noNN.S[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axS3, results["training/$(result_names[3])"].sols_dimensional_noNN.S[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axS4, results["validation/$(result_names_validation[1])"].sols_dimensional_noNN.S[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axS5, results["validation/$(result_names_validation[2])"].sols_dimensional_noNN.S[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axS6, results["validation/$(result_names_validation[3])"].sols_dimensional_noNN.S[:, timeframe], zC, color = colors[2], linewidth=initial_linewidth)

    lines!(axb1, bs_noNN[1], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axb2, bs_noNN[2], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axb3, bs_noNN[3], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axb4, bs_noNN_validation[1], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axb5, bs_noNN_validation[2], zC, color = colors[2], linewidth=initial_linewidth)
    lines!(axb6, bs_noNN_validation[3], zC, color = colors[2], linewidth=initial_linewidth)

    #####
    ##### Plot base closure + neural network results
    #####
    
    lines!(axT1, results["training/$(result_names[1])"].sols_dimensional.T[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth, label="Base closure + neural network")
    lines!(axT2, results["training/$(result_names[2])"].sols_dimensional.T[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axT3, results["training/$(result_names[3])"].sols_dimensional.T[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axT4, results["validation/$(result_names_validation[1])"].sols_dimensional.T[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axT5, results["validation/$(result_names_validation[2])"].sols_dimensional.T[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axT6, results["validation/$(result_names_validation[3])"].sols_dimensional.T[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)

    lines!(axS1, results["training/$(result_names[1])"].sols_dimensional.S[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axS2, results["training/$(result_names[2])"].sols_dimensional.S[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axS3, results["training/$(result_names[3])"].sols_dimensional.S[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axS4, results["validation/$(result_names_validation[1])"].sols_dimensional.S[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axS5, results["validation/$(result_names_validation[2])"].sols_dimensional.S[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axS6, results["validation/$(result_names_validation[3])"].sols_dimensional.S[:, timeframe], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)

    lines!(axb1, bs[1], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axb2, bs[2], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axb3, bs[3], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axb4, bs_validation[1], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axb5, bs_validation[2], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    lines!(axb6, bs_validation[3], zC, color = :black, linestyle=NDE_linestyle, linewidth=initial_linewidth)

    #####
    ##### Format axes
    #####
    
    axTs = [axT1, axT2, axT3, axT4, axT5, axT6]
    axSs = [axS1, axS2, axS3, axS4, axS5, axS6]
    axbs = [axb1, axb2, axb3, axb4, axb5, axb6]
    axs = vcat(axTs, axSs, axbs)
    axs_interior = vcat(axTs[2:end], axSs[2:end], axbs[2:end])

    linkyaxes!(axs...)

    for ax in axs
        hidexdecorations!(ax, ticks=false, ticklabels=false)
        hideydecorations!(ax, ticks=false, ticklabels=false, label=false)
        hidespines!(ax, :t, :r)
    end

    for ax in axs_interior
        hideydecorations!(ax, ticks=false)
    end

    Label(fig[0, 1:3], "Training", fontsize=30, font=:bold)
    Label(fig[0, 4:6], "Validation", fontsize=30, font=:bold)

    Legend(fig[7, :], axT1, orientation=:horizontal, patchsize=(50, 20))

    display(fig)
    # save("./paper_figures/training_results.pdf", fig)
end
#%%
