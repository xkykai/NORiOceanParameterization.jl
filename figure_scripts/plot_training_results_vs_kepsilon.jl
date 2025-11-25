using CairoMakie
using JLD2
using Makie
using NORiOceanParameterization
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using SeawaterPolynomials.TEOS10

#####
##### Configuration
#####

# Physical constants
const ρ₀ = TEOS10EquationOfState().reference_density
colors = Makie.wong_colors();

# Directories for model results
PHYSICALCLOSURE_DIR = "./figure_data/k_epsilon_inference_results"  # Can also use CATKE results
NN_DIR = "./figure_data/NN_inference_results"

# Load LES data and scaling parameters
SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
cases = readdir(data_path)

nn_model_path = joinpath(pwd(), "calibrated_parameters", "NNclosure_weights.jld2")
scaling_params = jldopen(nn_model_path, "r") do file
    return file["scaling"]
end
scaling = construct_zeromeanunitvariance_scaling(scaling_params)

#####
##### Select cases to compare
#####

result_names = ["winds_07", "freeconvection_20", "windsandconvection_53", 
                "windsandconvection_42", "windsandconvection_38", "windsandconvection_44"]
result_titles = ["Pure wind\n(with rotation)", "Free convection\n(cooling)", 
                 "Strong wind weak convection\n(strong rotation)", "Strong wind strong convection\n(strong rotation)", 
                 "Weak wind strong convection\n(with rotation)", "Weak wind strong convection\n(no rotation)"]

# Load LES datasets (skip initial spinup period)
initial_timeframe = 25
field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in result_names]
full_timeframes = [initial_timeframe:length(data["ubar"].times) for data in field_datasets]
dataset = LESDatasets(field_datasets, scaling, full_timeframes)

#####
##### Helper functions
#####

b_from_ρ(ρ) = -g_Earth * (ρ - ρ₀) / ρ₀

function read_columnmodel_dataset(filepath, initial_timeframe)
    """Read column model output and extract profiles starting from initial_timeframe"""
    file = jldopen(filepath, "r")
    time_indices = keys(file["timeseries/t"])[initial_timeframe:end]
    Nt = length(time_indices)
    Nz = file["grid/Nz"]
    
    # Preallocate arrays
    u = zeros(Nz, Nt)
    v = zeros(Nz, Nt)
    T = zeros(Nz, Nt)
    S = zeros(Nz, Nt)
    ρ = zeros(Nz, Nt)

    # Extract time series data
    for (i, t) in enumerate(time_indices)
        u[:, i] .= file["timeseries/u/$t"][1, 1, 1:Nz]
        v[:, i] .= file["timeseries/v/$t"][1, 1, 1:Nz]
        T[:, i] .= file["timeseries/T/$t"][1, 1, 1:Nz]
        S[:, i] .= file["timeseries/S/$t"][1, 1, 1:Nz]
        ρ[:, i] .= file["timeseries/ρ/$t"][1, 1, 1:Nz]
    end

    b = b_from_ρ.(ρ)
    close(file)

    return (u=u, v=v, T=T, S=S, ρ=ρ, b=b)
end

#####
##### Load column model results
#####

nn_datasets = [read_columnmodel_dataset(joinpath(NN_DIR, "$(name).jld2"), initial_timeframe) for name in result_names]
physicalclosure_datasets = [read_columnmodel_dataset(joinpath(PHYSICALCLOSURE_DIR, "$(name).jld2"), initial_timeframe) for name in result_names]

#####
##### Plotting setup
#####

zC = -252:8:0
z_indices = length(zC)-19:length(zC)  # Focus on upper ocean
timeframe = 264

# Line style configuration
initial_linestyle = :dash
initial_linewidth = 5
NDE_linestyle = :solid
LES_linewidth = 15
LES_color = (colors[3], 0.5)

#####
##### Prepare buoyancy data (normalized by surface value)
#####

bs_initial = [data.b[z_indices, 1] for data in nn_datasets]
bs_top = [b[end] for b in bs_initial]
bs_initial = [b .- b_top for (b, b_top) in zip(bs_initial, bs_top)]

bs_LES = [b_from_ρ.(data.profile.ρ.unscaled[z_indices, timeframe]) for data in dataset.data]
bs_LES = [b .- b_top for (b, b_top) in zip(bs_LES, bs_top)]

bs_physicalclosure = [data.b[z_indices, timeframe] for data in physicalclosure_datasets]
bs_physicalclosure = [b .- b_top for (b, b_top) in zip(bs_physicalclosure, bs_top)]

bs_nn = [data.b[z_indices, timeframe] for data in nn_datasets]
bs_nn = [b .- b_top for (b, b_top) in zip(bs_nn, bs_top)]

#####
##### Create figure
#####

with_theme(theme_latexfonts()) do
    fig = Figure(size = (2000, 1600), fontsize=25)
    
    # Create axes
    axT1 = Axis(fig[1, 1], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[1], titlesize=plot_titlesize)
    axT2 = Axis(fig[1, 2], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[2], titlesize=plot_titlesize)
    axT3 = Axis(fig[1, 3], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[3], titlesize=plot_titlesize)
    axT4 = Axis(fig[1, 4], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[4], titlesize=plot_titlesize)
    axT5 = Axis(fig[1, 5], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[5], titlesize=plot_titlesize)
    axT6 = Axis(fig[1, 6], xlabel = L"Temperature ($\degree$C)", ylabel = "z (m)", title = result_titles[6], titlesize=plot_titlesize)
    Label(fig[2, :], L"Temperature ($\degree$C)")

    axS1 = Axis(fig[3, 1], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS2 = Axis(fig[3, 2], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS3 = Axis(fig[3, 3], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS4 = Axis(fig[3, 4], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS5 = Axis(fig[3, 5], xlabel = "Salinity (psu)", ylabel = "z (m)")
    axS6 = Axis(fig[3, 6], xlabel = "Salinity (psu)", ylabel = "z (m)", xticks=LinearTicks(3))
    Label(fig[4, :], "Salinity (psu)")

    axb1 = Axis(fig[5, 1], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(3))
    axb2 = Axis(fig[5, 2], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(3))
    axb3 = Axis(fig[5, 3], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(3))
    axb4 = Axis(fig[5, 4], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(3))
    axb5 = Axis(fig[5, 5], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=LinearTicks(3))
    axb6 = Axis(fig[5, 6], xlabel = L"Buoyancy (m s$^{-2}$)", ylabel = "z (m)", xticks=[-2e-3, 0])
    Label(fig[6, :], L"Buoyancy (m s$^{-2}$)")

    axTs = [axT1, axT2, axT3, axT4, axT5, axT6]
    axSs = [axS1, axS2, axS3, axS4, axS5, axS6]
    axbs = [axb1, axb2, axb3, axb4, axb5, axb6]

    # Plot initial conditions
    for (i, ax) in enumerate(axTs)
        label = i == 1 ? "Initial stratification" : nothing
        lines!(ax, nn_datasets[i].T[z_indices, 1], zC[z_indices], color=colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth, label=label)
    end
    for (i, ax) in enumerate(axSs)
        lines!(ax, nn_datasets[i].S[z_indices, 1], zC[z_indices], color=colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    end
    for (i, ax) in enumerate(axbs)
        lines!(ax, bs_initial[i], zC[z_indices], color=colors[1], linestyle=initial_linestyle, linewidth=initial_linewidth)
    end

    # Plot LES
    for (i, ax) in enumerate(axTs)
        label = i == 1 ? "Large-eddy simulation" : nothing
        lines!(ax, dataset.data[i].profile.T.unscaled[z_indices, timeframe], zC[z_indices], color=LES_color, linewidth=LES_linewidth, label=label)
    end
    for (i, ax) in enumerate(axSs)
        lines!(ax, dataset.data[i].profile.S.unscaled[z_indices, timeframe], zC[z_indices], color=LES_color, linewidth=LES_linewidth)
    end
    for (i, ax) in enumerate(axbs)
        lines!(ax, bs_LES[i], zC[z_indices], color=LES_color, linewidth=LES_linewidth)
    end

    # Plot k-epsilon closure
    for (i, ax) in enumerate(axTs)
        label = i == 1 ? "k-ϵ closure" : nothing
        lines!(ax, physicalclosure_datasets[i].T[z_indices, timeframe], zC[z_indices], color=colors[6], linestyle=NDE_linestyle, linewidth=initial_linewidth, label=label)
    end
    for (i, ax) in enumerate(axSs)
        lines!(ax, physicalclosure_datasets[i].S[z_indices, timeframe], zC[z_indices], color=colors[6], linestyle=NDE_linestyle, linewidth=initial_linewidth)
    end
    for (i, ax) in enumerate(axbs)
        lines!(ax, bs_physicalclosure[i], zC[z_indices], color=colors[6], linestyle=NDE_linestyle, linewidth=initial_linewidth)
    end
    
    # Plot NORi closure
    for (i, ax) in enumerate(axTs)
        label = i == 1 ? "NORi closure" : nothing
        lines!(ax, nn_datasets[i].T[z_indices, timeframe], zC[z_indices], color=:black, linestyle=NDE_linestyle, linewidth=initial_linewidth, label=label)
    end
    for (i, ax) in enumerate(axSs)
        lines!(ax, nn_datasets[i].S[z_indices, timeframe], zC[z_indices], color=:black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    end
    for (i, ax) in enumerate(axbs)
        lines!(ax, bs_nn[i], zC[z_indices], color=:black, linestyle=NDE_linestyle, linewidth=initial_linewidth)
    end

    # Configure axes
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

    Legend(fig[7, :], axT1, orientation=:horizontal, patchsize=(50, 20))

    display(fig)
    # save("./paper_figures/results_vskepsilon.pdf", fig)
end
