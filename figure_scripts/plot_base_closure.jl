using CairoMakie
using JLD2
using ComponentArrays
using ColorSchemes
using Makie
using NORiOceanParameterization
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using SeawaterPolynomials.TEOS10

#####
##### Load calibrated parameters and setup
#####

ps_baseclosure = jldopen(joinpath(@__DIR__, "..", "calibrated_parameters", "baseclosure_final.jld2"), "r")["u"]

# Compute diffusivities as function of Richardson number
Ris = -0.5:0.001:1
νs = [local_Ri_ν(Ri, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.Riᶜ, ps_baseclosure.ΔRi) for Ri in Ris]
κs = [local_Ri_κ(Ri, ps_baseclosure.ν_conv, ps_baseclosure.ν_shear, ps_baseclosure.Riᶜ, ps_baseclosure.ΔRi, ps_baseclosure.Pr_conv, ps_baseclosure.Pr_shear) for Ri in Ris]

#####
##### Load inference results and datasets
#####

SOLS_DIR = joinpath(get_ODE_inference_data_path(), "baseclosure_results.jld2")
sols = jldopen(SOLS_DIR, "r")

SOBLLES_path = joinpath(get_SOBLLES_data_path(), "SOBLLES_jld2")
data_path = joinpath(SOBLLES_path, "data")
plot_cases = ["winds_04", "freeconvection_08", "winds_07", "windsandheating_31"]

field_datasets = [FieldDataset(joinpath(data_path, sim, "instantaneous_timeseries.jld2"), backend=OnDisk()) for sim in plot_cases]

coarse_size = 32
start_timeframe = 25
full_timeframes = [start_timeframe:length(data["ubar"].times) for data in field_datasets]
dataset = LESDatasets(field_datasets, ZeroMeanUnitVarianceScaling, full_timeframes)

#####
##### Plotting configuration
#####

colors = Makie.wong_colors();
zC = -252:8:0  # Vertical coordinates

# Line styles
initial_linestyle = :dash
initial_linewidth = 5
NDE_linestyle = :solid
LES_linewidth = 8

# Physical constants
ρ₀ = TEOS10EquationOfState().reference_density

# Helper functions
b_from_ρ(ρ) = -g_Earth * (ρ - ρ₀) / ρ₀

#####
##### Create figure
#####

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1200, 1800), fontsize=25)
    gclosure = GridLayout(fig[1, 1])
    gresults = GridLayout(fig[2:7, 1])
    
    # Top panel: Diffusivity vs Richardson number
    ax = CairoMakie.Axis(gclosure[1, 1], xlabel="Ri", ylabel="Diffusivity (m² s⁻¹)", yscale=log10)
    
    # Result panels for 4 cases
    axuv1 = CairoMakie.Axis(gresults[1, 1], xlabel=L"Velocities $(\text{m} \, \text{s}^{-1})$", ylabel="z (m)", 
                            titlesize=20, titlefont=:bold, title="Wind (no rotation)")
    axuv2 = CairoMakie.Axis(gresults[1, 2], xlabel=L"Velocities $(\text{m} \, \text{s}^{-1})$", ylabel="z (m)", 
                            titlesize=20, titlefont=:bold, title="Free convection (cooling)", xticks=LinearTicks(3))
    axuv3 = CairoMakie.Axis(gresults[1, 3], xlabel=L"Velocities $(\text{m} \, \text{s}^{-1})$", ylabel="z (m)", 
                            titlesize=20, titlefont=:bold, title="Wind (rotation)", xticks=LinearTicks(3))
    axuv4 = CairoMakie.Axis(gresults[1, 4], xlabel=L"Velocities $(\text{m} \, \text{s}^{-1})$", ylabel="z (m)", 
                            titlesize=20, titlefont=:bold, title="Wind,\nheating + precipitation")
    
    Label(gresults[2, :], L"Velocities $(\text{m} \, \text{s}^{-1})$", tellwidth=false)
    
    # Temperature panels
    axT1 = CairoMakie.Axis(gresults[4, 1], xlabel=L"Temperature ($\degree$C)", ylabel="z (m)")
    axT2 = CairoMakie.Axis(gresults[4, 2], xlabel=L"Temperature ($\degree$C)", ylabel="z (m)")
    axT3 = CairoMakie.Axis(gresults[4, 3], xlabel=L"Temperature ($\degree$C)", ylabel="z (m)")
    axT4 = CairoMakie.Axis(gresults[4, 4], xlabel=L"Temperature ($\degree$C)", ylabel="z (m)")
    Label(gresults[5, :], L"Temperature ($\degree$C)", tellwidth=false)
    
    # Salinity panels
    axS1 = CairoMakie.Axis(gresults[6, 1], xlabel="Salinity (psu)", ylabel="z (m)", xticks=LinearTicks(3))
    axS2 = CairoMakie.Axis(gresults[6, 2], xlabel="Salinity (psu)", ylabel="z (m)")
    axS3 = CairoMakie.Axis(gresults[6, 3], xlabel="Salinity (psu)", ylabel="z (m)")
    axS4 = CairoMakie.Axis(gresults[6, 4], xlabel="Salinity (psu)", ylabel="z (m)")
    Label(gresults[7, :], "Salinity (psu)", tellwidth=false)
    
    # Buoyancy panels
    axσ1 = CairoMakie.Axis(gresults[8, 1], xlabel=L"Buoyancy (m s$^{-2}$)", ylabel="z (m)", xticks=LinearTicks(3))
    axσ2 = CairoMakie.Axis(gresults[8, 2], xlabel=L"Buoyancy (m s$^{-2}$)", ylabel="z (m)", xticks=LinearTicks(2))
    axσ3 = CairoMakie.Axis(gresults[8, 3], xlabel=L"Buoyancy (m s$^{-2}$)", ylabel="z (m)", xticks=LinearTicks(3))
    axσ4 = CairoMakie.Axis(gresults[8, 4], xlabel=L"Buoyancy (m s$^{-2}$)", ylabel="z (m)", xticks=LinearTicks(3))
    Label(gresults[9, :], L"Buoyancy (m s$^{-2}$)", tellwidth=false)

    # Labels for training vs testing
    Label(gresults[0, 1:2], "Training", font=:bold)
    Label(gresults[0, 3:4], "Testing", font=:bold)

    # Group axes for easier iteration
    axuvs = [axuv1, axuv2, axuv3, axuv4]
    axTs = [axT1, axT2, axT3, axT4]
    axSs = [axS1, axS2, axS3, axS4]
    axσs = [axσ1, axσ2, axσ3, axσ4]
    axresults = [axuvs; axTs; axSs; axσs]

    #####
    ##### Plot diffusivity parameterization
    #####
    
    lines!(ax, Ris, νs, label="Viscosity", linewidth=initial_linewidth, color=colors[5])
    lines!(ax, Ris, κs, label="Diffusivity", linewidth=initial_linewidth, color=colors[6])
    axislegend(ax, position=:rt, labelfont=:bold)
    hidedecorations!(ax, ticks=false, ticklabels=false, label=false)
    
    # Mark convection/shear regimes
    vlines!(ax, [0, ps_baseclosure.Riᶜ], color=:black, linestyle=:dash, linewidth=2)
    text!(ax, -0.3, 5e-4, text="Convection-driven\nmixing", align=(:center, :center), font=:bold)
    text!(ax, ps_baseclosure.Riᶜ / 2, 5e-4, text="Shear-driven\nmixing", align=(:center, :center), font=:bold)

    #####
    ##### Plot results for selected cases
    #####

    frame = 250  # Time snapshot to plot

    for (i, (case, axuv, axT, axS, axσ)) in enumerate(zip(plot_cases, axuvs, axTs, axSs, axσs))
        # Compute buoyancy (normalized to surface value)
        b_initial = b_from_ρ.(dataset.data[i].profile.ρ.unscaled[:, 1])
        b_top = b_initial[end]
        b = b_from_ρ.(sols[case].sols_dimensional.ρ[:, frame]) .- b_top
        b_LES = b_from_ρ.(dataset.data[i].profile.ρ.unscaled[:, frame]) .- b_top
        b_initial = b_initial .- b_top

        # Initial stratification
        lines!(axT, dataset.data[i].profile.T.unscaled[:, 1], zC, label="Initial profile", 
               color=colors[4], linestyle=(:dash, :dense), linewidth=initial_linewidth)
        lines!(axS, dataset.data[i].profile.S.unscaled[:, 1], zC, label="Initial profile", 
               color=colors[4], linestyle=(:dash, :dense), linewidth=initial_linewidth)
        lines!(axσ, b_initial, zC, label="Initial profile", 
               color=colors[4], linestyle=(:dash, :dense), linewidth=initial_linewidth)

        # LES results
        lines!(axuv, dataset.data[i].profile.u.unscaled[:, frame], zC, label="u (Large-eddy simulation)", 
               color=(colors[1], 0.5), linewidth=LES_linewidth)
        lines!(axuv, dataset.data[i].profile.v.unscaled[:, frame], zC, label="v (Large-eddy simulation)", 
               color=(colors[2], 0.5), linewidth=LES_linewidth)
        lines!(axT, dataset.data[i].profile.T.unscaled[:, frame], zC, label="Large-eddy simulation", 
               color=(colors[3], 0.5), linewidth=LES_linewidth)
        lines!(axS, dataset.data[i].profile.S.unscaled[:, frame], zC, label="Large-eddy simulation", 
               color=(colors[3], 0.5), linewidth=LES_linewidth)
        lines!(axσ, b_LES, zC, label="LES", color=(colors[3], 0.5), linewidth=LES_linewidth)

        # Base closure results
        lines!(axuv, sols[case].sols_dimensional.u[:, frame], zC, label="u (Base closure)", 
               color=colors[1], linewidth=initial_linewidth)
        lines!(axuv, sols[case].sols_dimensional.v[:, frame], zC, label="v (Base closure)", 
               color=colors[2], linewidth=initial_linewidth)
        lines!(axT, sols[case].sols_dimensional.T[:, frame], zC, label="Base closure", 
               color=colors[3], linewidth=initial_linewidth)
        lines!(axS, sols[case].sols_dimensional.S[:, frame], zC, label="Base closure", 
               color=colors[3], linewidth=initial_linewidth)
        lines!(axσ, b, zC, label="Base closure", color=colors[3], linewidth=initial_linewidth)
    end

    #####
    ##### Format axes
    #####
    
    Legend(gresults[3, :], axuv1, orientation=:horizontal, patchsize=(50, 20), labelsize=22)
    Legend(gresults[10, :], axT1, orientation=:horizontal, patchsize=(50, 20), labelsize=22)
    
    # Hide redundant decorations
    for ax in [axuv1, axT1, axS1, axσ1]
        hideydecorations!(ax, ticks=false, ticklabels=false, label=false)
    end

    for ax in axresults
        hidexdecorations!(ax, ticks=false, ticklabels=false)
        hidespines!(ax, :t, :r)
    end

    for ax in [axuv2, axuv3, axuv4, axT2, axT3, axT4, axS2, axS3, axS4, axσ2, axσ3, axσ4]
        hideydecorations!(ax, ticks=false)
        hidespines!(ax, :t, :r)
    end

    linkyaxes!(axresults...)
    xlims!(axuv2, (-1e-2, 1e-2))
    
    trim!(fig.layout)
    display(fig)
#     save("./figures/localbaseclosure_results.pdf", fig)
end