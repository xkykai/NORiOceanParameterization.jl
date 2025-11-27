using CairoMakie
using JLD2
using ComponentArrays
using ColorSchemes
using Makie

#####
##### Load Data
#####

colors = Makie.wong_colors();
SOLS_DIR = joinpath(@__DIR__, "..", "figure_data", "ODE_inference", "NN_results.jld2")

mapping = jldopen("./figure_data/mapping_old_new_simulation_suite.jld2", "r")["mapping"]

casename = "windsandconvection_01"
sol = jldopen(SOLS_DIR, "r")["training/$(casename)"]

# Vertical coordinates for cell centers and faces
zC = -252:8:0
zF = -256:8:0
zF_plot = zF[2:end-1]  # Exclude boundaries for gradient plots

#####
##### Extract Data at Specific Timeframe
#####

timeframe = 110

# Find vertical extent where NN is active (non-zero residual flux)
first_index = findfirst(sol.fluxes.wT.residual[:, timeframe] .!= 0) - 1
last_index = findlast(sol.fluxes.wT.residual[:, timeframe] .!= 0) + 1

# Extract profiles
u = sol.sols_dimensional.u[:, timeframe]
v = sol.sols_dimensional.v[:, timeframe]
T = sol.sols_dimensional.T[:, timeframe]
S = sol.sols_dimensional.S[:, timeframe]
σ = sol.sols_dimensional.ρ[:, timeframe]

# Gradients (excluding boundaries)
∂T∂z = sol.sols_dimensional.∂T∂z[:, timeframe][2:end-1]
∂S∂z = sol.sols_dimensional.∂S∂z[:, timeframe][2:end-1]
∂σ∂z = sol.sols_dimensional.∂ρ∂z[:, timeframe][2:end-1]
Ri = atan.(sol.diffusivities.Ri[:, timeframe][2:end-1])  # arctan transform for visualization

#####
##### Compute Axis Limits with Padding
#####

width_multiplier = 0.1  # Add 10% padding to axis limits

# Helper function for padded limits
function padded_lims(data, multiplier)
    width = diff(collect(extrema(data)))[1]
    return extrema(data) .+ (-multiplier * width, multiplier * width)
end

ulim = padded_lims(u, width_multiplier)
vlim = padded_lims(v, width_multiplier)
vellim = (min(ulim[1], vlim[1]), max(ulim[2], vlim[2]))  # Combined velocity limits
Tlim = padded_lims(T, width_multiplier)
Slim = padded_lims(S, width_multiplier)
σlim = padded_lims(σ, width_multiplier)

∂T∂zlim = padded_lims(∂T∂z, width_multiplier)
∂S∂zlim = padded_lims(∂S∂z, width_multiplier)
∂σ∂zlim = padded_lims(∂σ∂z, width_multiplier)
Rilim = padded_lims(Ri, width_multiplier)

#####
##### Plot Neural Network Inputs
#####

with_theme(theme_latexfonts()) do
    fig = Figure(size=(800, 1500), fontsize=15)
    
    # Left column: state variables
    axT = CairoMakie.Axis(fig[1, 1], xlabel=L"($\degree$C)", ylabel="z (m)", title="Temperature")
    axS = CairoMakie.Axis(fig[2, 1], xlabel="(psu)", ylabel="z (m)", title="Salinity")
    axσ = CairoMakie.Axis(fig[3, 1], xlabel=L"(kg m$^{-3}$)", ylabel="z (m)", title="Potential density")
    axvel = CairoMakie.Axis(fig[4, 1], xlabel=L"$(\text{m} \, \text{s}^{-1})$", ylabel="z (m)", title="Velocities")

    # Right column: gradients and Richardson number
    ax∂T∂z = CairoMakie.Axis(fig[1, 2], xlabel=L"($\degree$C m$^{-1}$)", ylabel="z (m)", title="Temperature gradient")
    ax∂S∂z = CairoMakie.Axis(fig[2, 2], xlabel=L"(psu m$^{-1}$)", ylabel="z (m)", title="Salinity gradient")
    ax∂σ∂z = CairoMakie.Axis(fig[3, 2], xlabel=L"(kg m$^{-3}$ m$^{-1}$)", ylabel="z (m)", title="Potential density gradient")
    axRi = CairoMakie.Axis(fig[4, 2], xlabel="arctan(Ri)", ylabel="z (m)", title="Richardson number")
    
    # Highlight NN active region with shaded polygons
    for (ax, lims) in [(axT, Tlim), (axS, Slim), (axσ, σlim), (axvel, vellim),
                       (ax∂T∂z, ∂T∂zlim), (ax∂S∂z, ∂S∂zlim), (ax∂σ∂z, ∂σ∂zlim), (axRi, Rilim)]
        poly!(ax, [(lims[1], zF[first_index]), (lims[2], zF[first_index]), 
                   (lims[2], zF[last_index]), (lims[1], zF[last_index])], 
              color=(colors[7], 0.3))
    end

    # Plot state variables
    lines!(axT, T, zC, color=colors[1], linewidth=5)
    lines!(axS, S, zC, color=colors[2], linewidth=5)
    lines!(axσ, σ, zC, color=colors[3], linewidth=5)
    lines!(axvel, u, zC, color=colors[5], linewidth=5, label="u")
    lines!(axvel, v, zC, color=colors[6], linewidth=5, label="v")

    # Plot gradients
    lines!(ax∂T∂z, ∂T∂z, zF_plot, color=colors[1], linewidth=5)
    lines!(ax∂S∂z, ∂S∂z, zF_plot, color=colors[2], linewidth=5)
    lines!(ax∂σ∂z, ∂σ∂z, zF_plot, color=colors[3], linewidth=5)
    lines!(axRi, Ri, zF_plot, color=colors[4], linewidth=5)

    axislegend(axvel, position=:lb, labelfont=:bold)

    # Set axis limits
    for (ax, lims) in [(axT, Tlim), (axS, Slim), (axσ, σlim), (axvel, vellim),
                       (ax∂T∂z, ∂T∂zlim), (ax∂S∂z, ∂S∂zlim), (ax∂σ∂z, ∂σ∂zlim), (axRi, Rilim)]
        xlims!(ax, lims)
    end

    linkyaxes!(axT, axS, axσ, axvel, ax∂T∂z, ax∂S∂z, ax∂σ∂z, axRi)

    # Hide decorations for cleaner appearance
    for ax in [axT, axS, axσ, axvel, ax∂T∂z, ax∂S∂z, ax∂σ∂z, axRi]
        hidedecorations!(ax, ticks=false, ticklabels=false, label=false)
    end

    trim!(fig.layout)
    display(fig)
    # save("./figures/NN_input_schematic.svg", fig)
end

#####
##### Plot Neural Network Outputs (Fluxes)
#####

# Extract flux components
wT_residual = sol.fluxes.wT.residual[:, timeframe][2:end-1]
wS_residual = sol.fluxes.wS.residual[:, timeframe][2:end-1]

wT_baseclosure = sol.fluxes.wT.diffusive_boundary[:, timeframe]
wS_baseclosure = sol.fluxes.wS.diffusive_boundary[:, timeframe]

# Total flux = NN residual + physical closure
wT_total = sol.fluxes.wT.residual[:, timeframe] .+ wT_baseclosure
wS_total = sol.fluxes.wS.residual[:, timeframe] .+ wS_baseclosure

wT_lim = padded_lims(wT_total, width_multiplier)
wS_lim = padded_lims(wS_total, width_multiplier)

with_theme(theme_latexfonts()) do
    fig = Figure(size=(400, 750), fontsize=15)
    axwT = CairoMakie.Axis(fig[1, 1], xlabel=L"($\degree$C m s$^{-1}$)", ylabel="z (m)", title="Temperature flux")
    axwS = CairoMakie.Axis(fig[2, 1], xlabel=L"(psu m s$^{-1}$)", ylabel="z (m)", title="Salinity flux", xticks=LinearTicks(4))
    
    # Highlight NN active region
    for (ax, lims) in [(axwT, wT_lim), (axwS, wS_lim)]
        poly!(ax, [(lims[1], zF[first_index]), (lims[2], zF[first_index]), 
                   (lims[2], zF[last_index]), (lims[1], zF[last_index])], 
              color=(colors[7], 0.3))
    end

    # Plot total flux (NN + physical closure)
    lines!(axwT, wT_total, zF, color=(colors[1], 0.5), linewidth=10, label="NN + physical closure")
    lines!(axwS, wS_total, zF, color=(colors[2], 0.5), linewidth=10, label="NN + physical closure")

    # Plot physical closure alone
    lines!(axwT, wT_baseclosure, zF, color=colors[1], linestyle=:dash, linewidth=5, label="Physical closure")
    lines!(axwS, wS_baseclosure, zF, color=colors[2], linestyle=:dash, linewidth=5, label="Physical closure")

    # Plot NN residual
    lines!(axwT, wT_residual, zF_plot, color=:black, linewidth=5, label="NN")
    lines!(axwS, wS_residual, zF_plot, color=:black, linewidth=5, label="NN")

    xlims!(axwT, wT_lim)
    xlims!(axwS, wS_lim)

    linkyaxes!(axwT, axwS)
    hidedecorations!(axwT, ticks=false, ticklabels=false, label=false)
    hidedecorations!(axwS, ticks=false, ticklabels=false, label=false)

    axislegend(axwT, position=:rb, labelfont=:bold, patchsize=(40, 20))
    axislegend(axwS, position=:lb, labelfont=:bold, patchsize=(40, 20))
    
    trim!(fig.layout)
    display(fig)
    # save("./figures/NN_output_schematic.svg", fig)
end