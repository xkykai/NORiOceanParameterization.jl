# We activate a special environment for this script for compatibility reasons since the simulation was run with the specific versions
using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using JLD2
using Makie
using Oceananigans
using Oceananigans.Units
using ColorSchemes
using Oceananigans.BuoyancyModels: g_Earth
using SeawaterPolynomials.TEOS10

#####
##### Physical constants
#####

const ρ₀ = TEOS10EquationOfState().reference_density
colors = Makie.wong_colors();

#####
##### Load data
#####

years = 2.5

# Uncomment this for 100 years data (shown in appendix of paper)
# years = 100

filepath = joinpath(@__DIR__, "..", "figure_data", "doublegyre", "zonal_average_stratification_CATKE_$(years)years.jld2")

NN_field, physicalclosure_field, baseclosure_field, Δ_field, Δ_baseclosure_field, grid, times = jldopen(filepath, "r") do file
    return (file["NN_field"], file["physicalclosure_field"], file["baseclosure_field"], 
            file["Δ_field"], file["Δ_baseclosure_field"], file["grid"], file["times"])
end

#####
##### Extract grid coordinates
#####

Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

xC = grid.xᶜᵃᵃ[1:Nx] ./ 1000  # Convert to km
yC = grid.yᵃᶜᵃ[1:Ny] ./ 1000
zC = grid.zᵃᵃᶜ[1:Nz]
zF = grid.zᵃᵃᶠ[1:Nz+1]

times = times / 24 / 60^2 / 360  # Convert to years

#####
##### Helper functions for colorbar limits
#####

find_min(a...) = minimum(minimum.([a...]))
find_max(a...) = maximum(maximum.([a...]))

#####
##### Compute plot limits
#####

# Absolute field limits (shared across NORi, CATKE, base closures)
fieldlim = (find_min(NN_field, physicalclosure_field, baseclosure_field), 
            find_max(NN_field, physicalclosure_field, baseclosure_field))

# Symmetric limits for differences (NORi - CATKE, NORi - base)
Δ_fieldlim = (-find_max(abs.(Δ_field), abs.(Δ_baseclosure_field)), 
               find_max(abs.(Δ_field), abs.(Δ_baseclosure_field)))

#####
##### Color schemes and contour levels
#####

field_colorscheme = colorschemes[:turbo];
Δ_colorscheme = colorschemes[:balance];  # Diverging colormap for differences

field_contourlevels = range(fieldlim[1], fieldlim[2], length=15)
Δ_contourlevels = range(Δ_fieldlim[1], Δ_fieldlim[2], length=10)

#####
##### Create figure
#####

aspect_ratio = 1.8

with_theme(theme_latexfonts()) do
    fig = Figure(size=(1800, 700), fontsize=25)
    
    # NORi closure takes up 2x2 grid space (larger)
    axNN = Axis(fig[1:2, 1:2], xlabel="y (km)", ylabel="z (m)", 
                title="NORi closure", aspect=AxisAspect(aspect_ratio))
    
    # Other closures and differences in right columns
    axphysicalclosure = Axis(fig[1, 3], xlabel="y (km)", ylabel="z (m)", 
                             title="CATKE closure", aspect=AxisAspect(aspect_ratio))
    axbase = Axis(fig[2, 3], xlabel="y (km)", ylabel="z (m)", 
                  title="Base closure", aspect=AxisAspect(aspect_ratio))
    axΔ = Axis(fig[1, 4], xlabel="y (km)", ylabel="z (m)", 
               title="NORi - CATKE", aspect=AxisAspect(aspect_ratio))
    axΔbase = Axis(fig[2, 4], xlabel="y (km)", ylabel="z (m)", 
                   title="NORi - base closure", aspect=AxisAspect(aspect_ratio))

    # Filled contour plots
    NN_ctf = contourf!(axNN, yC, zC, NN_field, colormap=field_colorscheme, levels=field_contourlevels)
    physicalclosure_ctf = contourf!(axphysicalclosure, yC, zC, physicalclosure_field, 
                                    colormap=field_colorscheme, levels=field_contourlevels)
    base_ctf = contourf!(axbase, yC, zC, baseclosure_field, 
                        colormap=field_colorscheme, levels=field_contourlevels)
    Δ_ctf = contourf!(axΔ, yC, zC, Δ_field, colormap=Δ_colorscheme, levels=Δ_contourlevels)
    Δbase_ctf = contourf!(axΔbase, yC, zC, Δ_baseclosure_field, 
                         colormap=Δ_colorscheme, levels=Δ_contourlevels)

    # Colorbars
    Colorbar(fig[3, 1:3], NN_ctf, label=L"Temperature ($\degree$C)", 
             vertical=false, flipaxis=false)
    Colorbar(fig[3, 4], Δ_ctf, label=L"Temperature anomaly ($\degree$C)", 
             vertical=false, flipaxis=false)

    # Hide decorations except for leftmost column
    hidedecorations!(axphysicalclosure, ticks=false)
    hidedecorations!(axbase, ticks=false)
    hidedecorations!(axΔ, ticks=false)
    hidedecorations!(axΔbase, ticks=false)

    # Link all axes to share zoom/pan
    linkaxes!(axNN, axbase, axphysicalclosure, axΔ, axΔbase)

    axs = [axNN, axbase, axphysicalclosure, axΔ, axΔbase]

    for ax in axs
        xlims!(ax, minimum(yC), maximum(yC))
        ylims!(ax, minimum(zC), maximum(zC))
    end

    display(fig)

    # save("./figures/double_gyre_zonalaverage_stratification_CATKE_$(years)years.pdf", fig)
end