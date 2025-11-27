# We activate a special environment for this script for compatibility reasons since the simulation was run with the specific versions
using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using JLD2
using Makie
using Oceananigans
using Oceananigans.Units
using ColorSchemes
colors = Makie.wong_colors();
using SeawaterPolynomials.TEOS10

const ρ₀ = TEOS10EquationOfState().reference_density
const g_Earth = Oceananigans.BuoyancyModels.g_Earth

FILE_DIR = joinpath(@__DIR__, "..", "figure_data", "doublegyre")

# Load pre-computed double gyre data from file
b_xy, b_yz, b_xz, b_zonal, Ψ, T_xy, z_depth, T_surfaces, S_surfaces, Ψ_data, ys, uw_surface = jldopen(joinpath(FILE_DIR, "doublegyre_setup_data.jld2"), "r") do file
    b_xy = file["b_xy"]
    b_yz = file["b_yz"]
    b_xz = file["b_xz"]
    b_zonal = file["b_zonal"]
    Ψ = file["Ψ"]
    T_xy = file["T_xy"]
    z_depth = file["z_depth"]
    T_surfaces = file["T_surfaces"]
    S_surfaces = file["S_surfaces"]
    Ψ_data = file["Ψ_data"]
    ys = file["ys"]
    uw_surface = file["uw_surface"]
    return b_xy, b_yz, b_xz, b_zonal, Ψ, T_xy, z_depth, T_surfaces, S_surfaces, Ψ_data, ys, uw_surface
end

Nx, Ny, Nz = Ψ_data.grid.Nx, Ψ_data.grid.Ny, Ψ_data.grid.Nz
Lx, Ly, Lz = Ψ_data.grid.Lx, Ψ_data.grid.Ly, Ψ_data.grid.Lz

# Extract grid coordinates: C = cell centers, F = cell faces
# Convert x and y from meters to kilometers for plotting
xC, yC, zC = Ψ_data.grid.xᶜᵃᵃ[1:Nx] ./ 1000, Ψ_data.grid.yᵃᶜᵃ[1:Ny] ./ 1000, Ψ_data.grid.zᵃᵃᶜ[1:Nz]
xF, yF = Ψ_data.grid.xᶠᵃᵃ[1:Nx+1] ./ 1000, Ψ_data.grid.yᵃᶠᵃ[1:Ny+1] ./ 1000
zF = Ψ_data.grid.zᵃᵃᶠ[1:Nz+1]

# Create coordinate grids for xy plane (horizontal surface at top)
xCs_xy = xC
yCs_xy = yC
zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

# Create coordinate grids for yz plane (vertical cross-section at constant x)
yCs_yz = yC
xCs_yz = range(xC[1], stop=xC[1], length=length(zC))  # Constant x value
zCs_yz = zeros(length(xCs_yz), length(yCs_yz))
for j in axes(zCs_yz, 2)
  zCs_yz[:, j] .= zC
end

# Create coordinate grids for xz plane (vertical cross-section at constant y)
xCs_xz = xC
yCs_xz = range(yC[1], stop=yC[1], length=length(zC))  # Constant y value
zCs_xz = zeros(length(xCs_xz), length(yCs_xz))
for i in axes(zCs_xz, 1)
  zCs_xz[i, :] .= zC
end

# Similar coordinate grids for face-centered data (F subscript)
xFs_xy = xC
yFs_xy = yC
zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]

yFs_yz = yC
xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
zFs_yz = zeros(length(xFs_yz), length(yFs_yz))
for j in axes(zFs_yz, 2)
  zFs_yz[:, j] .= zF
end

xFs_xz = xC
yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
zFs_xz = zeros(length(xFs_xz), length(yFs_xz))
for i in axes(zFs_xz, 1)
  zFs_xz[i, :] .= zF
end

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

b_from_ρ(ρ) = -g_Earth * (ρ - ρ₀) / ρ₀

frame_Ψ = 11  # Timestep index for barotropic streamfunction

b_colormap = colorschemes[:turbo];

blim = (find_min(b_xy, b_yz, b_xz, b_zonal), 
        find_max(b_xy, b_yz, b_xz, b_zonal))

b_color_range = blim

b_zonal_levels = range(blim[1], blim[2], length=15)

# Extract streamfunction and convert from m³/s to Sverdrups (10⁶ m³/s)
Ψ = interior(Ψ_data[frame_Ψ], :, :, 1) ./ 1e6
Ψlim = extrema(Ψ)
Ψlim = (Ψlim[1], -Ψlim[1])  # Symmetric colorbar limits
Ψ_colormap = colorschemes[:balance];
Ψ_levels = range(Ψlim[1], Ψlim[2], length=15)

Tlim = extrema(T_xy)
T_colormap = colorschemes[:plasma];
T_levels = range(Tlim[1], Tlim[2], length=10)

plot_aspect = (2, 2, 1)  # Aspect ratio for 3D plot (x:y:z)
zonal_plot_displacement = 500  # Offset for zonal mean contour plot (km)

with_theme(theme_latexfonts()) do
      fig = Figure(size=(2400, 1200), fontsize=30)
      # Create grid layout: g1 = 3D buoyancy, g2 = surface forcings, g3 = 2D fields
      g1 = fig[1:2, 1:2] = GridLayout()
      g2 = fig[1, 3:4] = GridLayout()
      g3 = fig[2, 3:4] = GridLayout()
  
      axb = Axis3(g1[1, 1], title="Buoyancy", xlabel="x (km)", ylabel="y (km)", zlabel="z (m)", viewmode=:fitzoom, aspect=plot_aspect, protrusions=(80, 30, 30, 30), perspectiveness=0.7, elevation=0.15π, xlabeloffset=90, ylabeloffset=90, zlabeloffset=90, azimuth = 1.2π)
      # Dual x-axes for temperature (blue) and salinity (red) restoration
      axJᵀ = Axis(g2[1, 1], xlabel=L"Temperature restoration ($\degree$C)", ylabel="y (km)", xticklabelcolor = :blue, xlabelcolor = :blue, xtickcolor = :blue)
      axJˢ = Axis(g2[1, 1], xlabel="Salinity restoration (psu)", xticklabelcolor = :red, xaxisposition = :top, xlabelcolor = :red, xtickcolor = :red)
      axτ = Axis(g2[1, 2], xlabel=L"Momentum flux ($\times 10^{-4}$ m$^{-2}$ s$^{-2}$)", ylabel="y (km)")
      Label(g2[0, 1:2], "Surface forcings", font=:bold, tellwidth=false)
  
      hidespines!(axJˢ)
      hideydecorations!(axJˢ)
  
      axΨ = Axis(g3[1, 1], title="Barotropic streamfunction (Sv)", xlabel="x (km)", ylabel="y (km)")
      axT = Axis(g3[1, 2], title="Temperature at $(z_depth)m depth", xlabel="x (km)", ylabel="y (km)")
  
      lines!(axJᵀ, T_surfaces, ys / 1000, color=:black, linewidth=4)
      lines!(axJˢ, S_surfaces, ys / 1000, color=:black, linewidth=4)
  
      # Convert momentum flux to ×10⁻⁴ units for readability
      lines!(axτ, uw_surface .* 1e4, ys / 1000, color=colors[1], linewidth=4)
  
      # 3D surfaces for buoyancy on three orthogonal planes
      b_xy_surface = surface!(axb, xCs_xy, yCs_xy, zCs_xy, color=b_xy, colormap=b_colormap, colorrange = b_color_range, rasterize=10)
      b_yz_surface = surface!(axb, xCs_yz, yCs_yz, zCs_yz, color=b_yz, colormap=b_colormap, colorrange = b_color_range, rasterize=10)
      b_xz_surface = surface!(axb, xCs_xz, yCs_xz, zCs_xz, color=b_xz, colormap=b_colormap, colorrange = b_color_range, rasterize=10)
  
      # Zonal mean contour on left side of 3D plot, displaced to avoid overlap
      contourf!(axb, yC, zC, b_zonal, transformation = (:yz, minimum(xC) - zonal_plot_displacement), colormap=b_colormap, levels=b_zonal_levels)

      Colorbar(g1[2,1], b_xy_surface, label=L"(m s$^{-2}$)", vertical=false, flipaxis=false)
  
      Ψ_ctf = contourf!(axΨ, xF, yC, Ψ, levels=Ψ_levels, colormap=Ψ_colormap, extendhigh=Ψ_colormap[end])
  
      Colorbar(g3[2, 1], Ψ_ctf, label="(Sv)", vertical=false, flipaxis=false)
  
      T_ctf = contourf!(axT, xC, yC, T_xy, levels=T_levels, colormap=T_colormap)
  
      Colorbar(g3[2, 2], T_ctf, label="(°C)", vertical=false, flipaxis=false)
  
      # Extend x-limits to accommodate displaced zonal mean plot
      xlims!(axb, (minimum(xC) - zonal_plot_displacement, maximum(xC)))
      ylims!(axb, (minimum(yC), maximum(yC)))
      zlims!(axb, (minimum(zC), 0))
  
      xlims!(axΨ, (minimum(xC), maximum(xC)))
      ylims!(axΨ, (minimum(yC), maximum(yC)))
  
      xlims!(axT, (minimum(xC), maximum(xC)))
      ylims!(axT, (minimum(yC), maximum(yC)))
      
      linkyaxes!(axJᵀ, axJˢ, axτ)
      linkaxes!(axΨ, axT)
  
      # Hide redundant decorations for cleaner appearance
      hideydecorations!(axJᵀ, ticks=false, ticklabels=false, label=false)
      hidexdecorations!(axJᵀ, ticks=false, ticklabels=false, label=false)
      hidexdecorations!(axJˢ, ticks=false, ticklabels=false, label=false)
  
      hideydecorations!(axτ, ticks=false)
      hidexdecorations!(axτ, ticks=false, ticklabels=false, label=false)
  
      hideydecorations!(axΨ, ticks=false, ticklabels=false, label=false)
      hidexdecorations!(axΨ, ticks=false, ticklabels=false, label=false)
  
      hideydecorations!(axT, ticks=false)
      hidexdecorations!(axT, ticks=false, ticklabels=false, label=false)

      display(fig)
  
      # save("./figures/double_gyre_setup.pdf", fig)
end