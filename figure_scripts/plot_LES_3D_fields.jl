using Oceananigans
using CairoMakie
using ColorSchemes
using NORiOceanParameterization.Utils: find_min, find_max
using Makie
using JLD2

#####
##### Load Data
#####

filepath = joinpath(pwd(), "figure_data", "LES_3D_fields.jld2")

b_xy_convection, b_yz_convection, b_xz_convection, 
b_xy_wind, b_yz_wind, b_xz_wind, 
w_xy_convection, w_yz_convection, w_xz_convection, 
w_xy_wind, w_yz_wind, w_xz_wind, 
Tbar_data_convection, Tbar_data_wind, 
Sbar_data_convection, Sbar_data_wind = jldopen(filepath, "r") do file
    return (
        file["b_xy_convection"], file["b_yz_convection"], file["b_xz_convection"],
        file["b_xy_wind"], file["b_yz_wind"], file["b_xz_wind"],
        file["w_xy_convection"], file["w_yz_convection"], file["w_xz_convection"],
        file["w_xy_wind"], file["w_yz_wind"], file["w_xz_wind"],
        file["Tbar_data_convection"], file["Tbar_data_wind"],
        file["Sbar_data_convection"], file["Sbar_data_wind"]
    )
end

#####
##### Time and Grid Parameters
#####

Nt = length(Tbar_data_convection.times)
n = 301  # Time index for 3D visualization
n_middle = 361  # Middle time index for profile comparison
times = Tbar_data_convection.times ./ 3600  # Convert to hours

# Extract grid dimensions and coordinates
Nx, Ny, Nz = Tbar_data_convection.grid.Nx, Tbar_data_convection.grid.Ny, Tbar_data_convection.grid.Nz
xC, yC, zC = Tbar_data_convection.grid.xᶜᵃᵃ[1:Nx], Tbar_data_convection.grid.yᵃᶜᵃ[1:Ny], Tbar_data_convection.grid.zᵃᵃᶜ[1:Nz]
zF = Tbar_data_convection.grid.zᵃᵃᶠ[1:Nz+1]
Lx, Ly, Lz = Tbar_data_convection.grid.Lx, Tbar_data_convection.grid.Ly, Tbar_data_convection.grid.Lz

#####
##### Create Coordinate Arrays for 3D Surfaces
#####

# xy-plane (horizontal slice at top of domain)
xCs_xy = xC
yCs_xy = yC
zCs_xy = [zC[Nz] for x in xCs_xy, y in yCs_xy]

# yz-plane (vertical slice at x = xC[1])
yCs_yz = yC
xCs_yz = range(xC[1], stop=xC[1], length=length(zC))
zCs_yz = repeat(zC', length(yCs_yz), 1)'

# xz-plane (vertical slice at y = yC[1])
xCs_xz = xC
yCs_xz = range(yC[1], stop=yC[1], length=length(zC))
zCs_xz = repeat(zC, 1, length(xCs_xz))'

# Same for face-centered vertical velocity
xFs_xy = xC
yFs_xy = yC
zFs_xy = [zF[Nz+1] for x in xFs_xy, y in yFs_xy]

yFs_yz = yC
xFs_yz = range(xC[1], stop=xC[1], length=length(zF))
zFs_yz = repeat(zF', length(yFs_yz), 1)'

xFs_xz = xC
yFs_xz = range(yC[1], stop=yC[1], length=length(zF))
zFs_xz = repeat(zF, 1, length(xFs_xz))'

#####
##### Extract Horizontally-Averaged Profiles
#####

# Final time profiles
Tbar_convection = interior(Tbar_data_convection[Nt], 1, 1, :)
Tbar_wind = interior(Tbar_data_wind[Nt], 1, 1, :)
Sbar_convection = interior(Sbar_data_convection[Nt], 1, 1, :)
Sbar_wind = interior(Sbar_data_wind[Nt], 1, 1, :)

# Initial profiles
Tbar_convection_initial = interior(Tbar_data_convection[1], 1, 1, :)
Tbar_wind_initial = interior(Tbar_data_wind[1], 1, 1, :)
Sbar_convection_initial = interior(Sbar_data_convection[1], 1, 1, :)
Sbar_wind_initial = interior(Sbar_data_wind[1], 1, 1, :)

# Middle time profiles
Tbar_convection_middle = interior(Tbar_data_convection[n_middle], 1, 1, :)
Tbar_wind_middle = interior(Tbar_data_wind[n_middle], 1, 1, :)
Sbar_convection_middle = interior(Sbar_data_convection[n_middle], 1, 1, :)
Sbar_wind_middle = interior(Sbar_data_wind[n_middle], 1, 1, :)

#####
##### Set Color Ranges (Exclude Ocean Interior to Improve Contrast)
#####

startheight = 100  # Grid point index to start color range calculation

# Buoyancy color ranges
blim_convection = (find_min(b_xz_convection[:, startheight:end]), find_max(b_xz_convection[:, startheight:end]))
blim_wind = (find_min(b_xz_wind[:, startheight:end]), find_max(b_xz_wind[:, startheight:end]))

# Vertical velocity color ranges (symmetric about zero)
wextrema_convection = (find_min(w_xz_convection[:, startheight:end]), find_max(w_xz_convection[:, startheight:end]))
wextrema_wind = (find_min(w_xz_wind[:, startheight:end]), find_max(w_xz_wind[:, startheight:end]))
wlim_convection = (-maximum(abs, wextrema_convection), maximum(abs, wextrema_convection)) ./ 2
wlim_wind = (-maximum(abs, wextrema_wind), maximum(abs, wextrema_wind)) ./ 2

#####
##### Plotting Configuration
#####

b_colormap = colorschemes[:turbo];
w_colormap = colorschemes[:balance];
rasterize_level = 5  # Rasterization level for faster rendering
protrusions = 10  # Space around 3D axes

#####
##### Create Figure
#####

with_theme(theme_latexfonts()) do
    fig = Figure(size=(2200, 1300), fontsize=25)
    
    # Row labels
    Label(fig[1, 0], "Free Convection", rotation=π/2, fontsize=25, font=:bold, tellheight=false, padding=(0, 20, 0, 0))
    Label(fig[3, 0], "Wind Stress", rotation=π/2, fontsize=25, font=:bold, tellheight=false, padding=(0, 20, 0, 0))

    # Column headers
    Label(fig[0, 1], "Buoyancy, $(round(times[n], digits=3)) hours", font=:bold, tellwidth=false)
    Label(fig[0, 2], "Vertical velocity, $(round(times[n], digits=3)) hours", font=:bold, tellwidth=false)
    Label(fig[0, 3], "Horizontally-averaged temperature", font=:bold, tellwidth=false)
    Label(fig[0, 4], "Horizontally-averaged salinity", font=:bold, tellwidth=false)

    # 3D axes
    axb_convection = Axis3(fig[1, 1], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", 
                           viewmode=:fitzoom, aspect=:data, perspectiveness=0.2, 
                           elevation=0.15π, protrusions=protrusions, 
                           xlabelfont=:bold, ylabelfont=:bold, zlabelfont=:bold)
    axb_wind = Axis3(fig[3, 1], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", 
                     viewmode=:fitzoom, aspect=:data, perspectiveness=0.2, 
                     elevation=0.15π, protrusions=protrusions, 
                     xlabelfont=:bold, ylabelfont=:bold, zlabelfont=:bold)
    axw_convection = Axis3(fig[1, 2], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", 
                           viewmode=:fitzoom, aspect=:data, perspectiveness=0.2, 
                           elevation=0.15π, protrusions=protrusions, 
                           xlabelfont=:bold, ylabelfont=:bold, zlabelfont=:bold)
    axw_wind = Axis3(fig[3, 2], xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", 
                     viewmode=:fitzoom, aspect=:data, perspectiveness=0.2, 
                     elevation=0.15π, protrusions=protrusions, 
                     xlabelfont=:bold, ylabelfont=:bold, zlabelfont=:bold)

    # Profile axes
    axTbar_convection = Axis(fig[1, 3], xlabel=L"$\text{T} \, (\degree \text{C})$", ylabel="z (m)")
    axTbar_wind = Axis(fig[3, 3], xlabel=L"$\text{T} \, (\degree \text{C})$", ylabel="z (m)")
    axSbar_convection = Axis(fig[1, 4], xlabel="S (psu)", ylabel="z (m)")
    axSbar_wind = Axis(fig[3, 4], xlabel="S (psu)", ylabel="z (m)")

    #####
    ##### Plot 3D Surfaces
    #####

    # Buoyancy - free convection
    surface!(axb_convection, xCs_xy, yCs_xy, zCs_xy, color=b_xy_convection, 
             colormap=b_colormap, colorrange=blim_convection, rasterize=rasterize_level, lowclip=b_colormap[1])
    surface!(axb_convection, xCs_yz, yCs_yz, zCs_yz, color=b_yz_convection, 
             colormap=b_colormap, colorrange=blim_convection, rasterize=rasterize_level, lowclip=b_colormap[1])
    b_convection_surf = surface!(axb_convection, xCs_xz, yCs_xz, zCs_xz, color=b_xz_convection, 
                                 colormap=b_colormap, colorrange=blim_convection, rasterize=rasterize_level, lowclip=b_colormap[1])

    # Buoyancy - wind stress
    surface!(axb_wind, xCs_xy, yCs_xy, zCs_xy, color=b_xy_wind, 
             colormap=b_colormap, colorrange=blim_wind, rasterize=rasterize_level, lowclip=b_colormap[1])
    surface!(axb_wind, xCs_yz, yCs_yz, zCs_yz, color=b_yz_wind, 
             colormap=b_colormap, colorrange=blim_wind, rasterize=rasterize_level, lowclip=b_colormap[1])
    b_wind_surf = surface!(axb_wind, xCs_xz, yCs_xz, zCs_xz, color=b_xz_wind, 
                          colormap=b_colormap, colorrange=blim_wind, rasterize=rasterize_level, lowclip=b_colormap[1])

    # Vertical velocity - free convection
    surface!(axw_convection, xCs_xy, yCs_xy, zCs_xy, color=w_xy_convection, 
             colormap=w_colormap, colorrange=wlim_convection, rasterize=rasterize_level, 
             highclip=w_colormap[end], lowclip=w_colormap[1])
    surface!(axw_convection, xFs_yz, yFs_yz, zFs_yz, color=w_yz_convection, 
             colormap=w_colormap, colorrange=wlim_convection, rasterize=rasterize_level, 
             highclip=w_colormap[end], lowclip=w_colormap[1])
    w_convection_surf = surface!(axw_convection, xFs_xz, yFs_xz, zFs_xz, color=w_xz_convection, 
                                 colormap=w_colormap, colorrange=wlim_convection, rasterize=rasterize_level, 
                                 highclip=w_colormap[end], lowclip=w_colormap[1])

    # Vertical velocity - wind stress
    surface!(axw_wind, xCs_xy, yCs_xy, zCs_xy, color=w_xy_wind, 
             colormap=w_colormap, colorrange=wlim_wind, rasterize=rasterize_level, 
             highclip=w_colormap[end], lowclip=w_colormap[1])
    surface!(axw_wind, xFs_yz, yFs_yz, zFs_yz, color=w_yz_wind, 
             colormap=w_colormap, colorrange=wlim_wind, rasterize=rasterize_level, 
             highclip=w_colormap[end], lowclip=w_colormap[1])
    w_wind_surf = surface!(axw_wind, xFs_xz, yFs_xz, zFs_xz, color=w_xz_wind, 
                          colormap=w_colormap, colorrange=wlim_wind, rasterize=rasterize_level, 
                          highclip=w_colormap[end], lowclip=w_colormap[1])

    # Colorbars
    Colorbar(fig[2, 1], b_convection_surf, label=L"$\text{m s}^{-2}$", vertical=false, flipaxis=false)
    Colorbar(fig[2, 2], w_convection_surf, label=L"$\text{m s}^{-1}$", vertical=false, flipaxis=false)
    Colorbar(fig[4, 1], b_wind_surf, label=L"$\text{m s}^{-2}$", vertical=false, flipaxis=false)
    Colorbar(fig[4, 2], w_wind_surf, label=L"$\text{m s}^{-1}$", vertical=false, flipaxis=false)

    #####
    ##### Plot Vertical Profiles
    #####

    # Temperature profiles
    lines!(axTbar_convection, Tbar_convection_initial, zC, linewidth=5, label="Initial stratification")
    lines!(axTbar_convection, Tbar_convection_middle, zC, linewidth=5, label="$(round(times[n_middle], digits=3)) hours")
    lines!(axTbar_convection, Tbar_convection, zC, linewidth=5, label="$(round(times[Nt], digits=3)) hours")
    
    lines!(axTbar_wind, Tbar_wind_initial, zC, linewidth=5, label="Initial stratification")
    lines!(axTbar_wind, Tbar_wind_middle, zC, linewidth=5, label="$(round(times[n_middle], digits=3)) hours")
    lines!(axTbar_wind, Tbar_wind, zC, linewidth=5, label="$(round(times[Nt], digits=3)) hours")

    # Salinity profiles
    lines!(axSbar_convection, Sbar_convection_initial, zC, linewidth=5, label="Initial stratification")
    lines!(axSbar_convection, Sbar_convection_middle, zC, linewidth=5, label="$(round(times[n_middle], digits=3)) hours")
    lines!(axSbar_convection, Sbar_convection, zC, linewidth=5, label="$(round(times[Nt], digits=3)) hours")

    lines!(axSbar_wind, Sbar_wind_initial, zC, linewidth=5, label="Initial stratification")
    lines!(axSbar_wind, Sbar_wind_middle, zC, linewidth=5, label="$(round(times[n_middle], digits=3)) hours")
    lines!(axSbar_wind, Sbar_wind, zC, linewidth=5, label="$(round(times[Nt], digits=3)) hours")

    #####
    ##### Set Axis Limits and Linking
    #####

    xlims!(axb_convection, (0, Lx))
    xlims!(axb_wind, (0, Lx))
    xlims!(axw_convection, (0, Lx))
    xlims!(axw_wind, (0, Lx))

    ylims!(axb_convection, (0, Ly))
    ylims!(axb_wind, (0, Ly))
    ylims!(axw_convection, (0, Ly))
    ylims!(axw_wind, (0, Ly))

    zlims!(axb_convection, (-Lz, 0))
    zlims!(axb_wind, (-Lz, 0))
    zlims!(axw_convection, (-Lz, 0))
    zlims!(axw_wind, (-Lz, 0))

    linkxaxes!(axTbar_convection, axTbar_wind)
    linkxaxes!(axSbar_convection, axSbar_wind)
    linkyaxes!(axTbar_convection, axSbar_convection, axTbar_wind, axSbar_wind)

    #####
    ##### Hide Redundant Decorations
    #####

    hidexdecorations!(axTbar_convection, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axSbar_convection, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axTbar_wind, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(axSbar_wind, ticks=false, ticklabels=false, label=false)

    hideydecorations!(axTbar_convection, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axSbar_convection, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axTbar_wind, ticks=false, ticklabels=false, label=false)
    hideydecorations!(axSbar_wind, ticks=false, ticklabels=false, label=false)

    axislegend(axTbar_convection, position=:rb, font=:bold)

    trim!(fig.layout)
    display(fig)

    # save("./figures/LES_3D_fields.pdf", fig)
end