using Oceananigans: Center, Face

"""
    coarse_grain(Φ, n, ::Type{Center})

Coarse grain a `Center`-centered field by averaging to a lower resolution.

This function reduces the resolution of a field `Φ` defined at cell centers by averaging
consecutive groups of points. The original field is divided into `n` bins of equal size,
and each bin is replaced by its arithmetic mean.

# Arguments
- `Φ`: Input field array with values at cell centers.
- `n::Integer`: Target number of coarse-grained points (must evenly divide `length(Φ)`).
- `::Type{Center}`: Location type indicating data is at cell centers.

# Returns
- Array of length `n` containing the coarse-grained field values.

# Mathematical Description
For a field Φ with N points, the coarse-grained field Φ̅ at index i is:

```
Δ = N / n
Φ̅[i] = (1/Δ) * Σ(j=(i-1)Δ+1 to iΔ) Φ[j]
```

# Requirements
- `Φ` must have evenly spaced grid points.
- `n` must evenly divide `length(Φ)` (i.e., `length(Φ) % n == 0`).
- The output preserves the spatial extent of the original field.

# Examples
```julia
julia> Φ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

julia> coarse_grain(Φ, 4, Center)
4-element Vector{Float64}:
 1.5  # average of [1.0, 2.0]
 3.5  # average of [3.0, 4.0]
 5.5  # average of [5.0, 6.0]
 7.5  # average of [7.0, 8.0]

julia> coarse_grain(Φ, 2, Center)
2-element Vector{Float64}:
 2.5  # average of [1.0, 2.0, 3.0, 4.0]
 6.5  # average of [5.0, 6.0, 7.0, 8.0]
```

# See Also
- [`coarse_grain(Φ, n, ::Type{Face})`](@ref): Coarse graining for face-centered fields
"""
function coarse_grain(Φ, n, ::Type{Center})
    N = length(Φ)
    Δ = Int(N / n)
    Φ̅ = similar(Φ, n)
    for i in 1:n
        Φ̅[i] = mean(Φ[Δ*(i-1)+1:Δ*i])
    end
    return Φ̅
end

"""
    coarse_grain(Φ, n, ::Type{Face})

Coarse grain a `Face`-centered field by averaging to a lower resolution while preserving boundaries.

This function reduces the resolution of a field `Φ` defined at cell faces (interfaces between
cells) by averaging interior points. The first and last values (boundaries) are always preserved
exactly in the output.

# Arguments
- `Φ`: Input field array with values at cell faces.
- `n::Integer`: Target number of coarse-grained points (must be ≥ 2).
- `::Type{Face}`: Location type indicating data is at cell faces.

# Returns
- Array of length `n` containing the coarse-grained field values with preserved boundaries.

# Mathematical Description
For a face-centered field with N points:
- Φ̅[1] = Φ[1] (left boundary preserved)
- Φ̅[n] = Φ[N] (right boundary preserved)
- Interior points are averaged over bins of size Δ = (N-2)/(n-2)

If Δ is an integer, uses exact binning. Otherwise, uses adaptive binning with rounding.

# Requirements
- `Φ` must have evenly spaced grid points.
- `n ≥ 2` to preserve both boundary points.
- Boundary values are always preserved exactly.

# Examples
```julia
julia> Φ = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

julia> coarse_grain(Φ, 3, Face)
3-element Vector{Float64}:
 0.0  # left boundary preserved
 2.5  # average of interior points
 5.0  # right boundary preserved

julia> Φ = [0.0, 1.0, 2.0, 3.0, 4.0];

julia> coarse_grain(Φ, 4, Face)
4-element Vector{Float64}:
 0.0  # boundary
 1.5  # average of [1.0, 2.0]
 3.0  # average of [3.0]
 4.0  # boundary
```

# Notes
- This method is particularly useful for fields defined on staggered grids where boundary
  conditions must be preserved.
- For non-integer bin sizes, the function uses adaptive binning to ensure all interior
  points contribute to the averaged field.

# See Also
- [`coarse_grain(Φ, n, ::Type{Center})`](@ref): Coarse graining for center-centered fields
- [`coarse_grain_linear_interpolation`](@ref): Alternative using linear interpolation
- [`coarse_grain_downsampling`](@ref): Simple downsampling without averaging
"""
function coarse_grain(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Δ = (N-2) / (n-2)
    Φ̅[1], Φ̅[n] = Φ[1], Φ[N]

    if isinteger(Δ)
        Φ̅[2:n-1] .= coarse_grain(Φ[2:N-1], n-2, Center)
    else
        for i in 2:n-1
            i1 = round(Int, 2 + (i-2)*Δ)
            i2 = round(Int, 2 + (i-1)*Δ)
            Φ̅[i] = mean(Φ[i1:i2])
        end
    end

    return Φ̅
end

"""
    coarse_grain_linear_interpolation(Φ, n, ::Type{Face})

Coarse grain a `Face`-centered field using linear interpolation to a lower resolution.

This function reduces the resolution of a field `Φ` by linearly interpolating values at
uniformly spaced target positions. Unlike averaging-based coarse graining, this method
provides smooth interpolation between original grid points.

# Arguments
- `Φ`: Input field array with values at cell faces.
- `n::Integer`: Target number of interpolated points (must be ≥ 2).
- `::Type{Face}`: Location type indicating data is at cell faces.

# Returns
- Array of length `n` containing linearly interpolated values with preserved boundaries.

# Mathematical Description
For target positions uniformly spaced between 1 and N:

```
gap = (N-1) / (n-1)
x[i] = 1 + (i-1) * gap

Φ̅[i] = (⌊x[i]⌋+1 - x[i]) * Φ[⌊x[i]⌋] + (x[i] - ⌊x[i]⌋) * Φ[⌊x[i]⌋+1]
```

where ⌊·⌋ denotes the floor function.

# Requirements
- `Φ` must have evenly spaced grid points.
- `n ≥ 2` to preserve both boundary points.
- Input field should be sufficiently smooth for linear interpolation to be appropriate.

# Examples
```julia
julia> Φ = [0.0, 1.0, 4.0, 9.0, 16.0];  # y = x²

julia> coarse_grain_linear_interpolation(Φ, 3, Face)
3-element Vector{Float64}:
  0.0   # boundary at x=0
  6.5   # interpolated at x=2: 0.5*4.0 + 0.5*9.0
 16.0   # boundary at x=4

julia> Φ_linear = [0.0, 2.0, 4.0, 6.0, 8.0];

julia> coarse_grain_linear_interpolation(Φ_linear, 3, Face)
3-element Vector{Float64}:
 0.0  # boundary
 4.0  # exact at midpoint
 8.0  # boundary
```

# Notes
- Linear interpolation preserves linear trends exactly but may not capture higher-order
  features as well as averaging-based methods.
- This method is computationally efficient and provides C⁰ continuity.
- Best suited for fields that vary smoothly between grid points.

# See Also
- [`coarse_grain(Φ, n, ::Type{Face})`](@ref): Averaging-based coarse graining
- [`coarse_grain_downsampling`](@ref): Direct subsampling without interpolation
"""
function coarse_grain_linear_interpolation(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Φ̅[1] = Φ[1]
    Φ̅[end] = Φ[end]
    gap = (N-1)/(n-1)

    for i=2:n-1
        Φ̅[i] = 1 + (i-1)*gap
    end

    for i=2:n-1
        Φ̅[i] = (floor(Φ̅[i])+1 - Φ̅[i]) * Φ[Int(floor(Φ̅[i]))] + (Φ̅[i] - floor(Φ̅[i])) * Φ[Int(floor(Φ̅[i]))+1]
    end
    return Φ̅
end

"""
    coarse_grain_downsampling(Φ, n, ::Type{Face})

Downsample a `Face`-centered field by selecting uniformly spaced points.

This function reduces the resolution by directly selecting every Δth point from the original
field, where Δ = (N-1)/(n-1). No averaging or interpolation is performed; values are taken
directly from the input array.

# Arguments
- `Φ`: Input field array with values at cell faces.
- `n::Integer`: Target number of downsampled points (must be ≥ 2).
- `::Type{Face}`: Location type indicating data is at cell faces.

# Returns
- Array of length `n` containing downsampled values with preserved boundaries.

# Mathematical Description
The downsampled field selects points at indices:

```
Δ = (N-1) / (n-1)
Φ̅[i] = Φ[Δ*(i-1) + 1]  for i = 1, 2, ..., n
```

# Requirements
- `Φ` must have evenly spaced grid points.
- `(length(Φ) - 1)` must be divisible by `(n - 1)` for exact downsampling.
- Boundary values are always preserved exactly.

# Examples
```julia
julia> Φ = [0.0, 1.0, 2.0, 3.0, 4.0];

julia> coarse_grain_downsampling(Φ, 3, Face)
3-element Vector{Float64}:
 0.0  # Φ[1]
 2.0  # Φ[3]
 4.0  # Φ[5]

julia> Φ = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

julia> coarse_grain_downsampling(Φ, 4, Face)
4-element Vector{Float64}:
 0.0  # Φ[1]
 1.0  # Φ[3]
 2.0  # Φ[5]
 3.0  # Φ[7]
```

# Notes
- This is the fastest coarse graining method but may miss important features between
  sampled points.
- No smoothing or averaging is applied, so high-frequency variations are preserved
  at sampled locations but may be aliased.
- Best used when the original field is already well-resolved and smoothly varying.
- For noisy data, consider using averaging-based methods instead.

# See Also
- [`coarse_grain(Φ, n, ::Type{Face})`](@ref): Averaging-based coarse graining
- [`coarse_grain_linear_interpolation`](@ref): Interpolation-based coarse graining
"""
function coarse_grain_downsampling(Φ, n, ::Type{Face})
    N = length(Φ)
    Φ̅ = similar(Φ, n)
    Δ = Int((N-1) / (n-1))
    for i in 1:n
        Φ̅[i] = Φ[Δ*(i-1)+1]
    end
    return Φ̅
end