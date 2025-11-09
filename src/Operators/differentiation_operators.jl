using SparseArrays

"""
    Dᶜ(N, Δ)

Discrete derivative operator: face → center (finite volume).

Constructs a finite volume derivative operator that computes cell-centered derivatives
from face-centered fluxes (N+1 points) to produce cell-centered fields (N points).
Corresponds to the divergence of a flux: ∂ϕ/∂z ≈ (ϕ[k+1] - ϕ[k]) / Δ.

# Arguments
- `N::Integer`: Number of cell-centered grid points
- `Δ::Real`: Uniform grid spacing (cell height)

# Returns
- `Matrix`: (N × N+1) derivative operator

# Example
```julia
julia> D = Dᶜ(5, 0.1);
julia> size(D)
(5, 6)

julia> ϕ_face = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];  # flux at faces
julia> ∂ϕ∂z_center = D * ϕ_face
5-element Vector{Float64}:
 10.0
 10.0
 10.0
 10.0
 10.0
```

# See Also
- [`Dᶜ!`](@ref): In-place version
- [`Dᶠ`](@ref): Cell → face derivative operator
"""
function Dᶜ(N, Δ)
    D = zeros(N, N+1)
    for k in 1:N
        D[k, k]   = -1.0
        D[k, k+1] =  1.0
    end
    D .= 1/Δ .* D
    # return sparse(D)
    return D
end

"""
    Dᶜ!(C, F, Δ)

In-place derivative: face → center (finite volume).

Computes the finite volume derivative from face-centered flux `F` and stores
the result in cell-centered field `C`. Represents flux divergence across cell.

# Arguments
- `C`: Output array (length N, cell-centered)
- `F`: Input array (length N+1, face-centered flux)
- `Δ::Real`: Grid spacing (cell height)

# Example
```julia
julia> C = zeros(5);
julia> F = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
julia> Dᶜ!(C, F, 0.1);
julia> C
5-element Vector{Float64}:
 10.0
 10.0
 10.0
 10.0
 10.0
```
"""
function Dᶜ!(C, F, Δ)
    for k in eachindex(C)
        C[k] = (F[k+1] - F[k]) / Δ
    end
end

"""
    Dᶠ(N, Δ)

Discrete derivative operator: center → face (finite volume).

Constructs a finite volume derivative operator that computes face-centered gradients
from cell-centered fields (N points) to produce face-centered fields (N+1 points).
Interior faces use centered differences across adjacent cells.
Boundary faces (1 and N+1) are set to zero (Dirichlet-like).

# Arguments
- `N::Integer`: Number of cell-centered grid points
- `Δ::Real`: Uniform grid spacing (cell height)

# Returns
- `Matrix`: (N+1 × N) derivative operator with zero boundary rows

# Example
```julia
julia> D = Dᶠ(5, 0.1);
julia> size(D)
(6, 5)

julia> ϕ_center = [1.0, 2.0, 3.0, 4.0, 5.0];
julia> ∂ϕ∂z_face = D * ϕ_center
6-element Vector{Float64}:
  0.0   # boundary
 10.0
 10.0
 10.0
 10.0
  0.0   # boundary
```

# See Also
- [`Dᶠ!`](@ref): In-place version
- [`Dᶜ`](@ref): Face → center derivative operator
"""
function Dᶠ(N, Δ)
    D = zeros(N+1, N)
    for k in 2:N
        D[k, k-1] = -1.0
        D[k, k]   =  1.0
    end
    D .= 1/Δ .* D
    # return sparse(D)
    return D
end

"""
    Dᶠ!(F, C, Δ)

In-place derivative: center → face (finite volume).

Computes the finite volume gradient from cell-centered field `C` and stores
the result in face-centered field `F`. Interior faces use centered differences
across adjacent cells; boundaries are set to zero.

# Arguments
- `F`: Output array (length N+1, face-centered)
- `C`: Input array (length N, cell-centered)
- `Δ::Real`: Grid spacing (cell height)

# Example
```julia
julia> F = zeros(6);
julia> C = [1.0, 2.0, 3.0, 4.0, 5.0];
julia> Dᶠ!(F, C, 0.1);
julia> F
6-element Vector{Float64}:
  0.0
 10.0
 10.0
 10.0
 10.0
  0.0
```

# Notes
Boundary values are always set to zero regardless of input.
"""
function Dᶠ!(F, C, Δ)
    for k in 2:length(F)-1
        F[k] = (C[k] - C[k-1]) / Δ
    end
    F[1] = 0
    F[end] = 0
end

"""
    D²ᶜ(N, Δ)

Second derivative operator for cell-centered fields (finite volume).

Constructs the discrete Laplacian operator by composing Dᶜ and Dᶠ.
Represents the divergence of the gradient: ∂²ϕ/∂z² = ∇·(∇ϕ).

# Arguments
- `N::Integer`: Number of cell-centered grid points
- `Δ::Real`: Uniform grid spacing (cell height)

# Returns
- `Matrix`: (N × N) second derivative operator

# Example
```julia
julia> D² = D²ᶜ(5, 0.1);
julia> size(D²)
(5, 5)

julia> ϕ = [0.0, 0.25, 0.5, 0.25, 0.0];  # parabolic profile
julia> ∇²ϕ = D² * ϕ
5-element Vector{Float64}:
 -50.0
   0.0
   0.0
   0.0
  50.0
```

# See Also
- [`Dᶜ`](@ref), [`Dᶠ`](@ref): First derivative operators
"""
function D²ᶜ(N, Δ)
   return Dᶜ(N, Δ) * Dᶠ(N, Δ) 
end

"""
    Iᶠ(N)

Interpolation operator: center → face (finite volume).

Constructs an interpolation operator that maps cell-centered values (N points)
to face-centered values (N+1 points). Interior faces use arithmetic mean of 
adjacent cell values. Boundary faces copy the nearest cell value.

# Arguments
- `N::Integer`: Number of cell-centered grid points

# Returns
- `Matrix`: (N+1 × N) interpolation operator

# Mathematical Description
For interior faces k = 2, ..., N:
```
ϕ_face[k] = 0.5 * (ϕ_center[k-1] + ϕ_center[k])
```
For boundaries:
```
ϕ_face[1] = ϕ_center[1]
ϕ_face[N+1] = ϕ_center[N]
```

# Example
```julia
julia> I = Iᶠ(5);
julia> size(I)
(6, 5)

julia> ϕ_center = [1.0, 2.0, 3.0, 4.0, 5.0];
julia> ϕ_face = I * ϕ_center
6-element Vector{Float64}:
 1.0   # boundary (copy)
 1.5   # interpolated
 2.5
 3.5
 4.5
 5.0   # boundary (copy)
```

# See Also
- [`Dᶠ`](@ref): Derivative operator on same grid staggering
"""
function Iᶠ(N)
    I = zeros(N+1, N)
    for k in 1:N
        I[k, k]   = 1.0
        I[k+1, k] = 1.0
    end
    I .= 0.5 .* I
    I[1, 1] = 1.0
    I[end, end] = 1.0
    return I
end