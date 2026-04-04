# Atom Ray Tracer

A physics-based ray tracer for visualising central force atomic orbitals, written in C++ with an OpenCL GPU compute backend and SDL2 rendering. Built as a personal project to explore the intersection of quantum mechanics, Monte Carlo methods, and GPU compute.

---

## What it does

Given a set of quantum numbers `n`, `l`, `m` from the command line, the program:

1. Evaluates the hydrogenic wavefunction ψ_nlm at a large number of candidate points in 3D space using the full radial (associated Laguerre polynomial) and angular (real spherical harmonic) decomposition
2. Accepts points via rejection sampling proportional to |ψ|²: denser regions of the electron cloud are represented by a higher density of points
3. Renders the resulting point cloud as a collection of small spheres using a pinhole perspective ray tracer
4. Colours each sphere by its normalised probability amplitude: blue for low density regions, red for high
5. Applies diffuse shading from a point light source

Camera control is interactive: click and drag to orbit, scroll to zoom.

Can cut the atom along he z=0 plane for better visibility where l->n

Shift+n to increase energy level, n to decrease, Shift+l to increase angular number,
l to decrease, Shift+m to increase magnetic number, m to decrease.

---

## Physics

The wavefunction separates as:
```
ψ_nlm(r, θ, φ) = R_nl(r) · Y_l^m(θ, φ)
```

**Radial part** `R_nl(r)` is computed via the associated Laguerre polynomial recurrence relation using double precision to avoid factorial overflow for higher quantum numbers. The normalisation uses `std::lgamma` for numerical stability.

**Angular part** `Y_l^m(θ, φ)` uses hardcoded real spherical harmonics up to `l=2`. Cartesian coordinates are converted to spherical at evaluation time. 

**Probability density** |ψ|² at each accepted point is stored and used both for rejection sampling and per-sphere colour encoding.

Atomic units are used throughout (a₀ = ħ = 1), so the bounding volume scales as `n²` and the camera distance scales accordingly.

---

## Monte Carlo sampling

Candidate points are drawn uniformly from a cube of side `4.0 · n` centred on the nucleus. A pre-pass over `NUM_CANDIDATES` random points establishes `psi_max` — the maximum observed |ψ|² within the volume. The main sampling pass then applies standard rejection sampling: a point at position **r** is accepted with probability `|ψ(**r**)|² / psi_max`.

The pre-pass and sampling pass use independent random sequences, which means `psi_max` can be a slight underestimate of the true maximum. This is intentional — it produces a modest natural stretch of the colour range across accepted points that improves visual clarity without requiring explicit gamma correction.

Accepted point count varies by orbital. Typical acceptance rates are 5–10% for low quantum numbers, lower for higher `n` where density is spread over a larger volume.

---

## Time Dependence
Can enable time dependence by uncommenting line 527 and 528, however performance
is very buggy and not recommended. Time dependence is simple and according to the
phase shift, which is a liberty taken purely for visual demonstration.

## GPU compute (OpenCL)

The render pipeline runs on the GPU via OpenCL 1.2, targeting Apple Silicon via the Metal-backed OpenCL implementation.

**Kernel architecture:**
- Single `render` kernel — one thread per pixel
- Sphere data uploaded once at startup as a `__constant` buffer
- Camera uploaded as a `__constant` struct (`CameraGPU`) and re-uploaded on camera movement
- Local memory tiling (`__local Sphere tile[TILE_SIZE]`) — threads within a work group cooperatively load sphere data from global memory into on-chip local memory before intersection testing, reducing redundant global memory fetches

**Host/device interop:**
CPU-side structs (`Camera`, `Sphere`, `Light`) use GLM. A dedicated `shared/bridge.h` defines matching GPU structs (`CameraGPU`, `SphereGPU`, `LightGPU`) using `cl_float4` throughout to guarantee correct memory layout — `cl_float3` is 16 bytes on most OpenCL implementations, not 12, which caused incorrect intensity reads before being fixed.

**Performance:**
- ~2.18ms render kernel time at 2000 spheres (1280×720)
- ~22ms at 20000 spheres: O(n) intersection tests per ray is the bottleneck at scale
    - Something that could be fixed, however cloud looks decent at smaller ranges
---

## Known limitations and future work

**Spatial subdivision** — the render kernel performs a flat O(n) intersection test against all spheres. For sphere counts above ~5000 this becomes the dominant cost. A uniform grid or BVH would reduce this to O(log n) per ray. Not yet implemented.

**Radial node visibility** — radial nodes (shells where R_nl = 0) are geometrically present in the point distribution as gaps, but are partially obscured by sphere radius overlap at typical point counts. Visualising the signed wavefunction ψ (rather than |ψ|²) with a diverging colour map (blue → black → red) would make nodes visually explicit.

**Time evolution** — stationary states have no time-dependent probability density (the phase factor cancels in |ψ|²). A superposition of two energy eigenstates would produce oscillating probability density at the beat frequency `ΔE/ħ`, which could be animated by passing a time uniform to an update kernel each frame. The infrastructure for this is partially in place.

**Sampling kernel** — the Monte Carlo pre-pass currently runs on the CPU. Moving it to a GPU kernel with `curand`-style per-thread LCG random number generation would parallelise wavefunction evaluation across hundreds of thousands of candidate points simultaneously.

**Higher orbitals** — spherical harmonics are hardcoded up to `l=2`. A general associated Legendre polynomial evaluator would support arbitrary `n`, `l`, `m`.

---

## Build

Requires SDL2 and GLM on the include path, and OpenCL (ships with macOS via `-framework OpenCL`).

```bash
make        # build
make clean  # remove binary and output.ppm
./atom_tracer <n> <l> <m>

# examples
./atom_tracer 1 0 0   # 1s orbital
./atom_tracer 2 1 0   # 2p orbital
./atom_tracer 3 2 1   # 3d orbital
```

**Controls:**
- Click and drag — orbit camera
- Scroll — zoom
- `S` — save current frame as `output.ppm`
- `Escape` — quit

---

## Dependencies

- SDL2
- GLM (header-only)
- OpenCL 1.2 (macOS: `-framework OpenCL`)
- C++17