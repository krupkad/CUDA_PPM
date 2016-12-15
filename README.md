# Parametric Pseudo-Manifolds Using CUDA

## CIS 565 - Fall 2016

## Daniel Krupka, Gabe Naghi

![](img/knot_color.png)

[Parametric Pseudo-Manifolds](http://repository.upenn.edu/cis_reports/877/) (PPMs) are mathematical constructs that
can be used to contruct smooth approximations to data defined on a triangular mesh. Our goal was to implement PPMs
with CUDA, and use them to efficiently generate and manipulate additional detail for 3D models. 

## Overview
PPMs work by associating a smooth, local approximation of the surface to each vertex. Then, when the position
of a point P on the approximation is needed, the three nearest patches are sampled and weighted by their distance to P.

![](img/diagram.png)

We can also perform this kind of interpolation any kind of per-vertex numerical data, such as vertex normals and
texture UV coordinates. An important consequence is that this separates geometry and topology, i.e. the structure of the approximant mesh is independent of
its vertex positions, normals, etc.

## Implementation

Since the approximations are built from data that is local to each vertex, and all patches are processed identically,
the PPM algorithm is highly parallelizable. The algorithm's outline is as follows:

* Pre-processing:
  * Build two lists of half-edges, one sorted by origin, one sorted by face.
    Additionally, sort each set of half-edges with the same vertex in clockwise order.
  * Generate new topology (but not geometry). We subdivide each face.
  * Precalculate any constant data. We use Bezier surfaces, so this includes evaluating all necessary
    polynomials at all points of interest.
* Main Loop:
  * Get/calculate any changes to vertex data. We use properties of Bernstein polynomials to move vertices
    without needing to fully re-fit Bezier surfaces.
  * For each generated point, evaluate its PPM position/normal/UV.

## Performance Analysis

### Zeroth Round - CPU
A CPU implementation was provided by the authors of the original PPM paper. It was prohibitively slow to test, taking
~2s for a 400 vertex mesh, and ~10 minutes for 100k vertices.

### First Round - Optimization
Our first round of analysis was to check the effects of various optimizations. The test platform was a Quadro K620,
computing only the generated vertex positions. We focused on the effects of Bezier patch degree and vertex count,
and kept the subdivision level at 10 subfaces per face.
* Shared Memory: Sampling Bezier surfaces requires numerous repeated memory accesses, suggesting
  the use of shared memory to reduce read time.
* Textures: The tessellation pattern is known during preprocessing and never changes, so the local coordinates
  and polynomial values for each patch may be obtained faster if stored and accessed through CUDA's texture
  subsystem.
* Rank-1 Updating: Naively, the coefficients for each patch must be re-fit each time the vertices change. For a
  least-squares fit, this requires a large matrix multiplication. If the shape of each vertex's deformation is
  proportional to a singular vector of this matrix, this can be replaced by a less computationally intensive
  rank-1 update.

![](img/optim_plot.png)

Testing shows that, as expected, shared memory use substantially improved computation time with an ~35% speedup. Rank-1 updating
was next most significant, showing on the order of a 25% improvement. Finally, the use of textures did help,
but was nonsubstatial in comparison to other optimizations (~10%). The total improvement was approximately 60%. Also as expected, shared memory
helped more in the degree-8 case, as more redundant reads of Bezier coefficients from main memory were prevented.

![](img/runtime1_plot.png)
There was also a clear correlation between mesh size and execution time, as well as the degree of the Bezier patches.

### Second Round - Stress
We then included all eight vertex attributes (XYZ position, XYZ normal, UV texture), which necessitated a change in SM use. Originally, we
read and held all attributes' coefficients in memory at once. With eight attributes, however, the required amount of SM led to unacceptably
few blocks being scheduled, if a kernel launch succeeded at all. Now, only one set was read at a time, and we additionally parallelized over attribute index.
Tests were conducted on a C2075, using degree-4 Bezier patches. Numbers in the legend are the number of vertices in the mesh being tested.

![](img/runtime2_plot.png)

All runs were slower than before, due to ~2.5x more work being done to process the additional attributes. Runtime increased steadily
as more faces were generated. At peak, the C2075 was generating ~23.6 million triangles per second before exhausting memory or kernel launch time.
Notably, the implementation can generate geometry faster than it can process it -- it is substantially cheaper to manipulate a rough mesh and fill in
details from a PPM than to manipulate a detailed mesh of similar complexity. Also critically, deformation incurs nearly zero overhead (~10us) -- the
full runtime is heavily dominated by PPM computation and not base-mesh-only events.

Notably, shared memory either did not provide any improvement or produced a slow-down. Likely, the combination of parallelizing over attribute
and the amount of additional data combined to make the overhead of loading/unloading SM substantial enough to cause degradation. It is possible
that for "sufficiently large" inputs, SM might again become useful.

## Demos

![](img/knot_anim.gif)

Colors correspond to normal direction. Note that the normals are much smoother than would be allowed by the base mesh, shown in red.

![](img/knot_spot.png)

View of the subdivided topology, with visible shading. Smoothness of the PPM gives the light/dark boundary more refinement than the base mesh provides.


