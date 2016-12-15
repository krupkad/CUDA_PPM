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

Our first round of analysis was to check the effects of various optimizations.
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

