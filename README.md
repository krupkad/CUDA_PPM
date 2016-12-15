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

