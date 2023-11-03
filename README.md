BaRe-ESA
=========

Description
-----------

This Python package provides a set of tools for the comparison, matching and interpolation of triangulated surfaces within the basis restricted elastic shape analysis setting. It allows specifically to solve the geodesic matching and distance computation problem between two surfaces with respect to a second order Sobolev metric. 


References
------------

@InProceedings{Hartman_2023_ICCV,
    author    = {Hartman, Emmanuel and Pierson, Emery and Bauer, Martin and Charon, Nicolas and Daoudi, Mohamed},
    title     = {BaRe-ESA: A Riemannian Framework for Unregistered Human Body Shapes},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {14181-14191}
}

Please cite [this paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hartman_BaRe-ESA_A_Riemannian_Framework_for_Unregistered_Human_Body_Shapes_ICCV_2023_paper.html) in your work.

Usage
-----------
Several scripts that demonstrate the usage of our main functions have been included in the demo folders. 



Dependencies
------------

BaRe-ESA is entirely written in Python while taking advantage of parallel computing on GPUs through CUDA. 
For that reason, it must be used on a machine equipped with an NVIDIA graphics card with recent CUDA drivers installed.
The code involves in addition the following Python libraries:

* Numpy 1.19.2 and Scipy 1.6.2
* Pytorch 1.4
* PyKeops 1.5 (https://www.kernel-operations.io/keops/index.html)
* Open3D 0.12.0

Note that Open3d is primarily used for surface reading, saving, visualization and simple mesh processing operations (decimation, subdivision...). Other libraries such as PyMesh could be used as potential replacement with relatively small modifications to our code.  


Licence
-------

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see http://www.gnu.org/licenses/.


Contacts
--------
    Emmanuel Hartman: ehartman(at)math.fsu.edu

    Emery Pierson:  (TBD)

    Martin Bauer:     bauer(at)math.fsu.edu

    Nicolas Charon:   ncharon1(at)jhu.edu


