General introduction to FreeSAS
===============================

FreeSAS is a Python package with small angles scattering tools in a MIT
type license.


Introduction
------------

Freesas has been write as a re-implementation of some ATSAS parts in
Python for a better integration in the BM29 ESRF beam-line processing
pipelines. It provides functions to read SAS data from pdb files and to
handle them. Parts of the code are written in Cython and parallelized to
speed-up the execution.

Installation
------------


Usage
-----

Freesas as a library
....................

Here are presented some basics way to use FreeSAS as a library.
Some abbreviations:

- DA = Dummy Atom
- DAM = Dummy Atoms Model
- NSD = Normalized Spatial Discrepancy

Read a pdb file:

.. code-block:: python

    from freesas.model import SASModel
    model1 = SASModel()                #create SASModel class object
    model1.read("dammif-01.pdb")       #read the pdb file
    #these 2 lines can be replaced by model1 = SASModel("dammif-01.pdb")
    print model1.header                #print pdb file content
    print model1.atoms                 #print dummy atoms coordinates
    print model1.rfactor               #print R-factor of the DAM
    
    
Some informations are extracted of the model atoms coordinates:

- fineness : average distance between a DA and its first neighbours
- radius of gyration
- Dmax : DAM diameter, maximal distance between 2 DA of the DAM
- center of mass
- inertia tensor
- canonical parameters : 3 parameters of translation and 3 euler
  angles, define the transformation to applied to the DAM to put it
  on its canonical position (center of mass at the origin, inertia axis
  aligned with coordinates axis)

.. code-block:: python

    print model1.fineness          #print the DAM fineness
    print model1.Rg                #print the DAM radius of gyration
    print model1.Dmax              #print the DAM diameter
    model1.centroid()              #calculate the DAM center of mass
    print model1.com
    model1.inertiatensor()         #calculate the DAM inertiatensor
    print model1.inertensor
    model1.canonical_parameters()  #calculate the DAM canonical_parameters
    print model1.can_param

Other methods:

.. code-block:: python

    param1 = model1.can_param           #parameters for the transformation
    symmetry = [1,1,1]                 #symmetry for the transformation
    model1.transform(param1, symmetry)
    #return DAM coordinates after the transformation

    model2 = SASModel("dammif-02.pdb") #create a second SASModel
    model2.canonical_parameters
    atoms1 = model1.atoms
    atoms2 = model2.atoms
    model1.dist(model2, atoms1, atoms2)#calculate the NSD between models

    param2 = model2.can_param
    symmetry = [1,1,1]
    model1.dist_after_movement(param2, model2, symmetry)
    #calculate the NSD, first model on its canonical position, second
    model after a transformation with param2 and symmetry
    
		
Supcomb script
..............
