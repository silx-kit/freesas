__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
from freesas.model import SASModel
import itertools

def assign_model(filename):
    """
    Create the molecule, on its canonical position
    
    Parameters
    ----------
    filename: name of the pdb file of the molecule
    """
    model = SASModel()
    model.read(filename)
    model.centroid()
    model.inertiatensor()
    model.canonical_position()
    model.centroid()
    model.inertiatensor()
    return model

def alignment(model1, model2):
    """
    Apply 8 combinations to model2 and select the one which minimize the distance between model1 and model2.
    The best position of the two models are save in two pdb files
    
    Parameters
    ----------
    model1 & model2: SASmodel, 2 molecules on their canonical position
    """
    combi = list(itertools.product((-1,1), repeat=3))
    combi = numpy.array(combi)
    
    d = None
    best = None
    rot_can = model2.canonical_rotate()
    mol = model2.atoms
    for i in range(8):
        rotation = rot_can
        for j in range(3):
            rotation[j] = rotation[j] * combi[i,j]
        
        model2.atoms = numpy.append(model2.atoms.T, numpy.ones((1,model2.atoms.shape[0])), axis=0)
        model2.atoms = numpy.dot(rotation, model2.atoms)
        model2.atoms = numpy.delete(model2.atoms, 3, axis=0)
        model2.atoms = model2.atoms.T
        
        distance = model1.dist(model2)
        
        if d == None:
            d = distance
        if distance <= d:
            d = distance
            best = i
        model2.atoms = mol
    
    if best==None:
        model1.save("model1.pdb")
        model2.save("model2.pdb")
    else:
        for j in range(3):
            rotation[j] = rotation[j] * combi[best,j]
            
        model2.atoms = numpy.append(model2.atoms.T, numpy.ones((1,model2.atoms.shape[0])), axis=0)
        model2.atoms = numpy.dot(rotation, model2.atoms)
        model2.atoms = numpy.delete(model2.atoms, 3, axis=0)
        model2.atoms = model2.atoms.T
    
        model1.save("model1.pdb")
        model2.save("model2.pdb")
        
    return d