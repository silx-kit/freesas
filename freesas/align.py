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
    
    mol2 = model2.atoms
    
    dist = model1.dist(model2)
    npermut = None
    
    same = numpy.identity(4, dtype="float")
    
    for i in range(combi.shape[0]-1):
        sym = same
        sym[0,0] = combi[i,0]
        sym[1,1] = combi[i,1]
        sym[2,2] = combi[i,2]
        
        molsym = mol2.T
        molsym = numpy.dot(sym, molsym)
        molsym = molsym.T
        model2.atoms = molsym
        
        d = model1.dist(model2)
        
        if d < dist:
            dist = d
            npermut = i
    
    if npermut != None:
        sym = same
        sym[0,0] = combi[npermut,0]
        sym[1,1] = combi[npermut,1]
        sym[2,2] = combi[npermut,2]
        
        molsym = mol2.T
        molsym = numpy.dot(sym, molsym)
        molsym = molsym.T
        model2.atoms = molsym
    else:
        model2.atoms = mol2
    
    return dist