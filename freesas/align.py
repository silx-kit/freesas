__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
from freesas.model import SASModel
import itertools

def assign_model(filename):
    """
    Create the molecule, calculate its center of mass, inertia tensor and canonical parameters
    
    @param filename: name of the pdb file of the molecule
    """
    model = SASModel()
    model.read(filename)
    model.centroid()
    model.inertiatensor()
    model.canonical_parameters()
    return model

def alignment(model1, model2):
    """
    Apply 8 combinations to model2 and select the one which minimize the distance between model1 and model2.
    The best position of the two models are save in two pdb files.
    Save in model2.enantiomer the best combination
    
    @param model1, model2: SASmodel, 2 molecules on their canonical position
    @return dist: distance between model1 and model2 after the alignment
    """
    assert model1.can_param and model2.can_param, "canonical parameters not computed"
    assert model1.enantiomer and model2.enantiomer, "symmetry constants not computed"
    can_param1 = model1.can_param
    can_param2 = model2.can_param
    
    combi = list(itertools.product((-1,1), repeat=3))
    combi = numpy.array(combi)
    
    mol1_can = model1.transform(can_param1,[1,1,1])#molecule 1 (reference) put on its canonical position
    mol2_can = model2.transform(can_param2,[1,1,1])#molecule 2 put on its canonical position
    
    dist = model1.dist(model2, mol1_can, mol2_can)
    npermut = None
    
    same = numpy.eye(4, dtype="float")
    
    for i in range(combi.shape[0]-1):
        sym = same
        sym[0,0] = combi[i,0]
        sym[1,1] = combi[i,1]
        sym[2,2] = combi[i,2]
        
        mol2_sym = numpy.dot(sym, mol2_can.T).T
        
        d = model1.dist(model2, mol1_can, mol2_sym)
        
        if d < dist:
            dist = d
            npermut = i
            
    if npermut != None:
        combinaison = [combi[npermut,0], combi[npermut,1], combi[npermut,2]]
    else:
        combinaison = [1,1,1]
    return combinaison