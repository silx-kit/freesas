__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
from freesas.model import SASModel
import itertools
from scipy.optimize import fmin

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

def alignment_sym(model1, model2, enantiomorphs=True, slow=True):
    """
    Apply 8 combinations to model2 and select the one which minimize the distance between model1 and model2.
    
    @param model1, model2: SASmodel, 2 molecules
    @return combinaison: best symmetry to minimize NSD
    """
    assert model1.can_param and model2.can_param, "canonical parameters not computed"
    assert model1.enantiomer and model2.enantiomer, "symmetry constants not computed"
    can_param1 = model1.can_param
    can_param2 = model2.can_param
    
    combi = list(itertools.product((-1,1), repeat=3))
    combi = numpy.array(combi)
    
    mol1_can = model1.transform(can_param1,[1,1,1])#molecule 1 (reference) put on its canonical position
    mol2_can = model2.transform(can_param2,[1,1,1])#molecule 2 put on its canonical position
    
    if slow:
        symmetry = [1,1,1]
        arguments = (model2, symmetry)
        p0 = can_param2
        p = fmin(model1.dist_after_movement, p0, args=arguments,ftol= 1e-4,  maxiter=200)
        dist = model1.dist_after_movement(p, model2, symmetry)
    else:
        dist = model1.dist(model2, mol1_can, mol2_can)
    npermut = None
    
    same = numpy.eye(4, dtype="float")
    
    for i in range(combi.shape[0]-1):
        if not enantiomorphs:
            det = combi[i,0]*combi[i,1]*combi[i,2]
            if det==-1:
                break
        sym = same
        sym[0,0] = combi[i,0]
        sym[1,1] = combi[i,1]
        sym[2,2] = combi[i,2]
        
        mol2_sym = numpy.dot(sym, mol2_can.T).T
        
        if slow:
            symmetry = [sym[0,0], sym[1,1], sym[2,2]]
            arguments = (model2, symmetry)
            p0 = can_param2
            p = fmin(model1.dist_after_movement, p0, args=arguments,ftol= 1e-4,  maxiter=200)
            d = model1.dist_after_movement(p, model2, symmetry)
        else:
            d = model1.dist(model2, mol1_can, mol2_sym)
        
        if d < dist:
            dist = d
            npermut = i
            
    if npermut != None:
        combinaison = [combi[npermut,0], combi[npermut,1], combi[npermut,2]]
    else:
        combinaison = [1,1,1]
    return combinaison

def alignment_2models(filename1, filename2, optimize=True):
    """
    Align a SASModel with an other one and save the result in pdb files
    
    @param filename1 & 2: pdb files of the models, the first one is the reference
    @return dist: NSD after alignment
    """
    mol_ref = assign_model(filename1)
    mol = assign_model(filename2)
    
    symmetry = alignment_sym(mol_ref, mol)
    p0 = mol.can_param
    if optimize:
        arguments = (mol, symmetry)
        p = fmin(mol_ref.dist_after_movement, p0, args=arguments,ftol= 1e-4,  maxiter=200)
    else: p = p0
    
    mol.atoms = mol.transform(p, symmetry)
    mol_ref.atoms = mol_ref.transform(mol_ref.can_param, [1,1,1])
    dist = mol_ref.dist(mol, mol_ref.atoms, mol.atoms)
    mol.save("aligned-01.pdb")
    mol_ref.save("aligned-02.pdb")
    
    return dist