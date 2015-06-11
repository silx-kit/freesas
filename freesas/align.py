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
    @param enantiomorphs: check the two enantiomorphs if true
    @param slow: optimize NSD for each symmetry if true
    @return combinaison: best symmetry to minimize NSD
    """
    
    def optimize(reference, molecule, symmetry):
        """
        @param reference: SASmodel
        @param molecule: SASmodel
        @param symmetry: 3-list of +/-1
        @return
        """
        p, dist, niter, nfuncalls, warmflag = fmin(reference.dist_after_movement, molecule.can_param, args=(molecule, symmetry),ftol= 1e-4,  maxiter=200, full_output=True, disp=False)
        if niter==200: print "convergence not reached"
        else: print niter
        #logger.debug()
        return p, dist
    
    can_param1 = model1.can_param
    can_param2 = model2.can_param
    
    mol1_can = model1.transform(can_param1,[1,1,1])#molecule 1 (reference) put on its canonical position
    mol2_can = model2.transform(can_param2,[1,1,1])#molecule 2 put on its canonical position
    
    combinaison = None
    if slow:
        parameters, dist = optimize(model1, model2, [1,1,1])
    else:
        parameters = can_param2
        dist = model1.dist(model2, mol1_can, mol2_can)
    
    for comb in itertools.product((-1,1), repeat=3):
        if comb == (1,1,1):
            continue

        if not enantiomorphs and comb[0]*comb[1]*comb[2] == -1:
            continue

        sym = numpy.diag(comb+(1,))
        
        mol2_sym = numpy.dot(sym, mol2_can.T).T
        
        if slow:
            symmetry = [sym[0,0], sym[1,1], sym[2,2]]
            p, d = optimize(model1, model2, symmetry)
        else:
            p = can_param2
            d = model1.dist(model2, mol1_can, mol2_sym)
        
        if d < dist:
            dist = d
            parameters = p
            combinaison = comb
            
    if combinaison != None:
        combinaison = list(combinaison)
    else:
        combinaison = [1,1,1]
    return combinaison, parameters

def alignment_2models(filename1, filename2, enantiomorphs=True, slow=True):
    """
    Align a SASModel with an other one and save the result in pdb files
    
    @param filename1 & 2: pdb files of the models, the first one is the reference
    @return dist: NSD after alignment
    """
    mol_ref = assign_model(filename1)
    mol = assign_model(filename2)
    
    symmetry, p = alignment_sym(mol_ref, mol, enantiomorphs=enantiomorphs, slow=slow)
    
    if not slow:
        p, dist, niter, nfuncalls, warmflag = fmin(mol_ref.dist_after_movement, p, args=(mol, symmetry),ftol= 1e-4,  maxiter=200, full_output=True, disp=False)
    
    mol.atoms = mol.transform(p, symmetry)
    mol_ref.atoms = mol_ref.transform(mol_ref.can_param, [1,1,1])
    if slow:
        dist = mol_ref.dist(mol, mol_ref.atoms, mol.atoms)
    mol.save("aligned-01.pdb")
    mol_ref.save("aligned-02.pdb")
    
    return dist