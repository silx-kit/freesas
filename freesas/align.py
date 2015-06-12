__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
from freesas.model import SASModel
import itertools
from scipy.optimize import fmin

class AlignModels:
    def __init__(self):
        self.slow = True
        self.enantiomorphs = True
        self.inputfiles = []
        self.outputfiles = []
        self.models = []
        self.arrayNSD = None
        self.reference = None
    
    def __repr__(self):
        return "alignment process for %s models"%len(self.inputfiles)
    
    def assign_models(self):
        """
        Create SASModels from pdb files saved in self.inputfiles and saved them in self.models.
        Center of mass, inertia tensor and canonical parameters are computed for each SASModel.
        
        @return self.models: list of SASModel
        """
        if not self.inputfiles:
            print "No input files"
        
        for inputpdb in self.inputfiles:
            model = SASModel()
            model.read(inputpdb)
            model.centroid()
            model.inertiatensor()
            model.canonical_parameters()
            self.models.append(model)
        if len(self.inputfiles) != len(self.models):
            print "Problem of assignment\n%s models for %s files"%(len(self.models), len(self.inputfiles))
        
        return self.models
    
    def optimize(self, reference, molecule, symmetry):
        """
        Use scipy.optimize to optimize transformation parameters to minimize NSD
        
        @param reference: SASmodel
        @param molecule: SASmodel        
        @param symmetry: 3-list of +/-1
        @return p: transformation parameters optimized
        @return dist: NSD after optimization
        """
        p, dist, niter, nfuncalls, warmflag = fmin(reference.dist_after_movement, molecule.can_param, args=(molecule, symmetry),ftol= 1e-4,  maxiter=200, full_output=True, disp=False)
        if niter==200: print "convergence not reached"
        else: print niter
        #logger.debug()
        return p, dist
    
    def alignment_sym(self, reference, molecule):
        """
        Apply 8 combinations to model2 and select the one which minimize the distance between model1 and model2.
        
        @param model1, model2: SASmodel, 2 molecules
        @param enantiomorphs: check the two enantiomorphs if true
        @param slow: optimize NSD for each symmetry if true
        @return combinaison: best symmetry to minimize NSD
        @return p: transformation parameters optimized if slow is true, unoptimized else
        """
        can_paramref = reference.can_param
        can_parammol = molecule.can_param
        
        ref_can = reference.transform(can_paramref,[1,1,1])
        mol_can = molecule.transform(can_parammol,[1,1,1])
        
        if self.slow:
            parameters, dist = self.optimize(reference, molecule, [1,1,1])
        else:
            parameters = can_parammol
            dist = reference.dist(molecule, ref_can, mol_can)
        combinaison = None
        
        for comb in itertools.product((-1,1), repeat=3):
            if comb == (1,1,1):
                continue
            if not self.enantiomorphs and comb[0]*comb[1]*comb[2] == -1:
                continue
            
            sym = numpy.diag(comb+(1,))
            mol_sym = numpy.dot(sym, mol_can.T).T
            
            if self.slow:
                symmetry = [sym[0,0], sym[1,1], sym[2,2]]
                p, d = self.optimize(reference, molecule, symmetry)
            else:
                p = can_parammol
                d = reference.dist(molecule, ref_can, mol_sym)
            
            if d < dist:
                dist = d
                parameters = p
                combinaison = comb
        if combinaison != None:
            combinaison = list(combinaison)
        else:
            combinaison = [1,1,1]
        return combinaison, parameters
    
    def makeNSDarray(self):
        """
        Calculate the NSD correlation table and save it in self.arrayNSD
        
        @return self.arrayNSD: 2d array, NSD correlation table
        """
        if not self.models:
            models = self.assign_models()
        else:
            models = self.models
        size = len(models)
        self.arrayNSD = numpy.empty((size, size), dtype="float")
        self.arrayparam = numpy.empty((size, size))
        self.arraysym = numpy.empty((size, size))
        
        for i in range(size):
            reference = models[i]
            for j in range(size):
                if i==j:
                    self.arrayNSD[i,j] = 0.00
                elif i<j:
                    molecule = models[j]
                    symmetry, p = self.alignment_sym(reference, molecule)
                    if self.slow:
                        dist = reference.dist_after_movement(p, molecule, symmetry)
                    else:
                        p, dist = self.optimize(reference, molecule, symmetry)
                    self.arrayNSD[i,j] = self.arrayNSD[j,i] = dist
        return self.arrayNSD
    
    def find_reference(self):
        """
        Find the reference model among the models aligned.
        The reference model is the one with lower average NSD with other models.
        
        @return ref_number: position of the reference model in the list self.models
        """
        ref_number = None
        if len(self.arrayNSD)==0:
            table = self.makeNSDarray()
        else:
            table = self.arrayNSD
        
        averNSD = table.mean(axis=1)
        miniNSD = averNSD.min()
        for i in range(len(averNSD)):
            if averNSD[i]==miniNSD:
                ref_number = i
                break
        if not ref_number and ref_number!=0:
            print "No reference model found"
        self.reference = ref_number
        
        return ref_number
    
    def alignment_reference(self, ref_number=None):
        """
        Align all models in self.models with the reference one.
        The aligned models are saved in pdb files (names in list self.outputfiles)
        """
        if not self.reference and not ref_number and self.reference!=0:
            ref_number = self.find_reference()
        else:
            ref_number = self.reference
        
        models = self.models
        reference = models[ref_number]
        for i in range(len(models)):
            if i==ref_number:
                continue
            else:
                molecule = models[i]
                symmetry, p = self.alignment_sym(reference, molecule)
                if not self.slow:
                    p, dist = self.optimize(reference, molecule, symmetry)
                molecule.atoms = molecule.transform(p, symmetry)
                molecule.save(self.outputfiles[i])
        reference.atoms = reference.transform(reference.can_param, [1,1,1])
        reference.save(self.outputfiles[ref_number])
        return 0
    
    def alignment_2models(self):
        """
        Align two models using the first one as reference.
        The aligned models are save in pdb files.
        
        @return dist: NSD after alignment
        """
        if not self.models:
            models = self.assign_models()
        else:
            models = self.models
        reference = models[0]
        molecule = models[1]
        
        symmetry, p = self.alignment_sym(reference, molecule)
        if not self.slow:
            p, dist = self.optimize(reference, molecule, symmetry)
        molecule.atoms = molecule.transform(p, symmetry)
        reference.atoms = reference.transform(reference.can_param, [1,1,1])
        if self.slow:
            dist = reference.dist(molecule, reference.atoms, molecule.atoms)
        reference.save(self.outputfiles[0])
        molecule.save(self.outputfiles[1])
        
        return dist