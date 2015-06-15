__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy
import matplotlib
import matplotlib.pyplot as plot
from freesas.model import SASModel
import itertools
from scipy.optimize import fmin
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("log_freesas")

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

    def assign_models(self, molecule=None):
        """
        Create SASModels from pdb files saved in self.inputfiles and saved them in self.models.
        Center of mass, inertia tensor and canonical parameters are computed for each SASModel.
        
        @return self.models: list of SASModel
        """
        if not self.inputfiles and len(molecule)==0:
            logger.error("No input files")
        
        if self.inputfiles:
            for inputpdb in self.inputfiles:
                model = SASModel()
                model.read(inputpdb)
                model.centroid()
                model.inertiatensor()
                model.canonical_parameters()
                self.models.append(model)
            if len(self.inputfiles) != len(self.models):
                logger.error("Problem of assignment\n%s models for %s files"%(len(self.models), len(self.inputfiles)))
        
        elif len(molecule)!=0:
            model = SASModel()
            model.atoms = molecule
            model.centroid()
            model.inertiatensor()
            model.canonical_parameters()
            self.models.append(model)
        
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
        if niter==200: logger.debug("convergence not reached")
        else: logger.debug("convergence reach after %s iterations"%niter)
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

    def plotNSDarray(self):
        """
        """
        if len(self.arrayNSD)==0:
            self.arrayNSD = self.makeNSDarray()
        dammif_files = len(self.inputfiles)
        data = self.arrayNSD.sum(axis=-1)/dammif_files#average NSD for each model
        fig = plot.figure(figsize=(15, 10))
        
        xticks = 1 + numpy.arange(dammif_files)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.arrayNSD, interpolation="nearest", origin="upper")
        for i in range(dammif_files):
            for j in range(dammif_files):
                if i<j:
                    continue
                else:
                    nsd = self.arrayNSD[i,j]
                    ax1.text(i, j, "%.2f" % nsd, ha="center", va="center", size=12 * 8 // dammif_files)
                    ax1.text(j, i, "%.2f" % nsd, ha="center", va="center", size=12 * 8 // dammif_files)
        ax1.imshow(self.arrayNSD, interpolation="nearest", origin="upper")
        ax1.set_title(u"NSD correlation table")
        ax1.set_xticks(range(dammif_files))
        ax1.set_xticklabels([str(i) for i in range(1, 1 + dammif_files)])
        ax1.set_xlim(-0.5, dammif_files - 0.5)
        ax1.set_ylim(-0.5, dammif_files - 0.5)
        ax1.set_yticks(range(dammif_files))
        ax1.set_yticklabels([str(i) for i in range(1, 1 + dammif_files)])
        ax1.set_xlabel(u"Model number")
        ax1.set_ylabel(u"Model number")
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.bar(xticks - 0.5, data)
        ax2.set_title(u"NSD between any model and all others")
        ax2.set_ylabel("Normalized Spatial Discrepancy")
        ax2.set_xlabel(u"Model number")
        ax2.set_xticks(xticks)
        
        fig.savefig("nsd.png")
        return fig

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
            logger.error("No reference model found")
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

    def alignment_2models(self, save=True):
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
        if save:
            reference.save(self.outputfiles[0])
            molecule.save(self.outputfiles[1])
        
        return dist