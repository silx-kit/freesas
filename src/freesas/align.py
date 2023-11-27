__author__ = "Guillaume Bonamis"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import os
import sys
import numpy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from freesas.model import SASModel
import itertools
from scipy.optimize import fmin
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("log_freesas")


class InputModels:
    def __init__(self):
        self.inputfiles = []
        self.sasmodels = []
        self.rfactors = []
        self.rmax = None
        self.validmodels = []

    def __repr_(self):
        return "Preparation of %s models for alignment" % len(self.inputfiles)

    def assign_models(self, molecule=None):
        """
        Create SASModels from pdb files saved in self.inputfiles and saved them in self.models.
        Center of mass, inertia tensor and canonical parameters are computed for each SASModel.

        :param molecule: optional 2d array, coordinates of the atoms for the model to create
        :return self.models: list of SASModel
        """
        if not self.inputfiles and len(molecule) == 0:
            logger.error("No input files")

        if self.inputfiles:
            for inputpdb in self.inputfiles:
                model = SASModel()
                model.read(inputpdb)
                model.centroid()
                model.inertiatensor()
                model.canonical_parameters()
                self.sasmodels.append(model)
            if len(self.inputfiles) != len(self.sasmodels):
                logger.error("Problem of assignment\n%s models for %s files" % (len(self.sasmodels), len(self.inputfiles)))

        elif len(molecule) != 0:
            model = SASModel()
            model.atoms = molecule
            model.centroid()
            model.inertiatensor()
            model.canonical_parameters()
            self.sasmodels.append(model)

        return self.sasmodels

    def rcalculation(self):
        """
        Calculation the maximal value for the R-factors, which is the mean of all the R-factors of 
        inputs plus 2 times the standard deviation.
        R-factors are saved in the attribute self.rfactors, 1d array, and in percentage.

        :return rmax: maximal value for the R-factor 
        """
        if len(self.sasmodels) == 0:
            self.assign_models()
        models = self.sasmodels

        rfactors = numpy.empty(len(models), dtype="float")
        for i in range(len(models)):
            rfactors[i] = models[i].rfactor
        self.rfactors = 100.0 * rfactors

        rmax = self.rfactors.mean() + 2 * self.rfactors.std()
        self.rmax = rmax

        return rmax

    def models_selection(self):
        """
        Check if each model respect the limit for the R-factor

        :return self.validmodels: 1d array, 0 for a non valid model, else 1
        """
        if self.rmax is None:
            self.rcalculation()
        rmax = self.rmax

        validmodels = []
        for i in range(len(self.sasmodels)):
            rfactor = self.rfactors[i]
            if rfactor <= rmax:
                validmodels.append(1.0)
            else:
                validmodels.append(0.0)

        self.validmodels = numpy.array(validmodels, dtype="float")

        return self.validmodels

    def rfactorplot(self, filename=None, save=False):
        """
        Create a png file with the table of R factor for each model.
        A threshold is computed to discarded models with Rfactor>Rmax.

        :param filename: filename for the figure, default to Rfactor.png
        :param save: save automatically the figure if True, else show it
        :return fig: the wanted figures
        """
        if filename is None:
            filename = "Rfactor.png"
        if len(self.validmodels) == 0:
            self.models_selection()

        dammif_files = len(self.inputfiles)
        R = self.rfactors
        Rmax = self.rmax

        xticks = 1 + numpy.arange(dammif_files)
        fig = plt.figure(figsize=(7.5, 10))
        labels = [os.path.splitext(os.path.basename(self.inputfiles[i]))[0] for i in range(dammif_files)]

        ax2 = fig.add_subplot(1, 1, 1)
        ax2.set_title("Selection of dammif models based on R factor")
        ax2.bar(xticks - 0.5, R)
        ax2.plot([0.5, dammif_files + 0.5], [Rmax, Rmax], "-r", label="R$_{max}$ = %.3f" % Rmax)
        ax2.set_ylabel("R factor in percent")
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels, rotation=90)
        ax2.legend(loc=8)

        bbox_props = dict(fc="pink", ec="r", lw=1)
        for i in range(dammif_files):
            if not self.validmodels[i]:
                ax2.text(i + 0.95, Rmax / 2, "Discarded", ha="center", va="center", rotation=90, size=10, bbox=bbox_props)
                logger.info("model %s discarded, Rfactor > Rmax" % self.inputfiles[i])

        if save:
            fig.savefig(filename)
        else:
            fig.show()

        return fig


class AlignModels:
    """
    Used to align DAM from pdb files
    """

    def __init__(self, files, slow=True, enantiomorphs=True):
        """
        :param files: list of pdb files to read to create DAM
        :param slow: optimized every symmetry if True, else only optimized the best one
        :param enantiomorphs: take into account both enantiomorphs if True (i.e. inversion authorized)
        """
        self.slow = slow
        self.enantiomorphs = enantiomorphs
        self.inputfiles = files
        self.outputfiles = []
        self.models = []
        self.arrayNSD = None
        self.validmodels = []
        self.reference = None

    def __repr__(self):
        return "alignment process for %s models" % len(self.models)

    def assign_models(self):
        """
        Create SASModels from pdb files saved in self.inputfiles and saved them in self.models.
        Center of mass, inertia tensor and canonical parameters are computed for each SASModel.

        :return self.models: list of SASModel
        """
        for inputpdb in self.inputfiles:
            model = SASModel()
            model.read(inputpdb)
            model.centroid()
            model.inertiatensor()
            model.canonical_parameters()
            self.models.append(model)
        if len(self.inputfiles) != len(self.models):
            logger.error("Problem of assignment\n%s models for %s files" % (len(self.models), len(self.inputfiles)))

        return self.models

    def optimize(self, reference, molecule, symmetry):
        """
        Use scipy.optimize to optimize transformation parameters to minimize NSD

        :param reference: SASmodel
        :param molecule: SASmodel        
        :param symmetry: 3-list of +/-1
        :return p: transformation parameters optimized
        :return dist: NSD after optimization
        """
        p, dist, niter, nfuncalls, warmflag = fmin(reference.dist_after_movement, molecule.can_param, args=(molecule, symmetry), ftol=1e-4, maxiter=200, full_output=True, disp=False)
        if niter == 200:
            logger.debug("convergence not reached")
        else:
            logger.debug("convergence reach after %s iterations" % niter)
        return p, dist

    def alignment_sym(self, reference, molecule):
        """
        Apply 8 combinations to the molecule and select the one which minimize the distance between it and the reference.

        :param reference: SASModel, the one which do not move
        :param molecule: SASModel, the one wich has to be aligned
        :return combinaison: best symmetry to minimize NSD
        :return p: transformation parameters optimized if slow is true, unoptimized else
        """
        can_paramref = reference.can_param
        can_parammol = molecule.can_param

        ref_can = reference.transform(can_paramref, [1, 1, 1])
        mol_can = molecule.transform(can_parammol, [1, 1, 1])

        if self.slow:
            parameters, dist = self.optimize(reference, molecule, [1, 1, 1])
        else:
            parameters = can_parammol
            dist = reference.dist(molecule, ref_can, mol_can)
        combinaison = None

        for comb in itertools.product((-1, 1), repeat=3):
            if comb == (1, 1, 1):
                continue
            if not self.enantiomorphs and comb[0] * comb[1] * comb[2] == -1:
                continue

            sym = numpy.diag(comb + (1,))
            mol_sym = numpy.dot(sym, mol_can.T).T

            if self.slow:
                symmetry = [sym[0, 0], sym[1, 1], sym[2, 2]]
                p, d = self.optimize(reference, molecule, symmetry)
            else:
                p = can_parammol
                d = reference.dist(molecule, ref_can, mol_sym)

            if d < dist:
                dist = d
                parameters = p
                combinaison = comb
        if combinaison is not None:
            combinaison = list(combinaison)
        else:
            combinaison = [1, 1, 1]
        return combinaison, parameters

    def makeNSDarray(self):
        """
        Calculate the NSD correlation table and save it in self.arrayNSD

        :return self.arrayNSD: 2d array, NSD correlation table
        """
        models = self.models
        size = len(models)
        valid = self.validmodels
        self.arrayNSD = numpy.zeros((size, size), dtype="float")

        for i in range(size):
            if valid[i] == 1.0:
                reference = models[i]
            else:
                self.arrayNSD[i, :] = 0.00
                continue
            for j in range(size):
                if i == j:
                    self.arrayNSD[i, j] = 0.00
                elif i < j:
                    if valid[j] == 1.0:
                        molecule = models[j]
                        symmetry, p = self.alignment_sym(reference, molecule)
                        if self.slow:
                            dist = reference.dist_after_movement(p, molecule, symmetry)
                        else:
                            p, dist = self.optimize(reference, molecule, symmetry)
                    else:
                        dist = 0.00
                    self.arrayNSD[i, j] = self.arrayNSD[j, i] = dist
        return self.arrayNSD

    def plotNSDarray(self, rmax=None, filename=None, save=False):
        """
        Create a png file with the table of NSD and the average NSD for each model.
        A threshold is computed to segregate good models and the ones to exclude.

        :param rmax: threshold of R factor for the validity of a model
        :param filename: filename for the figure, default to nsd.png
        :param save: save automatically the figure if True, else show it
        :return fig: the wanted figures
        """
        if self.arrayNSD is None:
            self.makeNSDarray()
        if not self.reference:
            self.reference = self.find_reference()
        if filename is None:
            filename = "nsd.png"

        dammif_files = len(self.inputfiles)
        valid_models = self.validmodels
        labels = [os.path.splitext(os.path.basename(self.outputfiles[i]))[0] for i in range(dammif_files)]
        mask2d = (numpy.outer(valid_models, valid_models))
        tableNSD = self.arrayNSD * mask2d
        maskedNSD = numpy.ma.masked_array(tableNSD, mask=numpy.logical_not(mask2d))
        data = valid_models * (tableNSD.sum(axis=-1) / (valid_models.sum() - 1))  # mean for the valid models, excluding itself
        
        fig = plt.figure(figsize=(15, 10))
        xticks = 1 + numpy.arange(dammif_files)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # first subplot : the NSD table
        lnsd = []
        for i in range(dammif_files):
            for j in range(dammif_files):
                nsd = maskedNSD[i, j]
                if not maskedNSD.mask[i, j]:
                    ax1.text(i, j, "%.2f" % nsd, ha="center", va="center", size=12 * 8 // dammif_files)
                    ax1.text(j, i, "%.2f" % nsd, ha="center", va="center", size=12 * 8 // dammif_files)
                    if i != j:
                        lnsd.append(nsd)

        lnsd = numpy.array(lnsd)
        nsd_max = lnsd.mean() + lnsd.std()  # threshold for nsd mean

        ax1.imshow(maskedNSD, interpolation="nearest", origin="upper", cmap="YlOrRd", norm=matplotlib.colors.Normalize(vmin=min(lnsd)))
        ax1.set_title(u"NSD correlation table")
        ax1.set_xticks(range(dammif_files))
        ax1.set_xticklabels(labels, rotation=90)
        ax1.set_xlim(-0.5, dammif_files - 0.5)
        ax1.set_ylim(-0.5, dammif_files - 0.5)
        ax1.set_yticks(range(dammif_files))
        ax1.set_yticklabels(labels)

        # second subplot : the NSD mean for each model
        ax2.bar(xticks - 0.5, data)
        ax2.plot([0.5, dammif_files + 0.5], [nsd_max, nsd_max], "-r", label=u"NSD$_{max}$ = %.2f" % nsd_max)
        ax2.set_title(u"NSD between any model and all others")
        ax2.set_ylabel("Normalized Spatial Discrepancy")
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels, rotation=90)
        bbox_props = dict(fc="cyan", ec="b", lw=1)
        ax2.text(self.reference + 0.95, data[self.reference] / 2, "Reference", ha="center", va="center", rotation=90, size=10, bbox=bbox_props)
        ax2.legend(loc=8)

        bbox_props = dict(fc="pink", ec="r", lw=1)
        valid_number = 0
        for i in range(dammif_files):
            if data[i] > nsd_max:
                ax2.text(i + 0.95, data[self.reference] / 2, "Discarded", ha="center", va="center", rotation=90, size=10, bbox=bbox_props)
                logger.debug("model %s discarded, nsd > nsd_max" % self.inputfiles[i])
            elif not valid_models[i]:
                if rmax:
                    ax2.text(i + 0.95, data[self.reference] / 2, "Discarded, Rfactor = %s > Rmax = %s" % (100.0 * self.models[i].rfactor, rmax), ha="center", va="center", rotation=90, size=10, bbox=bbox_props)
                else:
                    ax2.text(i + 0.95, data[self.reference] / 2, "Discarded", ha="center", va="center", rotation=90, size=10, bbox=bbox_props)
            else:
                if valid_models[i] == 1.0:
                    valid_number += 1

        logger.debug("%s valid models" % valid_number)

        if save:
            fig.savefig(filename)
        else:
            fig.show()
        return fig

    def find_reference(self):
        """
        Find the reference model among the models aligned.
        The reference model is the one with lower average NSD with other models.
        
        :return ref_number: position of the reference model in the list self.models
        """
        if self.arrayNSD is None:
            self.makeNSDarray()
        if len(self.validmodels) == 0:
            logger.error("Validity of models is not computed")
        valid = self.validmodels
        valid = valid.astype(bool)

        averNSD = numpy.zeros(len(self.models))
        averNSD += sys.maxsize
        averNSD[valid] = ((self.arrayNSD.sum(axis=-1)) / (valid.sum() - 1))[valid]

        self.reference = averNSD.argmin()

        return self.reference

    def alignment_reference(self, ref_number=None):
        """
        Align all models in self.models with the reference one.
        The aligned models are saved in pdb files (names in list self.outputfiles)
        """
        if self.reference is None and ref_number is None:
            self.find_reference()
        
        ref_number = self.reference
        models = self.models
        reference = models[ref_number]
        for i in range(len(models)):
            if i == ref_number:
                continue
            else:
                molecule = models[i]
                symmetry, p = self.alignment_sym(reference, molecule)
                if not self.slow:
                    p, dist = self.optimize(reference, molecule, symmetry)
                molecule.atoms = molecule.transform(p, symmetry)  # molecule sent on its canonical position
                molecule.atoms = molecule.transform(reference.can_param, [1, 1, 1], reverse=True)  # molecule sent on reference position
                molecule.save(self.outputfiles[i])
        reference.save(self.outputfiles[ref_number])
        return 0

    def alignment_2models(self, save=True):
        """
        Align two models using the first one as reference.
        The aligned models are save in pdb files.

        :return dist: NSD after alignment
        """
        models = self.models
        reference = models[0]
        molecule = models[1]

        symmetry, p = self.alignment_sym(reference, molecule)
        if not self.slow:
            p, dist = self.optimize(reference, molecule, symmetry)

        molecule.atoms = molecule.transform(p, symmetry)
        molecule.atoms = molecule.transform(reference.can_param, [1, 1, 1], reverse=True)
        if self.slow:
            dist = reference.dist(molecule, reference.atoms, molecule.atoms)
        if save:
            molecule.save(self.outputfiles)

        return dist
