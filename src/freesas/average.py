__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
from freesas.model import SASModel


class Grid:
    """
    This class is used to create a grid which include all the input models
    """
    def __init__(self, inputfiles):
        """
        :param inputfiles: list of pdb files needed for averaging
        """
        self.inputs = inputfiles
        self.size = []
        self.nbknots = None
        self.radius = None
        self.coordknots = []

    def __repr__(self):
        return "Grid with %i knots"%self.nbknots

    def spatial_extent(self):
        """
        Calculate the maximal extent of input models
        
        :return self.size: 6-list with x,y,z max and then x,y,z min
        """
        atoms = []
        models_fineness = []
        for files in self.inputs:
            m = SASModel(files)
            if len(atoms)==0:
                atoms = m.atoms
            else:
                atoms = numpy.append(atoms, m.atoms, axis=0)
            models_fineness.append(m.fineness)
        mean_fineness = sum(models_fineness) / len(models_fineness)

        coordmin = atoms.min(axis=0) - mean_fineness
        coordmax = atoms.max(axis=0) + mean_fineness
        self.size = [coordmax[0],coordmax[1],coordmax[2],coordmin[0],coordmin[1],coordmin[2]]

        return self.size

    def calc_radius(self, nbknots=None):
        """
        Calculate the radius of each point of a hexagonal close-packed grid, 
        knowing the total volume and the number of knots in this grid.

        :param nbknots: number of knots wanted for the grid
        :return radius: the radius of each knot of the grid
        """
        if len(self.size)==0:
            self.spatial_extent()
        nbknots = nbknots if nbknots is not None else 5000
        size = self.size
        dx = size[0] - size[3]
        dy = size[1] - size[4]
        dz = size[2] - size[5]
        volume = dx * dy * dz

        density = numpy.pi / (3*2**0.5)
        radius = ((3 /( 4 * numpy.pi)) * density * volume / nbknots)**(1.0/3)
        self.radius = radius

        return radius

    def make_grid(self):
        """
        Create a grid using the maximal size and the radius previously computed.
        The geometry used is a face-centered cubic lattice (fcc).

        :return knots: 2d-array, coordinates of each dot of the grid. Saved as self.coordknots.
        """
        if len(self.size)==0:
            self.spatial_extent()
        if self.radius is None:
            self.calc_radius()

        radius = self.radius
        a = numpy.sqrt(2.0)*radius

        xmax = self.size[0]
        xmin = self.size[3]
        ymax = self.size[1]
        ymin = self.size[4]
        zmax = self.size[2]
        zmin = self.size[5]

        x = 0.0
        y = 0.0
        z = 0.0

        xlist = []
        ylist = []
        zlist = []
        knots = numpy.empty((1,4), dtype="float")
        while (zmin + z) <= zmax:
            zlist.append(z)
            z += a
        while (ymin + y) <= ymax:
            ylist.append(y)
            y += a
        while (xmin + x) <= xmax:
            xlist.append(x)
            x += a

        for i in range(len(zlist)):
            z = zlist[i]
            if i % 2 ==0:
                for j in range(len(xlist)):
                    x = xlist[j]
                    if j % 2 == 0:
                        for y in ylist[0:-1:2]:
                            knots = numpy.append(knots, [[xmin+x, ymin+y, zmin+z, 0.0]], axis=0)
                    else:
                        for y in ylist[1:-1:2]:
                            knots = numpy.append(knots, [[xmin+x, ymin+y, zmin+z, 0.0]], axis=0)
            else:
                for j in range(len(xlist)):
                    x = xlist[j]
                    if j % 2 == 0:
                        for y in ylist[1:-1:2]:
                            knots = numpy.append(knots, [[xmin+x, ymin+y, zmin+z, 0.0]], axis=0)
                    else:
                        for y in ylist[0:-1:2]:
                            knots = numpy.append(knots, [[xmin+x, ymin+y, zmin+z, 0.0]], axis=0)

        knots = numpy.delete(knots, 0, axis=0)
        self.nbknots = knots.shape[0]
        self.coordknots = knots

        return knots


class AverModels():
    """
    Provides tools to create an averaged models using several aligned dummy atom models
    """
    def __init__(self, inputfiles, grid):
        """
        :param inputfiles: list of pdb files of aligned models
        :param grid: 2d-array coordinates of each point of a grid, fourth column full of zeros
        """
        self.inputfiles = inputfiles
        self.models = []
        self.header = []
        self.radius = None
        self.atoms = []
        self.grid = grid

    def __repr__(self):
        return "Average SAS model with %i atoms"%len(self.atoms)

    def read_files(self, reference=None):
        """
        Read all the pdb file in the inputfiles list, creating SASModels.
        The SASModels created are save in a list, the reference model is the first model in the list.

        :param reference: position of the reference model file in the inputfiles list
        """
        ref = reference if reference is not None else 0
        inputfiles = self.inputfiles

        models = []
        models.append(SASModel(inputfiles[ref]))
        for i in range(len(inputfiles)):
            if i==ref:
                continue
            else:
                models.append(SASModel(inputfiles[i]))
        self.models = models

        return models

    def calc_occupancy(self, griddot):
        """
        Assign an occupancy and a contribution factor to the point of the grid.

        :param griddot: 1d-array, coordinates of a point of the grid
        :return tuple: 2-tuple containing (occupancy, contribution)
        """
        occ = 0.0
        contrib = 0
        for model in self.models:
            f = model.fineness
            for i in range(model.atoms.shape[0]):
                dx = model.atoms[i, 0] - griddot[0]
                dy = model.atoms[i, 1] - griddot[1]
                dz = model.atoms[i, 2] - griddot[2]
                dist = dx * dx + dy * dy + dz * dz
                add = max(1 - (dist / f), 0)
                if add != 0:
                    contrib += 1
                    occ += add
        return occ, contrib

    def assign_occupancy(self):
        """
        For each point of the grid, total occupancy and contribution factor are computed and saved.
        The grid is then ordered with decreasing value of occupancy.
        The fourth column of the array correspond to the occupancy of the point and the fifth to 
        the contribution for this point.

        :return sortedgrid: 2d-array, coordinates of each point of the grid
        """
        grid = self.grid
        nbknots = grid.shape[0]
        grid = numpy.append(grid, numpy.zeros((nbknots, 1), dtype="float"), axis=1)

        for i in range(nbknots):
            occ, contrib = self.calc_occupancy(grid[i, 0:3])
            grid[i, 3] = occ
            grid[i, 4] = contrib

        order = numpy.argsort(grid, axis=0)[:, -2]
        sortedgrid = numpy.empty_like(grid)
        for i in range(nbknots):
            sortedgrid[nbknots - i - 1, :] = grid[order[i], :]

        return sortedgrid

    def make_header(self):
        """
        Create the layout of the pdb file for the averaged model.
        """
        header = []
        header.append("Number of files averaged : %s\n"%len(self.inputfiles))
        for i in self.inputfiles:
            header.append(i + "\n")
        header.append("Total number of dots in the grid : %s\n"%self.grid.shape[0])

        decade = 1
        for i in range(self.grid.shape[0]):
            line = "ATOM         CA  ASP    1                                    20.00   2 201\n"
            line = line[:7] + "%4.i"%(i + 1) + line[11:]
            if not (i + 1) % 10:
                decade += 1
            line = line[:21] + "%4.i"%decade + line[25:]
            header.append(line)
        self.header = header
        return header

    def save_aver(self, filename):
        """
        Save the position of each occupied dot of the grid, its occupancy and its contribution 
        in a pdb file.

        :param filename: name of the pdb file to write
        """
        if len(self.header) == 0:
            self.make_header()
        assert self.grid.shape[-1] == 5

        nr = 0
        with open(filename, "w") as pdbout:
            for line in self.header:
                if line.startswith("ATOM"):
                    if nr < self.grid.shape[0] and self.grid[nr, 4] != 0:
                        coord = "%8.3f%8.3f%8.3f" % tuple(self.grid[nr, 0:3])
                        occ = "%6.2f" % self.grid[nr, 3]
                        contrib = "%2.f" % self.grid[nr, 4]
                        line = line[:30] + coord + occ + line[60:66] + contrib + line[68:]
                    else:
                        line = ""
                    nr += 1
                pdbout.write(line)
