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
    def __init__(self, inputfiles, outputfile=None, reference=None):
        """
        :param inputfiles: list of pdb files of aligned models
        :param outputfile: name of the output pdb file, aver-model.pdb by default
        :param reference: position of the reference model in the inputfile list, first one by default
        """
        self.inputfiles = inputfiles
        self.reference = reference if reference is not None else 0
        self.outputfile = outputfile if outputfile is not None else "aver-model.pdb"
        self.header = []
        self.radius = None
        self.atoms = []
        self.grid = None

    def __repr__(self):
        return "Average SAS model with %i atoms"%len(self.atoms)

    def models_pooling(self):
        """
        Pool the atoms of each input model in self.atoms
        
        :return self.atoms: coordinates of each atom considerated
        """
        for files in self.inputfiles:
            m = SASModel()
            m.read(files)
            if len(self.atoms)==0:
                self.atoms = m.atoms
            else:
                self.atoms = numpy.append(self.atoms, m.atoms, axis=0)
        return self.atoms

    def trilin_interp(self, atom, gridpoint):
        """
        """
        radius = self.radius
        lattice_len = 2*radius
        lattice_surf = 2*radius*lattice_len
        lattice_vol = 2*radius*lattice_surf

        x = atom[0]
        y = atom[1]
        z = atom[2]
        x0 = gridpoint[0]
        y0 = gridpoint[1]
        z0 = gridpoint[2]

        xd = abs(x-x0)
        yd = abs(y-y0)
        zd = abs(z-z0)

        if xd>=2*radius or yd>=2*radius or zd>=2*radius:
            fact = 0.0

        elif xd==0 or yd==0 or zd==0:
            if xd==yd==zd==0:
                fact = 1.0
            
            elif xd==yd==0 or yd==zd==0 or xd==zd==0:
                if xd != 0:
                    dist = xd
                elif yd != 0:
                    dist = yd
                else:
                    dist = zd
                fact = dist/(lattice_len)

            else:
                if xd == 0:
                    surf = yd * zd
                elif yd == 0:
                    surf = xd * zd
                else:
                    surf = xd * yd
                fact = surf/(lattice_surf)
        else:
            vol = xd*yd*zd
            fact = vol/(lattice_vol)
        return fact

    def assign_occupancy(self):
        """
        Assign an occupancy for each point of the grid.
        
        @return grid: 2d array, fourth column is occupancy of the point
        """
        if len(self.grid)==0:
            self.grid = self.makegrid()
        atoms = self.atoms
        grid = self.grid

        for i in range(atoms.shape[0]):
            for j in range(grid.shape[0]):
                fact = self.trilin_interp(atoms[i], grid[j])/len(self.inputfiles)
                grid[j, 3] += fact

        order = numpy.argsort(grid, axis=0)[:,-1]
        sortedgrid = numpy.empty_like(grid)
        for i in range(grid.shape[0]):
            sortedgrid[grid.shape[0]-i-1,:] = grid[order[i], :]

        self.grid = sortedgrid
        return sortedgrid

if __name__ == "__main__":
    inputfiles = ["damaver.pdb"]
    grid = Grid(inputfiles)
    grid.spatial_extent()
    grid.calc_radius()
    print grid.radius
    lattice = grid.make_grid()
    print grid.nbknots

    m = SASModel()
    m.read("filegrid.pdb")
    m.atoms = lattice
    m.save("filegrid.pdb")

    print "DONE"