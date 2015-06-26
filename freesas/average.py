__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
from freesas.model import SASModel

class Grid():
    def __init__(self, nbknots=None):
        self.inputs = []
        self.size = []
        self.nbknots = nbknots if nbknots is not None else 5000
        self.radius = None
        self.coordknots = []

    def __repr__(self):
        return "Grid with %i knots"%self.knots.shape[0]

    def spatial_extent(self):
        """
        Calculate the maximal extent of input models
        
        @return self.size: 6-list with x,y,z max and then x,y,z min
        """
        atoms = []
        for files in self.inputs:
            m = SASModel()
            m.read(files)
            if len(atoms)==0:
                atoms = m.atoms
            else:
                atoms = numpy.append(atoms, m.atoms, axis=0)
        
        coordmin = atoms.min(axis=0)
        coordmax = atoms.max(axis=0)
        self.size = [coordmax[0],coordmax[1],coordmax[2],coordmin[0],coordmin[1],coordmin[2]]
        
        return self.size

    def calc_radius(self):
        """
        Calculate the radius of each point of a hexagonal close-packed grid, 
        knowing the total volume and the number of knots in this grid.
        
        @return radius: the radius of each knot of the grid
        """
        if len(self.size)==0:
            self.spatial_extent()
        size = self.size
        dx = size[0] - size[3]
        dy = size[1] - size[4]
        dz = size[2] - size[5]
        volume = dx*dy*dz
        
        density = numpy.pi/(3*2**0.5)
        radius = ((3/(4*numpy.pi))*density*volume/self.nbknots)**(1.0/3)
        self.radius = radius
        
        return radius

    def make_grid(self):
        """
        """
        if len(self.size)==0:
            self.spatial_extent()
        if self.radius is None:
            self.calc_radius()
        
        radius = self.radius
        a = 2*radius
        h = (3**0.5/2)*a
        c = (8.0/3)**(1.0/3)*a
        
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
        
        while (zmin + z) <= zmax + a:
            zlist.append(z)
            z += 0.5*c
        while (ymin + y) <= ymax + a:
            ylist.append(y)
            y += h
        while (xmin + x) <= xmax + a:
            xlist.append(x)
            x += 0.5*a
        
        for z in zlist:
            for j in range(len(xlist)):
                x = xlist[j]
                if j % 2 == 0:
                    for y in ylist[0:-1:2]:
                        knots = numpy.append(knots, [[xmin+x, ymin+y, zmin+z, 0.0]], axis=0)
                else:
                    for y in ylist[1:-1:2]:
                        knots = numpy.append(knots, [[xmin+x, ymin+y, zmin+z, 0.0]], axis=0)
        
        knots = numpy.delete(knots, 0, axis=0)
        return knots

class AverModels():
    def __init__(self, filename=None, reference=None):
        self.inputfiles = []
        self.reference = reference if reference is not None else 0#position of reference model in the list of pdb files
        self.outputfile = filename if filename is not None else "aver-model.pdb"
        self.header = []
        self.radius = None
        self.atoms = []
        self.grid = None

    def __repr__(self):
        return "Average SAS model with %i atoms"%len(self.atoms)

    def models_pooling(self):
        """
        Pool the atoms of each input models in self.atoms
        
        @return self.atoms: coordinates of each atom considerated
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
    grid = Grid()
    grid.inputs = ["aligned-01.pdb", "aligned-02.pdb", "aligned-03.pdb", "aligned-04.pdb", "aligned-11.pdb"]
    grid.spatial_extent()
    grid.calc_radius()
    lattice = grid.make_grid()
    
    aver = AverModels()
    aver.inputfiles = ["aligned-01.pdb", "aligned-02.pdb", "aligned-03.pdb", "aligned-04.pdb", "aligned-11.pdb"]
    aver.models_pooling()
    aver.radius = grid.radius
    aver.grid = lattice
    print aver.atoms.shape[0]
    print aver.grid.shape[0]
    aver.assign_occupancy()
    print aver.grid
    nb = 0
    
    avermodel = numpy.empty((1,4), dtype="float")
    for i in range(aver.grid.shape[0]):
        if aver.grid[i,-1]>0:
            nb += 1
            atom = aver.grid[i,:].reshape((1,4))
            avermodel = numpy.append(avermodel, atom, axis=0)
    print nb
    avermodel = numpy.delete(avermodel, 0, axis=0)
    
    m = SASModel()
    m.read("filegrid.pdb")
    m.atoms = lattice
    m.save("filegrid.pdb")
    
    n = SASModel()
    n.read("filegrid.pdb")
    n.atoms = avermodel
    n.save("avermodel.pdb")
    
    
    print "DONE"