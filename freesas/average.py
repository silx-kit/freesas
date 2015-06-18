__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"

import numpy
from freesas.model import SASModel

class AverModels():
    def __init__(self, filename=None):
        self.inputfiles = []
        self.outputfile = filename if filename is not None else "aver-model.pdb"
        self.header = []
        self.atoms = []
        self.radius = None
        self.size = []
        self.grid = None

    def __repr__(self):
        return "Average SAS model with %i atoms"%len(self.atoms)

    def models_pooling(self):
        """
        Pool the atoms of each input models in self.atoms
        
        @return self.atoms: coordinates of each atom considerated
        """
        radius = []
        for files in self.inputfiles:
            m = SASModel()
            m.read(files)
            m._calc_fineness()
            if len(self.atoms)==0:
                self.atoms = m.atoms
            else:
                self.atoms = numpy.append(self.atoms, m.atoms, axis=0)
            radius.append(0.5*m.fineness)
        self.radius = min(radius)
        return self.atoms

    def gridsize(self):
        """
        Calculate the maximal area occupied by models.
        
        @return self.size: 6-list with coordinates x,y,z max and x,y,z min
        """
        if len(self.atoms)==0:
            self.atoms = self.models_pooling()
        atoms = self.atoms
        
        coordmin = atoms.min(axis=0)
        coordmax = atoms.max(axis=0)
        self.size = [coordmax[0],coordmax[1],coordmax[2],coordmin[0],coordmin[1],coordmin[2]]
        
        return self.size

    def makegrid(self):
        """
        Create the new grid points using the area occupied by models.
        
        @return grid: 2d array, coordinates of each point of the grid, fourth column for the occupancy.
        """
        if not self.radius:
            self.models_pooling()
        if not self.size:
            self.size = self.gridsize()
        size = self.size
        xmax = size[0]
        ymax = size[1]
        zmax = size[2]
        xmin = size[3]
        ymin = size[4]
        zmin = size[5]
        radius = self.radius
        
        x = 0.0
        y = 0.0
        z = 0.0
        xlist = []
        ylist = []
        zlist = []
        while (xmin+x)<=xmax:
            xlist.append(xmin+x)
            x += radius
        while (ymin+y)<=ymax:
            ylist.append(ymin+y)
            y += radius
        while (zmin+z)<=zmax:
            zlist.append(zmin+z)
            z += radius
        knots = len(xlist)*len(ylist)*len(zlist)
        
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                for k in range(len(zlist)):
                    if i==j==k==0:
                        grid = numpy.array([[xlist[i], ylist[j], zlist[k], 0.0]], dtype="float")
                    else:
                        grid = numpy.append(grid, [[xlist[i], ylist[j], zlist[k], 0.0]], axis=0)
        if grid.shape[0] != knots:
            print "pb with grid lenght"
        self.grid = grid
        return grid

    def assign_occupancy(self):
        """
        Assign an occupancy for each point of the grid.
        Occupancy is the number of atoms closer to a point of the grid
        
        @return grid: 2d array, fourth column is occupancy of the point
        """
        if len(self.grid)==0:
            self.grid = self.makegrid()
        atoms = self.atoms
        grid = self.grid
        radius = self.radius
        
        for i in range(atoms.shape[0]):
            d = None
            num = None
            x1 = atoms[i, 0]
            y1 = atoms[i, 1]
            z1 = atoms[i, 2]
            for j in range(grid.shape[0]):
                x2 = grid[j, 0]
                y2 = grid[j, 1]
                z2 = grid[j, 2]
                h = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
                if h<=radius:
                    num = j
                    print j
                    continue
                elif not d or h<=d:
                    d = h
                    num = j
            grid[num, 3] += 1
        self.grid = grid
        return grid
    
    def keepcore(self, filename):
        """
        """
        grid = self.grid
        averdam = None
        for i in range(grid.shape[0]):
            if grid[i,3]>=5:
                if averdam is None:
                    averdam = grid[i,:]
                    averdam.shape = (1, 4)
                else:
                    coord = numpy.array([[grid[i,0], grid[i,1], grid[i,2], grid[i,3]]])
                    averdam = numpy.append(averdam, coord, axis=0)
        print averdam.shape
        avermodel = SASModel()
        avermodel.read(filename)
        avermodel.atoms = averdam
        avermodel.save(self.outputfile)

if __name__ == "__main__":
    aver = AverModels()
    aver.inputfiles = ["aligned-02.pdb", "aligned-03.pdb", "aligned-04.pdb", "aligned-05.pdb", "aligned-06.pdb", "aligned-07.pdb", "aligned-08.pdb", "aligned-10.pdb", "aligned-11.pdb", "aligned-12.pdb", "aligned-13.pdb", "aligned-14.pdb", "aligned-15.pdb", "aligned-16.pdb"]
    aver.models_pooling()
    print aver.atoms.shape[0]
    print aver.radius
    aver.gridsize()
    aver.makegrid()
    print aver.grid.shape[0]
    aver.assign_occupancy()
    aver.keepcore("aligned-02.pdb")
    print "DONE"