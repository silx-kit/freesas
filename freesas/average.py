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
        files = self.inputfiles
        radius = []
        for i in files:
            m = SASModel()
            m.read(i)
            m.fineness()
            if len(self.atoms)==0:
                self.atoms = m.atoms
            else:
                numpy.append(self.atoms, m.atoms, axis=0)
            radius.append(0.5*m._fineness)
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
                        grid = numpy.array([[xlist[i], ylist[j], zlist[k], 1.0]], dtype="float")
                    else:
                        grid = numpy.append(grid, [[xlist[i], ylist[j], zlist[k], 1.0]], axis=0)
        if grid.shape[0] != knots:
            print "pb with grid lenght"
        self.grid = grid
        return grid