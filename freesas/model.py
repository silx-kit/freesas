__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF" 

import numpy

class SASModel:
    def __init__(self):
        self.atoms = []
        self.radius = 1.0
        self.header = "" # header of the PDB file
    
    def __repr__(self):
        return "SAS model with %i atoms"%len(self.atoms)
    
    def read(self, filename):
        """
        read the PDB file
        extract coordinates of each dummy atom
        """
        header = []
        atoms = []
        for line in open(filename):
            if line.startswith("ATOM"):
                args = line.split()
                x = float(args[6])
                y = float(args[7])
                z = float(args[8])
                atoms.append([x, y, z])
            header.append(line)
        self.header = "".join(header) 
        self.atoms = numpy.array(atoms)
    
    def save(self, filename):
        """
        save the position of each dummy atom in a PDB file
        """
        nr = 0
        with open(filename, "w") as pdbout:
            for line in self.header:
                if line.startswith("ATOM"):
                    line = line[:30]+"%8.3f%8.3f%8.3f"%tuple(self.atoms[nr])+line[54:]
                    nr += 1
                pdbout.write(line)
    