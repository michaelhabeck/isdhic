"""
Parser for various data.
"""
import numpy as np

from collections import defaultdict

class HiCData(object):
    """HiCData

    Simple class for storing and filtering contact data from single-cell
    HiC experiments.
    """
    def __init__(self, data):
        """HiCData

        This is a list of tuples specifying the indices of the loci that
        are in contact.

        Parameters
        ----------

        data : list of tuples
        """
        self.data = map(tuple, data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def add(self, (i,j)):
        self.data.append((i,j))
                
    def remove_self_contacts(self):
        """
        Remove contacts between one and the same locus. Self-contacts can
        occur due to mapping high-resolution contact data to a low-resolution
        representation of the chromatin fiber.
        """
        contacts = np.array(self.data)
        mask = contacts[:,0] != contacts[:,1]

        self.__init__(contacts[mask])
        
    def remove_redundant_contacts(self):
        """
        Remove contacts that are duplicated or equivalent (e.g. (1,2) is
        equivalent to (2,1)).
        """
        unique = []

        for i, j in self:

            i, j = min(i,j), max(i,j)
            if not (i,j) in unique:
                unique.append((i,j))

        self.__init__(unique)

    def coarsen(self, n_beads, chrsize):

        scale  = n_beads / float(chrsize)

        self.__init__((np.array(self.data) * scale).astype('i'))

class HiCParser(object):

    def __init__(self, filename, chromosome1=None, chromosome2=None):

        self.filename    = filename
        self.chromosome1 = chromosome1
        self.chromosome2 = chromosome2

    def parse(self):
        """
        Reads contacts from a text file
        """
        datasets = defaultdict(list)
        
        with open(self.filename) as f:

            header = f.readline().strip().split('\t')

            while 1:

                line = f.readline()

                if line == '': break

                chr1, i, chr2, j = line.split('\t')

                if self.chromosome1 and str(self.chromosome1) != chr1: continue
                if self.chromosome2 and str(self.chromosome2) != chr2: continue

                datasets[(chr1,chr2)].append((int(i),int(j)))

        for k, v in datasets.items():
            datasets[k] = HiCData(v)

        return datasets

