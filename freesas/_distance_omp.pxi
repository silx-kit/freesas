#OpenMP version of distance calculation

from cython import parallel
cimport openmp 

@cython.wraparound(False)
@cython.boundscheck(False)
def calc_distance(floating[:, :] atoms1, floating[:, :] atoms2, floating fineness1, floating fineness2):
    """
    Calculate the Normalized Spatial Discrepancy (NSD) between two molecules
    
    @param atoms1,atoms2: 2d-array with atom coordinates[[x,y,z],...]
    @param fineness1, fineness2: fineness of each molecule
    @return: NSD atoms1-atoms2
    """
    
    cdef:
        int i, j, size1 = atoms1.shape[0], size2 = atoms2.shape[0]
        int threadid, numthreads = openmp.omp_get_max_threads() 
        double d, d2, dx, dy, dz, x1, y1, z1
        double s1 = 0.0, s2 = 0.0, big = sys.maxsize
        double[:, ::1] min_col = numpy.zeros((numthreads, size2), numpy.float64) + big
    assert atoms1.shape[1] >= 3
    assert atoms2.shape[1] >= 3
    assert size1 > 0
    assert size2 > 0
    
    for i in parallel.prange(size1, nogil=True):
        threadid = parallel.threadid()
        x1 = atoms1[i, 0]
        y1 = atoms1[i, 1]
        z1 = atoms1[i, 2]
        d = big
        for j in range(size2):            
            dx = atoms2[j, 0] - x1
            dy = atoms2[j, 1] - y1
            dz = atoms2[j, 2] - z1
            d2 = dx * dx + dy * dy + dz * dz
            d = min(d, d2)
            min_col[threadid, j] = min(min_col[threadid, j], d2) 
        s1 += d
    for j in parallel.prange(size2, nogil=True):
        d = big
        for i in range(numthreads):
            d = min(d, min_col[i, j])
        s2 += d

    return sqrt(0.5 * ((1.0 / (size1 * fineness2 * fineness2)) * s1 + (1.0 / (size2 * fineness1 * fineness1)) * s2))


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_density(floating[:, :] atoms, floating dmax, int npt, floating r=0.0, bint hard=True):
    """
    Calculate the density rho(r)
    
    #TODO: formula for rigid sphere:
    A = (4*R+d)*(2*R-d)**2/16.0/R**3
    
    @param atoms: 2d-array with atom coordinates[[x,y,z],...]
    @param dmax: Diameter of the model
    @param npt: number of point in the density
    @param r: radius of an atom
    @param hard: use hard spheres model
    @return: 1d-array of 
    """
    
    cdef:
        int i, j, k, size = atoms.shape[0]        
        int threadid, numthreads = openmp.omp_get_max_threads()
        int width = 1 if hard else 2 
        floating d, dmax_plus, dx, dy, dz, x1, y1, z1
        floating delta, d_min, d_max, d1, den 
        double[:, ::1] tmp = numpy.zeros((numthreads, npt), numpy.float64)
        double[::1] out = numpy.zeros(npt, numpy.float64)
        double s
        
    assert atoms.shape[1] >= 3
    assert size > 0
    assert dmax > 0
    dmax_plus = dmax * (1.0 + numpy.finfo(numpy.float32).eps)
    delta = dmax_plus / npt
    
    for i in parallel.prange(size, nogil=True):
        threadid = parallel.threadid()
        x1 = atoms[i, 0]
        y1 = atoms[i, 1]
        z1 = atoms[i, 2]
        for j in range(size):
            dx = atoms[j, 0] - x1
            dy = atoms[j, 1] - y1
            dz = atoms[j, 2] - z1
            d = sqrt(dx * dx + dy * dy + dz * dz)
            d_min = max(0.0, d - width * r)
            d_max = min(dmax, d + width * r)
            for k in range(<int>(d_min / delta), <int>(d_max / delta)+1):
                if hard:
                    tmp[threadid, k] += hard_sphere(fabs(k * delta - d), r)
                else:
                    tmp[threadid, k] += soft_sphere(fabs(k * delta - d), r)
    for j in parallel.prange(npt, nogil=True):
        s = 0
        for i in range(numthreads):
            s = s + tmp[i, j]
        out[j] += s

    return numpy.asarray(out)
