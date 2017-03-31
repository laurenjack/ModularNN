"""Domain object used to keep track of the overall kernel density estimate
for the current epoch."""
class DensityStats:

    def __init__(self, fz, pz, df_distances, kernel_sums, refs, D, h, n, reg_eta):
        #The kde f(z) calculated for each reference, in each direction, across all data points
        #This will be updated each batch (r x dir matrix)
        self.fz = fz
        #The prior p(z) evaluated for each reference in each direction, hence pz is an r x dir
        #matrix too
        self.pz = pz
        #A batch-wise list gradients of f(z) w.r.t to the D^t.(Z - Zi) distances
        #between points. That is,  each is an r x dir x m tensor which holds
        #the gradient of f(z) with respect to the distance of the ith reference
        #from the kth observation in the jth direction. This quantity itself is
        #noticibly obtuse but it is required in this form to compute f'(z) as f(z)
        #changes each batch
        self.df_distances = df_distances
        #A batch-wise list of r x dir matricies which holds the sum of kernels
        #every reference, projected onto dir directions,
        # across m batches. These old sums are held so that we may remove
        # them from f(z) and replace them with the sum for the new zi's
        # of the updated batch
        self.kernel_sums = kernel_sums
        self.refs = refs
        self.D = D
        self.h = h
        self.n = n
        self.reg_eta = reg_eta



