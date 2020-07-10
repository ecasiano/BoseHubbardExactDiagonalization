# Module to build Hamiltonian representing Bose-Hubbard lattices
# in 1,2,3D and calculate ground state energy via diagonalization.

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sympy.utilities.iterables import multiset_permutations,ordered_partitions
import scipy.sparse
import scipy.sparse.linalg


def build_adjacency_matrix(L,D,boundary_condition='pbc'):

    # Number of lattice sites
    M = L**D

    # Define the basis vectors
    a1_vec = np.array((1,0,0))
    a2_vec = np.array((0,1,0))
    a3_vec = np.array((0,0,1))

    # Norms of basis vectors
    a1 = np.linalg.norm(a1_vec)
    a2 = np.linalg.norm(a2_vec)
    a3 = np.linalg.norm(a3_vec)

    # Initialize array that will store the lattice vectors
    points = np.zeros(M,dtype=(float,D))

    # Build the lattice vectors
    ctr = 0 # iteration counter
    if D == 1:
        for i1 in range(L):
            points[i1] = i1*a1
    elif D == 2:
        for i1 in range(L):
            for i2 in range(L):
                points[ctr] = np.array((i1*a1,i2*a2))
                ctr += 1
    else: # D == 3
        for i1 in range(L):
            for i2 in range(L):
                for i3 in range(L):
                    points[ctr] = np.array((i1*a1,i2*a2,i3*a3))
                    ctr += 1

    # Initialize adjacency matrix
    A = np.zeros((M,M),dtype=int)

    # Set Nearest-Neighbor (NN) distance
    r_NN = a1

    # Build the adjacency matrix by comparing internode distances
    for i in range(M):
        for j in range(i+1,M):

            if boundary_condition=='pbc':
                A[i,j] = (np.linalg.norm(points[i]-points[j]) <= r_NN \
                or np.linalg.norm(points[i]-points[j]) == L-1)
            elif boundary_condition=='obc':
                A[i,j] = (np.linalg.norm(points[i]-points[j]) <= r_NN)

            # Symmetric elements
            A[j,i] = A[i,j]

    return A

'----------------------------------------------------------------------------------'

def bosonic_configurations(L,D,N):
    '''Input: 1D Lattice Size and Number of Bosons
    Output: All possible configurations of bosons'''
    
    #List that will store all configurations
    configurations = []
      
    #Store ordered partitions of N as a list
    partitions = list(ordered_partitions(N))
    
    for p in partitions:
        #BH Lattice containing a partition of N followed by zeros
        auxConfig = [0]*L**D
        auxConfig[0:len(p)] = p

        #Generate permutations based on current partition of N
        partitionConfigs = list(multiset_permutations(auxConfig))

        #Append permutations of current partition to list containing all configurations
        configurations += partitionConfigs
    
    #Promote configurations list to numpy array
    configurations = np.array(configurations)
      
    return configurations

'----------------------------------------------------------------------------------'

def bh_kinetic(configurations,t,A):
    '''Build the kinetic matrix K'''
    
    # From adjacency matrix, get nearest neighbor pairs
    nearest_neighbor_pairs = np.transpose(np.nonzero(A))
    
    # Get the size of the Hilbert Space
    basis_size = len(configurations)
    
    # Initialize lists that will store non-zero elements of K and indices
    row_ind = []
    col_ind = []
    K_elements = [] 
    
    # Evaluate all matrix elements of the kinetic operator
    for m in range(basis_size):
        bra = configurations[m] # <bra|
        for n in range(m+1,basis_size):
            kinetic_element=0
            ket = np.array(configurations[n]) # |ket>
            for j,i in nearest_neighbor_pairs:
                
                #print((configurations[n]==ket).all())
                
                # Particles originally in j,i
                n_j = ket[j]
                n_i = ket[i]
                  
                # Calculate N.N contribution to matrix element from <j,i> pair
                if n_j-1 != bra[j] or n_i+1 != bra[i]: continue        
                #elif n_j == 0: continue
                else:
                    ket_post = np.array(ket)
                    ket_post[j] = n_j-1
                    ket_post[i] = n_i+1
                    if (bra==ket_post).all(): # This is faster than np.array_equal()
                        kinetic_element += t*(np.sqrt(n_j*(n_i+1)))     
            
            # Save non-zero K elements and their row,column indices
            if kinetic_element != 0:
                row_ind.extend((m,n))
                col_ind.extend((n,m))
                K_elements.extend((kinetic_element,kinetic_element))   
                
    # Construct the sparse K matrix
    K = scipy.sparse.coo_matrix((K_elements,(row_ind,col_ind)))
    
    return -K

'----------------------------------------------------------------------------------'

def bh_diagonal(configurations,U,A):
    '''Build the diagonal of the diagonal matrix H_0'''
    
    # Get the size of the Hilbert Space
    basis_size = len(configurations)
    
    # Initialize diagonal energy matrix
    H_0 = np.zeros(basis_size)
    
    # Evaluate all matrix elements of the diagonal operator
    for m,ket in enumerate(configurations):
        diagonal_element = 0
        for i in range(len(ket)):
            diagonal_element += (U/2)*ket[i]*(ket[i]-1)

        # Fill kinetic matrix with corresponding element
        H_0[m] = diagonal_element
        
    return scipy.sparse.diags(H_0)    

'----------------------------------------------------------------------------------'

def Pn_BH(psi,configurations,lA):
    '''Input: Quantum State in second quantization bipartitioned into 
    subregions of size lA and L-lA'''
    '''Output: Probabilities of measuring states with particle number n in subregion A'''  
    L = np.shape(configurations[-1])[0]
    N = configurations[-1,0]
    hilbertSize = np.shape(configurations)[0]

    #Array to store probabilities
    Pn = np.zeros(N+1)
    for n in range(N+1):
        for i in range(hilbertSize):
            if np.sum(configurations[i][0:lA]) == n:
                Pn[n] += psi[i]**2

    return Pn

'----------------------------------------------------------------------------------'

# def boseHubbardHamiltonian(configurations,U,t,A):
#     '''Input: Set of all possible configurations of bosons on a 1D Lattice'''
    
#     #Store HilbertSpace Size
#     hilbertSize = np.shape(configurations)[0]
#     #print(hilbertSize)
    
#     #Store Lattice Size
#     L = np.shape(configurations)[1]
    
#     #Initialize Hamiltonian Matrix
#     H = np.zeros((hilbertSize,hilbertSize))
    
#     #Print out configurations to determine ordering of the basis
#     #print("Configurations: ", configurations)
#     #Fill in upper diagonal of the Hamiltonian
#     for i in range(hilbertSize):
#         bra = configurations[i]
#         for j in range(i,hilbertSize):
#             ket = configurations[j]
#             H[i,j] = bh_kinetic(t,A)
#             H[j,i] = H[i,j] #Use Hermiticity to fill up lower diagonal
# #             print("bra:",bra)
# #             print("ket:",ket)
# #             print(bh_kinetic(bra,ket))
            
#     return H

'----------------------------------------------------------------------------------'

'''---Command Line Arguments---'''

# Positional arguments
parser = argparse.ArgumentParser()

parser.add_argument("L",help="Number of sites per dimension",type=int)
parser.add_argument("N",help="Total number of bosons",type=int)
parser.add_argument("U",help="Interaction potential",type=float)   

# Optional arguments
parser.add_argument("--t",help="Hopping parameter (default: 1.0)",
                    type=float,metavar='\b')
parser.add_argument("--D",help="Lattice dimension (default: 1)",
                    type=int,metavar='\b')
parser.add_argument("--lA",help="Partition size (default: 1)",
                    type=int,metavar='\b')

# Parse arguments
args = parser.parse_args()

#Positional arguments
L = args.L
N = args.N
U = args.U

# Optional arguments (done this way b.c of some argparse bug) 
t = 1.0 if not(args.t) else args.t
D = int(1) if not(args.D) else args.D
lA = 1.0 if not(args.lA) else args.lA

'----------------------------------------------------------------------------------'

# Main

#BUG REPORT: ONLY WORKS FOR N <= L at the moment

'''---Bosons---'''

# Build adjacency matrix
A = build_adjacency_matrix(L,D,boundary_condition='pbc')
nearest_neighbor_pairs = np.transpose(np.nonzero(A))
print("(1/5)Building adjacency matrix...")
    
#Store all possible configurations of N bosons on L lattice sites
print("(2/5)Generating bosonic basis...")
configurations = bosonic_configurations(L,D,N)

# Build the kinetic energy matrix
print("(3/5)Building kinetic matrix...")
K = bh_kinetic(configurations,t,A)

    
# Build the diagonal energy matrix
print("(4/5)Building diagonal energy matrix...")
H_0 = bh_diagonal(configurations,U,A)

# Build the Hamiltonian by adding H = K + H_0
H = K+H_0

#Find ground state energy and state of the Hamiltonian
print("(5/5)Diagonalizing H...")
eigs,evecs = scipy.sparse.linalg.eigsh(H,which='SA',k=1) # SPARSE VERSION
egs = eigs[0]
psi = evecs[:,0] # Done this way b.c eigh return eigenvec matrix
print("Ground State Energy: ",egs,"\n")



