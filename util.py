import pyrosetta as pyr
import numpy as np
import os
import sys
import numpy.random as random
from mpi4py import MPI
import time

from pyrosetta import Pose
#from pyrosetta import MPIJobDistributor
from pyrosetta.rosetta.protocols.relax import ClassicRelax
from pyrosetta.toolbox import cleanATOM
from pyrosetta.toolbox import get_secstruct
from pyrosetta.toolbox import mutate_residue
from pyrosetta import teaching as pyrt

import mdtraj as md

# Disable Printing:
def block_print():
    sys.stdout = open(os.devnull, "w")

# Restore Printing:
def enable_print():
    sys.stdout = sys.__stdout__

def clean_pdb_files(decoy_dir, file_list):
    cwd = os.getcwd()
    os.chdir(decoy_dir)

    for pdb_file in file_list:
        cleanATOM(pdb_file) #outputs file_name.clean.pdb for file_name.pdb

    os.chdir(cwd)

def compute_average_and_sd(list_of_params):
    average_params = None
    n_threads = len(list_of_params)
    collected_params = []
    for mparams in list_of_params:
        if average_params is None:
            average_params = np.copy(mparams)
        else:
            average_params += mparams
    average_params /= float(n_threads)

    params_sd = np.zeros(np.shape(average_params))
    for mparams in list_of_params:
        diff = (mparams - average_params) ** 2
        params_sd += diff
    params_sd = np.sqrt(params_sd / float(n_threads))

    return average_params, params_sd

def compute_pairwise(pose, scorefxn, order, weights, nresidues=35, use_contacts=None):
    total_E = scorefxn(pose)
    pair_E = np.zeros((nresidues,nresidues))
    if use_contacts is None:
        these_contacts = []
        for idx in range(1, nresidues+1):
            for jdx in range(idx+1, nresidues+1):
                these_contacts.append([idx, jdx])
    else:
        these_contacts = use_contacts

    for contact in these_contacts:
        idx = contact[0]
        jdx = contact[1]
        emap = pyrt.EMapVector()
        scorefxn.eval_ci_2b(pose.residue(idx), pose.residue(jdx), pose, emap)
        this_E = 0.
        for thing,wt in zip(order,weights):
            this_E += emap[thing] * wt
        pair_E[idx-1, jdx-1] = this_E
    #assert (total_E - np.sum(pair_E)) < 0.01

    return pair_E

def get_possible_residues(pose):
    all_residues = ["G", "A", "V", "L", "I", "M", "F", "W", "P", "S", "T", "C", "Y", "N", "Q", "D", "E", "K", "R", "H"]
    sequence = pose.sequence()
    possible_residues = []
    for thing in sequence:
        possible_residues.append(thing)
    return possible_residues

def determine_close_residues_from_file(native_file, probability_cutoff=0.8, radius_cutoff=0.5):
    traj = md.load(native_file)

    use_pairs, use_pairs_zero, use_score = _determine_close_residues(traj, probability_cutoff=probability_cutoff, radius_cutoff=radius_cutoff)

    print "In file %s, found %d close residues" % (native_file, np.shape(use_pairs)[0])
    return use_pairs, use_pairs_zero, use_score

def _determine_close_residues(traj, probability_cutoff=0.8, radius_cutoff=0.5):
    n_frames = traj.n_frames
    top = traj.top
    collected_pairs = []
    for i in range(top.n_residues):
        for j in range(i+2, top.n_residues):
            collected_pairs.append([i,j])

    collected_pairs = np.array(collected_pairs).astype(int)
    distances, residue_pairs = md.compute_contacts(traj, contacts=collected_pairs)
    #distances, residue_pairs = md.compute_contacts(traj)

    n_contacts = np.shape(residue_pairs)[0]
    assert np.shape(distances)[1] == n_contacts

    count_matrix = np.zeros(np.shape(distances)).astype(float)
    count_matrix[np.where(distances <= radius_cutoff)] = 1.

    score_array = np.sum(count_matrix.astype(int), axis=0)
    max_indices = np.max(residue_pairs)
    score_matrix = np.zeros((max_indices+1,max_indices+1))
    for i_con in range(n_contacts):
        idx = residue_pairs[i_con, 0]
        jdx = residue_pairs[i_con, 1]
        score_matrix[idx, jdx] = score_array[i_con]
        score_matrix[jdx, idx] = score_array[i_con]
    score_by_residue = np.sum(score_matrix, axis=0)

    probability_matrix = np.sum(count_matrix, axis=0) / float(n_frames)

    assert np.shape(probability_matrix)[0] == n_contacts

    use_pairs = []
    use_pairs_zero = []
    use_score = []
    for i_con in range(n_contacts):
        if probability_matrix[i_con] > probability_cutoff:
            indices = residue_pairs[i_con, :]
            idx = indices[0]
            jdx = indices[1]
            use_pairs_zero.append(indices)
            use_pairs.append(indices + 1) # plus 1 converts to residue indices for rosetta
            use_score.append(score_by_residue[idx] + score_by_residue[jdx])

    return use_pairs, use_pairs_zero, use_score

def get_mpi_jobs(njobs, rank, size):
    job_numbers = range(rank, njobs, size)

    return job_numbers
