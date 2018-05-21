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
#from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
#from mutants import mutate_residue








def compute_local_frustration(native_file, nresidues=35, savedir=None, mutation=False, ndecoys=1000, pack_radius=10., mutation_scheme="simple", relax_native=False, rcutoff=0.5, pcutoff=0.8, use_hbonds=False, configurational_traj_file=None, configurational_dtraj=None, use_config_individual_pairs=False,  configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, remove_high=None, min_use=10, save_pairs=None):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create unique scratch dir for mpi runs
    if rank == 0:
        scratch_head_dir = "%s/mpi_scratch" % (os.getcwd())
        if not os.path.isdir(scratch_head_dir):
            os.mkdir(scratch_head_dir)

        scratch_dir = "%s/scratch_%f" % (scratch_head_dir, time.time())
        os.mkdir(scratch_dir)

        for i in range(1, size):
            comm.send(scratch_dir, dest=i, tag=3)

    else:
        scratch_dir = comm.recv(source=0, tag=3)

    comm.Barrier()

    if savedir is None:
        savedir = os.getcwd()

    if rank == 0:
        print "Computing Local Frustration"
        print "Using %d Threads" % size

        if not os.path.isdir(savedir):
            os.mkdir(savedir)

    if use_hbonds:
        weights = [1.0, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        order = [pyrt.fa_atr, pyrt.fa_rep, pyrt.fa_sol, pyrt.lk_ball_wtd, pyrt.fa_elec, pyrt.hbond_lr_bb, pyrt.hbond_sr_bb, pyrt.hbond_bb_sc, pyrt.hbond_sc]

    else:
        weights = [1.0, 0.55, 1.0, 1.0, 1.0, 1.25]
        order = [pyrt.fa_atr, pyrt.fa_rep, pyrt.fa_sol, pyrt.lk_ball_wtd, pyrt.fa_elec, pyrt.pro_close]

        weights = weights[:-1]
        order = order[:-1]
    assert len(weights) == len(order)
    scorefxn_custom = pyrt.ScoreFunction()
    for thing, wt in zip(order,weights):
        scorefxn_custom.set_weight(thing, wt)

    native_pose = pyr.pose_from_pdb(native_file)

    close_contacts, close_contacts_zero, contacts_scores = determine_close_residues_from_file(native_file, probability_cutoff=pcutoff, radius_cutoff=rcutoff)

    if relax_native:
        relaxer = ClassicRelax()
        relaxer.set_scorefxn(pyrt.get_fa_scorefxn())
        relaxer.apply(native_pose)

    native_pair_E = compute_pairwise(native_pose, scorefxn_custom, order, weights, use_contacts=close_contacts, nresidues=nresidues)

    if mutation:
        decoy_avg, decoy_sd = compute_mutational_pairwise_mpi(native_pose, scorefxn_custom, order, weights, ndecoys=ndecoys, nresidues=nresidues, pack_radius=pack_radius, mutation_scheme=mutation_scheme, use_contacts=close_contacts, contacts_scores=contacts_scores, remove_high=remove_high)
    else:
        decoy_avg, decoy_sd, decoy_list = compute_configurational_pairwise_mpi(scorefxn_custom, order, weights, native_file, configurational_traj_file, scratch_dir, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, use_config_individual_pairs=use_config_individual_pairs, native_contacts=close_contacts_zero, nresidues=nresidues, pcutoff=pcutoff, min_use=10)
        if rank == 0:
            if not use_config_individual_pairs:
                np.save("%s/decoy_E_list" % savedir, decoy_list)
                np.savetxt("%s/decoy_E_list.dat" % savedir, decoy_list)
            else:
                if save_pairs is not None:
                    for i in range(np.shape(save_pairs)[0]):
                        pidx = save_pairs[i][0]
                        pjdx = save_pairs[i][1]
                        this_E_list = decoy_list[pidx][pjdx]
                        np.savetxt("%s/decoy_E_list_%d-%d.dat" % (savedir, pidx, pjdx), this_E_list)

    if rank == 0:
        np.savetxt("%s/native_pairwise.dat" % savedir, native_pair_E)
        np.savetxt("%s/decoy_avg.dat" % savedir, decoy_avg)
        np.savetxt("%s/decoy_sd.dat" % savedir, decoy_sd)
