import pyrosetta as pyr
import numpy as np
import os
import sys
import numpy.random as random
import multiprocessing
import multiprocessing.managers as mpmanagers
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

class ConstructConfigurationalMPI(object):
    def __init__(self, nresidues, top_file, configurational_traj_file, configurational_dtraj=None, configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, verbose=False, native_contacts=None):
        self.verbose = verbose
        self.nresidues = nresidues

        self.top_file = top_file
        self.configurational_traj_file = configurational_traj_file
        self.configurational_dtraj = configurational_dtraj
        self.configurational_parameters = configurational_parameters
        self.native_contacts = native_contacts
        self.decoy_r_cutoff = configurational_parameters["decoy_r_cutoff"]

        self._convert_parameters_into_list()
        self.poison_pill = False # set to False to terminmate the run job
        self._initialize_empty_results()

    def _convert_parameters_into_list(self):
        if self.configurational_dtraj is None:
            traj_check = md.iterload(self.configurational_traj_file, top=self.top_file)
            self.n_total_frames = traj_check.n_frames
            index_list = range(self.n_total_frames)
        else:
            # need to do some sorting
            high_cut = self.configurational_parameters["highcutoff"]
            low_cut = self.configurational_parameters["lowcutoff"]
            stride = self.configurational_parameters["stride_length"]
            n_indices = np.shape(self.configurational_dtraj)[0]

            index_list = []
            for i_check in range(0, n_indices, stride):
                value = self.configurational_dtraj[i_check]
                if (value >= low_cut) and (value <= high_cut):
                    index_list.append(i_check)

        self.inputs_collected = index_list
        if self.verbose:
            print "Computing %d frames" % (len(self.inputs_collected))

    def _initialize_empty_results(self):
        self.E_list = []
        self.E_avg = np.zeros((self.nresidues, self.nresidues))
        self.E_sd = np.zeros((self.nresidues, self.nresidues))

    def assign_E_results(self, avg, std):
        if self.native_contacts is None:
            for idx in range(self.nresidues):
                for jdx in range(self.nresidues):
                    self.E_avg[idx, jdx] = avg
                    self.E_sd[idx, jdx] = std
        else:
            for i_count in range(len(self.native_contacts)):
                idx = self.native_contacts[i_count][0]
                jdx = self.native_contacts[i_count][1]
                self.E_avg[idx, jdx] = avg
                self.E_sd[idx, jdx] = std

    def process_results_q(self, results_q):
        # take a queue as input, and then analyze the results
        # for configurational, anticipate a list of pair energies
        count = 0
        for results in results_q:
            count += 1
            self.E_list.append(results)

        E_avg, E_sd = compute_average_and_sd(self.E_list)

        self.assign_E_results(E_avg, E_sd)

        if self.verbose:
            print "Completed %d saves" % count

    def get_saved_results(self):
        return self.E_avg, self.E_sd

class ComputeConfigMPI(object):
    def __init__(self, thread_number, nresidues, traj_file, top_file, scorefxn, order, weights, scratch_dir, pcutoff=0.8, rcutoff=0.5):
        self.thread_number = thread_number
        print "Thread %d Starting" % self.thread_number

        self.nresidues = nresidues
        self.traj_file = traj_file
        self.top_file = top_file

        self.save_q = [] # list of pairwise energies
        self.scorefxn = scorefxn
        self.order = order
        self.weights = weights
        self.pcutoff = pcutoff
        self.rcutoff = rcutoff
        self.scratch_dir = scratch_dir

        self.still_going = True # default action is to keep going
        self.start_time = time.time()
        self.n_jobs_run = 0

        random.seed(int(time.time()) + int(self.thread_number*1000))

    def print_status(self):
        print "THREAD%2d --- %6f minutes: %6d Frames Complete" % (self.thread_number, (time.time() - self.start_time)/60., self.n_jobs_run)

    def clean_and_return_pose(self, index):
        save_pdb_file = "thread_%d.pdb" % (self.thread_number)
        rosetta_pdb_file = "thread_%d.clean.pdb" % (self.thread_number)

        traj = md.load_frame(self.traj_file, index, top=self.top_file)
        traj.save(save_pdb_file)
        print "here"
        cleanATOM(save_pdb_file)

        pose = pyr.pose_from_pdb(rosetta_pdb_file)

        return traj, pose

    def run(self, index):
        block_print()
        self.still_going = True
        if self.n_jobs_run % 1000 == 0:
            # print what step you are on
            enable_print()
            self.print_status()
            block_print()

        this_traj, this_pose = self.clean_and_return_pose(index)

        # get residue (1-indexed) contacts
        close_contacts, close_contacts_zero, contacts_scores = _determine_close_residues(this_traj, probability_cutoff=self.pcutoff, radius_cutoff=self.rcutoff)

        this_pair_E = compute_pairwise(this_pose, self.scorefxn, self.order, self.weights, use_contacts=close_contacts, nresidues=self.nresidues)

        for contact_index in close_contacts_zero:
            idx = contact_index[0] #use the 0-indexed
            jdx = contact_index[1] #use the 0-indexed
            self.save_q.append(this_pair_E[idx, jdx] )

        self.still_going = False
        enable_print()

        self.n_jobs_run += 1

        return

class ConstructConfigIndividualMPI(ConstructConfigurationalMPI):
    def __init__(self, *args, **kwargs):
        specific_args = ["min_use"]
        new_kwargs = {}
        for thing in kwargs:
            if thing in specific_args:
                pass
            else:
                new_kwargs[thing] = kwargs[thing]
        super(ConstructConfigIndividualMPI, self).__init__(*args, **new_kwargs)

        if "min_use" in kwargs:
            self.min_use = kwargs["min_use"]
        else:
            self.min_use = 0


    def _initialize_empty_results(self):
        self.E_list = [[[] for i in range(self.nresidues)] for j in range(self.nresidues)]
        self.E_avg = np.zeros((self.nresidues, self.nresidues))
        self.E_sd = np.zeros((self.nresidues, self.nresidues))

    def assign_E_results(self, idx, jdx, avg, std):
        self.E_avg[idx, jdx] = avg
        self.E_sd[idx, jdx] = std

    def process_results_q(self, results_q):
        # take a queue as input, and then analyze the results
        # for configurational, anticipate a list of pair energies
        count = 0
        for results in results_q:
            count += 1
            idx = results["idx"]
            jdx = results["jdx"]
            E = results["E"]
            self.E_list[idx][jdx].append(E)
            self.E_list[jdx][idx].append(E)

        zero_count = 0
        found_count = 0
        min_count = 0
        for idx in range(self.nresidues):
            for jdx in range(self.nresidues):
                found_count += 1
                this_list = self.E_list[idx][jdx]
                if len(this_list) == 0:
                    E_avg = 0
                    E_sd = 0
                    zero_count += 1
                elif len(this_list) < self.min_use:
                    E_avg = 0
                    E_sd = 0
                    min_count += 1
                else:
                    E_avg, E_sd = compute_average_and_sd(self.E_list[idx][jdx])
                self.assign_E_results(idx, jdx, E_avg, E_sd)

        if self.verbose:
            print "Completed %d saves, %f of the pairs had zero count while %f of the pairs had non-zero counts but were below the minimum threshold of %d" % (count, float(zero_count)/float(found_count), float(min_count)/float(found_count), self.min_use)

class ComputeConfigIndividualMPI(ComputeConfigMPI):
    def run(self, index):
        block_print()
        self.still_going = True
        if self.n_jobs_run % 1000 == 0:
            # print what step you are on
            enable_print()
            self.print_status()
            block_print()

        this_traj, this_pose = self.clean_and_return_pose(index)

        # get residue (1-indexed) contacts
        close_contacts, close_contacts_zero, contacts_scores = _determine_close_residues(this_traj, probability_cutoff=self.pcutoff, radius_cutoff=self.rcutoff)

        this_pair_E = compute_pairwise(this_pose, self.scorefxn, self.order, self.weights, use_contacts=close_contacts, nresidues=self.nresidues)

        for contact_index in close_contacts_zero:
            idx = contact_index[0] #use the 0-indexed
            jdx = contact_index[1] #use the 0-indexed
            save_dict = {"idx":idx, "jdx":jdx, "E": this_pair_E[idx,jdx]}
            self.save_q.append(save_dict)

        self.still_going = False
        enable_print()

        self.n_jobs_run += 1

        return

class ConstructMutationalMPI(object):
    def __init__(self, nresidues, use_contacts=None, contacts_scores=None, verbose=False):
        self.verbose = verbose
        self.nresidues = nresidues
        self.use_contacts = use_contacts
        self.contacts_scores=None
        self._convert_parameters_into_list()
        self.poison_pill = False # set to False to terminmate the run job
        self._initialize_empty_results()

    def _convert_parameters_into_list(self):
        if self.use_contacts is None:
            all_indices = []
            for idx in range(1, self.nresidues + 1):
                for jdx in range(idx + 1, self.nresidues + 1):
                    all_indices.append({"idx":idx, "jdx":jdx})

        else:
            all_indices = []
            for contact in self.use_contacts:
                assert contact[0] > 0
                assert contact[1] > 0
                all_indices.append({"idx":contact[0], "jdx":contact[1]})

            if self.contacts_scores is not None:
                sort_indices = np.argsort(self.contacts_scores)
                sort_indices = sort_indices[-1::-1] # reverse to descending order instead of ascending
                # do the actual sorting
                new_indices = []
                new_scores = []
                for sort_idx in sort_indices:
                    new_indices.append(all_indices[sort_idx])
                    new_scores.append(self.contacts_scores[sort_idx])
                all_indices = new_indices
                self.inputs_scores = new_scores

        self.inputs_collected = all_indices
        if self.verbose:
            print "Computing between %d pairs" % (len(self.inputs_collected))

    def _add_results_q(self):
        E_avg, E_std = self.process_results_q(self.save_q)

    def _initialize_empty_results(self):
        self.E_avg = np.zeros((self.nresidues, self.nresidues))
        self.E_sd = np.zeros((self.nresidues, self.nresidues))

    def process_results_q(self, results_q):
        # take a queue as input, and then analyze the results

        count = 0
        print_every = ((self.nresidues) ** 2 ) / 20
        for results in results_q:
            if self.verbose and (count % print_every == 0):
                print "Completed %d saves" % count
            count += 1
            """
            idx = results[0] # still 1-indexed
            jdx = results[1] # still 1-indexed
            average = results[2]
            sd = results[3]
            """
            idx = results["idx"] # still 1-indexed
            jdx = results["jdx"] # still 1-indexed
            average = results["average"]
            sd = results["sd"]

            zidx = idx - 1
            zjdx = jdx - 1

            self.E_avg[zidx, zjdx] = average
            self.E_sd[zidx, zjdx] = sd

        if self.verbose:
            print "Completed %d saves" % count

    def get_saved_results(self):
        return self.E_avg, self.E_sd

class ComputePairMPI(object):
    def __init__(self, thread_number, pair_list, native_pose, scorefxn, order, weights, ndecoys, pack_radius=10., mutation_scheme="simple", remove_high=None):
        self.thread_number = thread_number
        print "Thread %d Starting" % self.thread_number

        self.pair_list = pair_list
        self.save_q = []
        self.native_pose = native_pose
        self.scorefxn = scorefxn
        self.order = order
        self.weights = weights
        self.ndecoys = ndecoys
        self.pack_radius = pack_radius
        self.still_going = True # default action is to keep going
        self.start_time = time.time()
        self.n_jobs_run = 0
        self.possible_residues = get_possible_residues(self.native_pose)

        self.remove_high = remove_high

        random.seed(int(time.time()) + int(self.thread_number*1000))

        if mutation_scheme == "simple":
            print "Repacking locally with radius %f" % self.pack_radius
            self.mutate_residues_and_change = self.mutate_simple
        elif mutation_scheme == "repack_all":
            print "Repacking all side-chain atoms"
            self.mutate_residues_and_change = self.mutate_repack
        elif mutation_scheme == "relax_all":
            print "Relaxing side-chain and backbone parameters."
            self.mutate_residues_and_change = self.mutate_relax

    def mutate_residue_pair(self, idx, jdx, possible_residues):
        new_res1 = random.choice(possible_residues)
        new_res2 = random.choice(possible_residues)
        new_pose = Pose()
        new_pose.assign(self.native_pose)
        mutate_residue(new_pose, idx, new_res1, pack_radius=0)
        mutate_residue(new_pose, jdx, new_res2, pack_radius=0)
        return new_pose

    def mutate_relax(self, idx, jdx, possible_residues):
        new_pose = self.mutate_residue_pair(idx, jdx, possible_residues)
        relaxer = ClassicRelax()
        relaxer.set_scorefxn(pyrt.get_fa_scorefxn())
        relaxer.apply(new_pose)

        return new_pose

    def mutate_repack(self, idx, jdx, possible_residues):
        new_res1 = random.choice(possible_residues)
        new_res2 = random.choice(possible_residues)
        new_pose = Pose()
        new_pose.assign(self.native_pose)
        mutate_residue(new_pose, idx, new_res1, pack_radius=0)
        mutate_residue(new_pose, jdx, new_res2, pack_radius=50)
        """

        task = pyr.standard_packer_task(new_pose)
        task.restrict_to_repacking()
        pack_mover = PackRotamersMover(generic_scorefxn, task)
        pack_mover.apply(new_pose)
        """

        return new_pose

    def mutate_simple(self, idx, jdx, possible_residues):
        new_res1 = random.choice(possible_residues)
        new_res2 = random.choice(possible_residues)
        new_pose = Pose()
        new_pose.assign(self.native_pose)
        mutate_residue(new_pose, idx, new_res1, pack_radius=self.pack_radius)
        mutate_residue(new_pose, jdx, new_res2, pack_radius=self.pack_radius)

        return new_pose

    def print_status(self):
        print "THREAD%2d --- %6f minutes: %6d Pairs Complete" % (self.thread_number, (time.time() - self.start_time)/60., self.n_jobs_run)

    def run(self, list_of_index):
        block_print()
        self.still_going = True
        for index in list_of_index:
            new_E = None
            if self.n_jobs_run % 10 == 0:
                # print what step you are on
                enable_print()
                self.print_status()
                block_print()

            new_params = self.pair_list[index]
            idx = new_params["idx"] # 1-indexed
            jdx = new_params["jdx"] # 1-indexed
            all_E = []
            for i_decoy in range(self.ndecoys):
                new_pose = self.mutate_residues_and_change(idx, jdx, self.possible_residues)
                emap = pyrt.EMapVector()
                self.scorefxn.eval_ci_2b(new_pose.residue(idx), new_pose.residue(jdx), new_pose, emap)
                this_E = 0.
                for thing,wt in zip(self.order, self.weights):
                    this_E += emap[thing] * wt
                all_E.append(this_E)

            if self.remove_high is not None:
                temp_E = np.array(all_E)
                new_E = temp_E[np.where(temp_E < self.remove_high)]
            else:
                new_E = all_E
            this_avg, this_std = compute_average_and_sd(new_E)
            self.save_q.append({"idx":idx, "jdx":jdx, "average":this_avg, "sd":this_std})
            #self.save_q.put([idx, jdx, this_avg, this_std])

            self.n_jobs_run += 1

        self.still_going = False
        enable_print()

        return

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

def compute_mutational_pairwise_mpi(pose, scorefxn, order, weights, ndecoys=1000, nresidues=35, pack_radius=10., mutation_scheme="simple", use_contacts=None, contacts_scores=None, remove_high=None):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    analysis_object = ConstructMutationalMPI(nresidues, use_contacts=use_contacts, contacts_scores=contacts_scores)

    analyze_pairs = analysis_object.inputs_collected
    n_analyze = len(analyze_pairs)
    new_computer = ComputePairMPI(rank, analyze_pairs, pose, scorefxn, order, weights, ndecoys, pack_radius=pack_radius, mutation_scheme=mutation_scheme, remove_high=remove_high)

    job_indices = get_mpi_jobs(n_analyze, rank, size)
    new_computer.run(job_indices)

    new_computer.print_status()

    # wait until all jobs finish
    comm.Barrier()

    # send block
    if rank == 0:
        print "Finished All Calculations"
        all_results = []
        all_results.append(new_computer.save_q)
        for i in range(1, size):
            results = comm.recv(source=i, tag=3)
            all_results.append(results)
    else:
        comm.send(new_computer.save_q, dest=0, tag=3)

    # process results block
    if rank == 0:
        for results in all_results:
            analysis_object.process_results_q(results)

    E_avg, E_std = analysis_object.get_saved_results()

    comm.Barrier()

    return E_avg, E_std

def compute_configurational_pairwise_mpi(scorefxn, order, weights, top_file, configurational_traj_file, scratch_dir, configurational_dtraj=None, configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, nresidues=35, pcutoff=0.8, native_contacts=None, use_contacts=None, contacts_scores=None, use_config_individual_pairs=False, min_use=10):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        use_verbose = True
    else:
        use_verbose = False
    if use_config_individual_pairs:
        analysis_object = ConstructConfigIndividualMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, min_use=min_use, verbose=use_verbose)

        new_computer = ComputeConfigIndividualMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, pcutoff=0.8, rcutoff=analysis_object.decoy_r_cutoff)
    else:
        analysis_object = ConstructConfigurationalMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, verbose=use_verbose)

        new_computer = ComputeConfigMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, pcutoff=0.8, rcutoff=analysis_object.decoy_r_cutoff)

    analyze_indices = analysis_object.inputs_collected
    n_analyze = len(analyze_indices)

    job_indices = get_mpi_jobs(n_analyze, rank, size)

    cwd = os.getcwd()
    os.chdir(scratch_dir)
    for index in job_indices:
        new_computer.run(index)
    os.chdir(cwd)

    new_computer.print_status()

    # wait until all jobs finish
    comm.Barrier()

    # send AND process results block
    if rank == 0:
        print "Finished All Calculations"
        analysis_object.process_results_q(new_computer.save_q)
        for i in range(1, size):
            results = comm.recv(source=i, tag=3)
            analysis_object.process_results_q(results)
    else:
        comm.send(new_computer.save_q, dest=0, tag=3)

    # process results block

    E_avg, E_std = analysis_object.get_saved_results()

    comm.Barrier()

    return E_avg, E_std, analysis_object.E_list

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
