from .util import *
from .mutational import ConstructMutationalMPI, ComputePairMPI, ConstructMutationalSingleMPI, ComputeSingleMPI
from .configurational import ConstructConfigurationalMPI, ComputeConfigMPI, ConstructConfigIndividualMPI, ComputeConfigIndividualMPI, ConstructConfigSingleResidueMPI, ComputeConfigSingleResidueMPI
from .pose_manipulator import FrusPose
import datetime

def _func_gauss(x, mu, sigma, total=1.):
    if sigma == 0:
        sigma = 10 ** -16
    scale = total / math.sqrt(2. * math.pi * (sigma**2))
    exponent = -1. * ((x - mu)**2) / (2. * (sigma**2))

    results = scale * np.exp(exponent)

    return results


def compute_gaussian_and_chi(decoyE, spacing=0.2):

    n_decoys = float(np.shape(decoyE)[0])
    if n_decoys > 0:
        max_value = ((math.ceil(np.max(np.abs(decoyE)) / spacing)) * spacing) + spacing
        ebins = np.arange(-max_value-(0.5*spacing), max_value+spacing, spacing)
        avg = np.sum(decoyE) / n_decoys
        sd = np.sqrt(np.sum((decoyE - avg)**2 )/n_decoys)
        hist_values, bin_edges = np.histogram(decoyE, bins=ebins)
        sd_values = np.sqrt(hist_values.astype(float))
        center_values = (bin_edges[1:] + bin_edges[:-1]) * 0.5
        true_values = _func_gauss(center_values, avg, sd, total=n_decoys*spacing)
        chi_pieces = ((hist_values - true_values)**2) / (sd_values ** 2)
        chi_pieces[np.where(hist_values==0)] = 0
        chi = np.sum(chi_pieces) / float(np.shape(center_values)[0])
    else:
        chi = 0
        avg = 0
        sd = 1

    return chi, avg, sd

class BookKeeper(object):
    def __init__(self, native_file, nresidues, savedir=None, use_hbonds=False, relax_native=False, rcutoff=0.6, pcutoff=0.8, mutate_traj=None, delete_traj=None, repack_radius=10, compute_all_neighbors=False, min_use=0):
        self.native_file = native_file
        if savedir is None:
            savedir = os.getcwd()

        self.savedir = savedir
        self.use_hbonds = use_hbonds
        self.relax_native = relax_native
        self.rcutoff = rcutoff
        self.pcutoff = pcutoff
        self.nresidues = nresidues

        self.mutate_traj = mutate_traj
        self.delete_traj = delete_traj
        self.repack_radius = repack_radius
        self.compute_all_neighbors = compute_all_neighbors
        self.min_use = min_use

        self.decoy_avg = None
        self.decoy_sd = None
        self.decoy_list = None
        self.decoy_list_array = None
        self.this_native_indices = []

        self.initialize_rosetta()

        native_trajs = md.load(native_file)
        self.n_native_structures = native_trajs.n_frames

        self.initialize_basic()
        if self.size > self.n_native_structures:
            skip = self.n_native_structures
        else:
            skip = self.size
        if self.rank < self.n_native_structures:
            # this should do the heavy lifting, prepare simulations and save them
            for i in range(self.n_native_structures)[self.rank::skip]:
                this_prefix = "initial_native_%d" % (i)
                this_native_file = "%s/%s.pdb" % (self.scratch_dir, this_prefix)
                this_native_file_cleaned = "%s/%s.clean.pdb" % (self.scratch_dir, this_prefix)

                native_trajs[i].save(this_native_file)
                clean_pdb_files(self.scratch_dir, ["%s.pdb" % this_prefix])
                self.prepare_simulations(this_native_file_cleaned, i)
                self.this_native_indices.append(i)

        self.comm.Barrier()

        #this next function transfers and does all the moving in the end
        self.initialize_average_results()

    def initialize_average_results(self):
        # all the native results (don't want to recompute)
        self.all_native_pair_E = []
        self.all_single_residue_avg_E = []
        self.all_native_pose = []
        self.all_native_fpose = []

        for native_index in range(self.n_native_structures):
            native_pose = pyr.pose_from_pdb("%s/initial_native_%d.clean.pdb" % (self.scratch_dir, native_index))
            native_fpose = FrusPose(native_pose, repack_radius=self.repack_radius)
            new_native_pose = pyr.pose_from_pdb("%s/final_native_index%d.pdb" % (self.scratch_dir, native_index))
            native_fpose = native_fpose.duplicate_nochange_new_pose(new_native_pose)
            native_fpose.add_change_history(mutation_list=self.mutate_traj, deletion_ranges=self.delete_traj)

            self.all_native_pose.append(new_native_pose)
            self.all_native_fpose.append(native_fpose)
            self.all_native_pair_E.append(np.load("%s/native_pair_E_index%d.npy" % (self.scratch_dir, native_index)))
            self.all_single_residue_avg_E.append(np.load("%s/single_residue_avg_E_index%d.npy" % (self.scratch_dir, native_index)))
            self.process_close_contacts(np.load("%s/close_contacts_zero_index%d.npy" % (self.scratch_dir, native_index)))

            # check and maintain attributes from the native_fpose
            if self.true_size is None:
                self.true_size = native_fpose.old_nresidues
            else:
                assert self.true_size == native_fpose.old_nresidues

            if self.native_fpose_mutation_list is None:
                self.native_fpose_mutation_list = native_fpose.mutation_list

            if self.native_fpose_deletion_ranges is None:
                self.native_fpose_deletion_ranges = native_fpose.deletion_ranges

            if self.native_fpose_current_indices is None:
                self.native_fpose_current_indices = native_fpose.current_indices

            if self.final_nresidues is None:
                self.final_nresidues = native_fpose.nresidues

        self.native_pair_E = np.zeros((self.nresidues,self.nresidues))
        self.single_residue_avg_E = np.zeros((self.nresidues))
        for i in range(self.n_native_structures):
            self.native_pair_E += self.all_native_pair_E[i]
            self.single_residue_avg_E += self.all_single_residue_avg_E[i]

        self.native_pair_E /= self.n_native_structures
        self.single_residue_avg_E /= self.n_native_structures

    def initialize_basic(self):
        savedir = self.savedir
        use_hbonds = self.use_hbonds

        comm = MPI.COMM_WORLD

        rank = comm.Get_rank()
        size = comm.Get_size()

        self.comm = comm
        self.rank = rank
        self.size = size

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

        self.scorefxn_custom = scorefxn_custom
        self.order = order
        self.weights = weights
        self.scratch_dir = scratch_dir

        # attributes from native_fpose to check and confirm are identical
        self.true_size = None
        self.final_nresidues = None
        self.native_fpose_mutation_list = None
        self.native_fpose_deletion_ranges = None
        self.native_fpose_current_indices = None

        # residue pairs in close proximity in any frame
        self.close_contacts = [] # 1-indexed for later (because of rosetta)
        self.close_contacts_array = np.zeros((self.nresidues,self.nresidues))
        # list of 1-0 arrays denoting close residue pairs in each frame.
        self.all_close_contacts_arrays = []

    def prepare_simulations(self, native_file, native_index):
        relax_native = self.relax_native
        rcutoff = self.rcutoff
        pcutoff = self.pcutoff
        nresidues = self.nresidues

        # from the initialize basic part
        rank = self.rank
        scratch_dir = self.scratch_dir
        scorefxn_custom = self.scorefxn_custom
        order = self.order
        weights = self.weights

        native_pose = pyr.pose_from_pdb(native_file)
        native_fpose = FrusPose(native_pose, repack_radius=self.repack_radius)

        final_dump_file = "%s/final_native_index%d.pdb" % (self.scratch_dir, native_index)

        if (self.mutate_traj is None) and (self.delete_traj is None):
            pass
        else:
            print "Mutating the structure"
            dump_file = "%s/mutated_pdb_%d.pdb" % (scratch_dir, self.rank)
            native_fpose.modify_protein(mutation_list=self.mutate_traj, deletion_ranges=self.delete_traj)
            native_fpose.dump_to_pdb(dump_file)

            new_native_pose = pyr.pose_from_pdb(dump_file)
            native_fpose = native_fpose.duplicate_nochange_new_pose(new_native_pose)

            native_pose = new_native_pose

        if relax_native:
            #raise IOError("Option is not implemented for multiple threads or multiple mutations/deletions")
            dump_relaxed_file = "%s/relaxed_pdb_%d.pdb" % (scratch_dir, self.rank)
            relaxer = ClassicRelax()
            relaxer.set_scorefxn(pyrt.get_fa_scorefxn())
            relaxer.apply(native_pose)
            native_pose.dump_pdb(dump_relaxed_file)

            new_native_pose = pyr.pose_from_pdb(dump_relaxed_file)
            native_fpose = native_fpose.duplicate_nochange_new_pose(new_native_pose)
            native_pose = new_native_pose

        native_fpose.dump_to_pdb(final_dump_file)

        close_contacts, close_contacts_zero, contacts_scores =  determine_close_residues_from_file(final_dump_file, probability_cutoff=pcutoff, radius_cutoff=rcutoff)
        #debug
        #print native_pose.sequence()
        #raise
        #debug
        if self.compute_all_neighbors:
            native_pair_E = compute_pairwise_allinteractions(native_pose, scorefxn_custom, order, weights, use_contacts=close_contacts, nresidues=native_fpose.nresidues) # account for deletions
        else:
            native_pair_E = compute_pairwise(native_pose, scorefxn_custom, order, weights, use_contacts=close_contacts, nresidues=native_fpose.nresidues) # account for deletions

        single_residue_tot_E = np.zeros(native_fpose.nresidues)
        single_residue_count = np.zeros(native_fpose.nresidues)
        """
        # compute across all contact pairs. Makes NO SENSE if you have rcutoff>0.6
        for pair_cont in close_contacts: # note, close_contacts is 1-indexed
            idx = pair_cont[0] - 1
            jdx = pair_cont[1] - 1
            pair_E = native_pair_E[idx,jdx]
            single_residue_tot_E[idx] += pair_E
            single_residue_tot_E[jdx] += pair_E
            single_residue_count[idx] += 1
            single_residue_count[jdx] += 1
        """
        for i_res in range(native_fpose.nresidues):
            this_row = native_pair_E[i_res, :]
            for value in this_row:
                if value != 0:
                    single_residue_tot_E[i_res] += value
                    single_residue_count[i_res] += 1
        single_residue_count[np.where(single_residue_count == 0)] = 1000000
        single_residue_avg_E = single_residue_tot_E / single_residue_count

        np.save("%s/native_pair_E_index%d" % (scratch_dir, native_index), native_pair_E)
        np.save("%s/single_residue_avg_E_index%d" % (scratch_dir, native_index), single_residue_avg_E)

        np.save("%s/close_contacts_zero_index%d" % (scratch_dir, native_index), close_contacts_zero)


    def process_close_contacts(self, close_contacts_zero):
        # close_contacts_zero is zero-indexed
        this_contacts = np.zeros((self.nresidues,self.nresidues))
        for thing in close_contacts_zero:
            i = thing[0]
            j = thing[1]
            self.close_contacts_array[i,j] = 1
            self.close_contacts_array[j,i] = 1
            this_contacts[i,j] = 1
            this_contacts[j,i] = 1

        self.all_close_contacts_arrays.append(this_contacts)

        self.close_contacts = []
        for idx in range(self.nresidues):
            for jdx in range(idx+1, self.nresidues):
                if self.close_contacts_array[idx,jdx] == 1:
                    self.close_contacts.append([idx+1, jdx+1]) # this has to be 1-indexed

    def get_possible_native(self, idx, jdx):
        #idx and jdx are 0-indexed
        native_check = [val[idx,jdx] for val in self.all_close_contacts_arrays]

        good_natives = []
        for structure, check in zip(self.all_native_pose, native_check):
            if check == 1:
                good_natives.append(structure)

        return good_natives

    def initialize_rosetta(self, verbose=False):
        comm = MPI.COMM_WORLD

        rank = comm.Get_rank()
        size = comm.Get_size()
        kargs = {}
        kargs["extra_options"] = " -seed_offset %s" % (rank*10000)

        if verbose:
            pass
        else:
            kargs["set_logging_handler"] = "logging"
            #pyr.init(set_logging_handler="logging", **kargs)

        pyr.init(**kargs)

    def remap_pairE(self, data, size):
        # re-map the final matrix to new dimensions
        new_pair_E = np.zeros((size,size))
        for i in range(np.shape(data)[0]):
            for j in range(np.shape(data)[1]):
                # TODO save curent_indices for all native_fpose
                idx = self.native_fpose_current_indices[i]
                jdx = self.native_fpose_current_indices[j]
                value = data[i,j]

                new_pair_E[idx, jdx] = value

        return new_pair_E

    def remap_singleE(self, data, size):
        # re-map the final matrix to new dimensions
        new_sing_E = np.zeros(size)
        for i in range(np.shape(data)[0]):
            idx = self.native_fpose_current_indices[i]
            value = data[i]

            new_sing_E[idx] = value

        return new_sing_E

    def get_basic_info(self):
        info_str = ""
        today = datetime.datetime.today()
        info_str += "for : %s, on %s \n" % (self.savedir, today)
        info_str += "original native_file = %s\n" % (self.native_file)
        info_str += "repack_radius = %s\n" % str(self.repack_radius)
        info_str += "rcutoff = %0.2f \n" % (self.rcutoff)
        info_str += "pcutoff = %0.2f \n" % (self.pcutoff)
        info_str += "nresidues = %0.2f \n" % (self.nresidues)
        info_str += "use_hbonds = %s \n" % (str(self.use_hbonds))
        info_str += "relax_native = %s \n" % (str(self.relax_native))
        info_str += "mutation_list = %s \n" % (str(self.native_fpose_mutation_list))
        info_str += "deletion_ranges = %s \n" % (str(self.native_fpose_deletion_ranges))

        return info_str

    def save_basic_info(self, info=None):
        f = open("%s/info.txt" % (self.savedir), "w")
        f.write(self.get_basic_info())
        f.write("\n")
        f.write("Process Specific Details:\n")
        f.write(info)
        f.close()

        # dump a copy of this particular native.pdb file for future reference
        for idx,native_fpose in enumerate(self.all_native_fpose):
            native_fpose.dump_to_pdb("%s/native_%d.pdb" % (self.savedir, idx))

    def save_results(self, decoy_avg, decoy_sd):
        self.decoy_avg = decoy_avg
        self.decoy_sd = decoy_sd
        true_size = self.true_size

        if self.rank == 0:
            np.savetxt("%s/native_pairwise.dat" % self.savedir, self.remap_pairE(self.native_pair_E, true_size))
            np.savetxt("%s/native_single_residue.dat" % self.savedir, self.remap_singleE(self.single_residue_avg_E, true_size))

            if decoy_avg.ndim == 2:
                np.savetxt("%s/decoy_avg.dat" % self.savedir, self.remap_pairE(decoy_avg, true_size))
                np.savetxt("%s/decoy_sd.dat" % self.savedir, self.remap_pairE(decoy_sd, true_size))

            if decoy_avg.ndim == 1:
                np.savetxt("%s/decoy_avg.dat" % self.savedir, self.remap_singleE(decoy_avg, true_size))
                np.savetxt("%s/decoy_sd.dat" % self.savedir, self.remap_singleE(decoy_sd, true_size))

    def save_decoy_results(self, decoy_list):
        self.decoy_list = decoy_list
        if self.rank == 0:
            np.save("%s/decoy_E_list" % (self.savedir), decoy_list)

    def save_specific_pairs(self, decoy_list_array, save_pairs):
        self.decoy_list_array = decoy_list_array
        if self.rank == 0:
            for i in range(np.shape(save_pairs)[0]):
                pidx = save_pairs[i][0]
                pjdx = save_pairs[i][1]
                this_E_list = decoy_list_array[pidx][pjdx]
                np.save("%s/decoy_E_list_%d-%d" % (self.savedir, pidx, pjdx), this_E_list)

    def analyze_all_pairs(self, decoy_list_array, spacing=0.2):
        if self.rank == 0:
            chi_array = np.zeros((len(decoy_list_array),len(decoy_list_array)))
            count_array = np.zeros((len(decoy_list_array),len(decoy_list_array)))
            for idx in range(len(decoy_list_array)):
                for jdx in range(len(decoy_list_array)):
                    decoyE = decoy_list_array[idx][jdx]
                    count_array[idx,jdx] = np.shape(decoyE)[0]
                    if np.shape(decoyE)[0] > self.min_use:
                        # default value is zero if not a lot of data exists
                        chi, avg, sd = compute_gaussian_and_chi(decoyE)
                        chi_array[idx,jdx] = chi
            np.savetxt("%s/decoy_gaussian_reduced_chi2.dat" % (self.savedir), chi_array)
            np.savetxt("%s/decoy_gaussian_counts.dat" % (self.savedir), count_array)

    def save_specific_single_residues(self, decoy_list_array, save_residues):
        self.decoy_list_array = decoy_list_array
        if self.rank == 0:
            for i in range(np.shape(save_residues)[0]):
                pidx = save_residues[i]
                this_E_list = decoy_list_array[pidx]
                np.save("%s/decoy_E_list_%d" % (self.savedir, pidx), this_E_list)

    def analyze_all_single_residues(self, decoy_list_array, spacing=0.2):
        if self.rank == 0:
            chi_array = np.zeros(len(decoy_list_array))
            count_array = np.zeros(len(decoy_list_array))
            for idx in range(len(decoy_list_array)):
                decoyE = decoy_list_array[idx]
                count_array[idx] = np.shape(decoyE)[0]
                chi, avg, sd = compute_gaussian_and_chi(decoyE)
                chi_array[idx] = chi
            np.savetxt("%s/decoy_gaussian_reduced_chi2.dat" % (self.savedir), chi_array)
            np.savetxt("%s/decoy_gaussian_counts.dat" % (self.savedir), count_array)

class BookKeeperSingleResidue(BookKeeper):
    pass

    def initialize_average_results(self):
        super(BookKeeperSingleResidue, self).initialize_average_results()
        self.single_residue_energy = np.sum(self.native_pair_E, axis=0)

    def remap_pairE(self, data, size):
        # re-map the final matrix to new dimensions
        new_pair_E = np.zeros((size))
        for i in range(np.shape(data)[0]):
            idx = self.native_fpose_current_indices[i]
            value = data[i]

            new_pair_E[idx] = value

        return new_pair_E

    def save_results(self, decoy_avg, decoy_sd):
        self.decoy_avg = decoy_avg
        self.decoy_sd = decoy_sd
        true_size = self.true_size

        if self.rank == 0:
            np.savetxt("%s/native_singleresidue.dat" % self.savedir, self.remap_pairE(self.single_residue_energy, true_size))
            np.savetxt("%s/decoy_avg.dat" % self.savedir, self.remap_pairE(decoy_avg, true_size))
            np.savetxt("%s/decoy_sd.dat" % self.savedir, self.remap_pairE(decoy_sd, true_size))

def redo_compute_mutational_pairwise_mpi(book_keeper, ndecoys=1000, pack_radius=10., mutation_scheme="simple", use_contacts=None, contacts_scores=None, remove_high=None, compute_all_neighbors=False, save_pairs=None, save_residues=None, use_compute_single_residue=False):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    if use_compute_single_residue:
        print "Using Single Residue Mode"
        mutational_procedure_name = "Mutational Single Residue"
        Constructor = ConstructMutationalSingleMPI
        Runner = ComputeSingleMPI
    else:
        mutational_procedure_name = "Mutational Frustration"
        Constructor = ConstructMutationalMPI
        Runner = ComputePairMPI
    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights
    nresidues = book_keeper.final_nresidues

    analysis_object = Constructor(nresidues, use_contacts=use_contacts, contacts_scores=contacts_scores)

    analyze_pairs = analysis_object.inputs_collected
    n_analyze = len(analyze_pairs)
    new_computer = Runner(rank, analyze_pairs, book_keeper, scorefxn, order, weights, ndecoys, nresidues, pack_radius=pack_radius, mutation_scheme=mutation_scheme, remove_high=remove_high, compute_all_neighbors=compute_all_neighbors)

    job_indices = get_mpi_jobs(n_analyze, rank, size)
    new_computer.run(job_indices)

    new_computer.print_status()

    # wait until all jobs finish
    comm.Barrier()

    # send block
    if rank == 0:
        print "Finished All Calculations"
        analysis_object.process_results_q(new_computer.save_q)
        for i in range(1, size):
            results = comm.recv(source=i, tag=3)
            analysis_object.process_results_q(results)
        print "Finished Saving all results"
    else:
        comm.send(new_computer.save_q, dest=0, tag=3)

    comm.Barrier()

    if rank == 0:
        # get_saved results
        if use_compute_single_residue:
            book_keeper.analyze_all_single_residues(analysis_object.E_list)
            if save_residues is not None:
                book_keeper.save_specific_single_residues(analysis_object.E_list, save_residues)
        else:
            all_e_list = analysis_object.all_e_list
            book_keeper.analyze_all_pairs(all_e_list)

            if save_pairs is not None:
                for pair in save_pairs:
                    np.savetxt("%s/decoy_E_list_%d-%d.npy" % (book_keeper.savedir, pair[0], pair[1]), all_e_list[pair[0]-1][pair[1]-1])

        print "THE END"

def compute_mutational_pairwise_mpi(book_keeper, ndecoys=1000, pack_radius=10., mutation_scheme="simple", use_contacts=None, contacts_scores=None, remove_high=None, compute_all_neighbors=False, save_pairs=None, save_residues=None, use_compute_single_residue=False):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    if use_compute_single_residue:
        print "Using Single Residue Mode"
        mutational_procedure_name = "Mutational Single Residue"
        Constructor = ConstructMutationalSingleMPI
        Runner = ComputeSingleMPI
    else:
        mutational_procedure_name = "Mutational Frustration"
        Constructor = ConstructMutationalMPI
        Runner = ComputePairMPI
    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights
    nresidues = book_keeper.final_nresidues

    analysis_object = Constructor(nresidues, use_contacts=use_contacts, contacts_scores=contacts_scores)

    analyze_pairs = analysis_object.inputs_collected
    n_analyze = len(analyze_pairs)
    new_computer = Runner(rank, analyze_pairs, book_keeper, scorefxn, order, weights, ndecoys, nresidues, pack_radius=pack_radius, mutation_scheme=mutation_scheme, remove_high=remove_high, compute_all_neighbors=compute_all_neighbors)

    job_indices = get_mpi_jobs(n_analyze, rank, size)
    new_computer.run(job_indices)

    new_computer.print_status()

    # wait until all jobs finish
    comm.Barrier()

    # send block
    if rank == 0:
        print "Finished All Calculations"
        analysis_object.process_results_q(new_computer.save_q)
        for i in range(1, size):
            results = comm.recv(source=i, tag=3)
            analysis_object.process_results_q(results)
        print "Finished Saving all results"
    else:
        comm.send(new_computer.save_q, dest=0, tag=3)

    comm.Barrier()

    E_avg, E_std = analysis_object.get_saved_results()
    book_keeper.save_results(E_avg, E_std)

    if rank == 0:
        # get_saved results
        if use_compute_single_residue:
            book_keeper.analyze_all_single_residues(analysis_object.E_list)
            if save_residues is not None:
                book_keeper.save_specific_single_residues(analysis_object.E_list, save_residues)
        else:
            all_e_list = analysis_object.all_e_list
            book_keeper.analyze_all_pairs(all_e_list)

            if save_pairs is not None:
                for pair in save_pairs:
                    np.savetxt("%s/decoy_E_list_%d-%d.npy" % (book_keeper.savedir, pair[0], pair[1]), all_e_list[pair[0]-1][pair[1]-1])

        mut_info = "Mutational Frustration Computed with: %s \n" % (mutational_procedure_name)
        mut_info += "ndecoys = %d \n" % ndecoys
        mut_info += "pack_radius = %g \n" % pack_radius
        mut_info += "scheme = %s \n" % mutation_scheme
        mut_info += "use_contacts = %s \n" % str(use_contacts)
        mut_info += "contacts_scores = %s \n" % str(contacts_scores)
        mut_info += "remove_high = %s \n" % (str(remove_high))
        mut_info += "compute_all_neighbors = %s \n" % (str(compute_all_neighbors))

        book_keeper.save_basic_info(info=mut_info)

        print "THE END"

def compute_configurational_pairwise_mpi(book_keeper, top_file, configurational_traj_file, configurational_dtraj=None, configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, pcutoff=0.8, native_contacts=None, use_contacts=None, contacts_scores=None, use_config_individual_pairs=False, min_use=10, save_pairs=None, save_residues=None, remove_high=None, count_all_similar=False, use_compute_single_residue=False):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights
    scratch_dir = book_keeper.scratch_dir
    native_fpose = book_keeper.all_native_fpose[0]
    nresidues = book_keeper.final_nresidues

    if rank == 0:
        use_verbose = True
    else:
        use_verbose = False

    if use_config_individual_pairs and use_compute_single_residue:
        print "Warning: Both individual paris and single residue mode is activated. Defaulting to individual pairs"

    if use_config_individual_pairs:
        if count_all_similar:
            this_procedure_number = 2
        else:
            this_procedure_number = 1
        this_procedure_name = "Heterogeneous %d" % (this_procedure_number)
        analysis_object = ConstructConfigIndividualMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, min_use=min_use, verbose=use_verbose, remove_high=remove_high, count_all_similar=count_all_similar)

        new_computer = ComputeConfigIndividualMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, native_fpose, pcutoff=pcutoff, rcutoff=analysis_object.decoy_r_cutoff)

    elif use_compute_single_residue:
        this_procedure_name = "Single Residue"
        analysis_object = ConstructConfigSingleResidueMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, min_use=min_use, verbose=use_verbose, remove_high=remove_high)

        new_computer = ComputeConfigSingleResidueMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, native_fpose, pcutoff=pcutoff, rcutoff=analysis_object.decoy_r_cutoff)

    else:
        this_procedure_name = "Homogeneous"
        analysis_object = ConstructConfigurationalMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, verbose=use_verbose, remove_high=remove_high)

        new_computer = ComputeConfigMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, native_fpose, pcutoff=pcutoff, rcutoff=analysis_object.decoy_r_cutoff)

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
        t1 = time.time()
        analysis_object.process_results_q(new_computer.save_q)
        print "Currently been %f minutes to process the data" % ((time.time() - t1) / 60.)
        for i in range(1, size):
            results = comm.recv(source=i, tag=3)
            print "Currently been %f minutes to transfer the data" % ((time.time() - t1) / 60.)
            analysis_object.process_results_q(results)
            print "Currently been %f minutes to process the data" % ((time.time() - t1) / 60.)
    else:
        comm.send(new_computer.save_q, dest=0, tag=3)

    # process results block

    E_avg, E_std = analysis_object.get_saved_results()

    comm.Barrier()
    book_keeper.save_results(E_avg, E_std)


    if rank == 0:
        print "finished saving basic results"
        if use_config_individual_pairs: # heterogeneous 1 or 2
            book_keeper.analyze_all_pairs(analysis_object.E_list)
            if save_pairs is not None:
                book_keeper.save_specific_pairs(analysis_object.E_list, save_pairs)
        elif use_compute_single_residue: # single residue
            book_keeper.analyze_all_single_residues(analysis_object.E_list)
            if save_residues is not None:
                book_keeper.save_specific_single_residues(analysis_object.E_list, save_residues)

        else: # homogeneous
            book_keeper.save_decoy_results(analysis_object.E_list)
            chi, avg, sd = compute_gaussian_and_chi(analysis_object.E_list)
            f = open("%s/chi2.dat" % (book_keeper.savedir), "w")
            f.write("%f   %f   %f\n" % (chi, avg, sd))
            f.close()

        config_info = "Configurational Frustration Computed with: %s \n" % (this_procedure_name)
        config_info += "top_file = %s \n" % top_file
        config_info += "traj_file = %s \n" % configurational_traj_file
        config_info += "dtraj_file = %s \n" % str(configurational_dtraj)
        config_info += "dtraj highcutoff = %s \n" % (configurational_parameters["highcutoff"])
        config_info += "dtraj lowcutoff = %f \n" % (configurational_parameters["lowcutoff"])
        config_info += "stride = %d \n" % (configurational_parameters["stride_length"])
        config_info += "decoy_r_cutoff = %f \n" % (configurational_parameters["decoy_r_cutoff"])
        config_info += "pcutoff = %g \n" % pcutoff
        config_info += "native_contacts = %s \n" % str(native_contacts)
        config_info += "use_contacts = %s \n" % str(use_contacts)
        config_info += "contacts_scores = %s \n" % str(contacts_scores)
        config_info += "min_use = %d \n" % (min_use)
        config_info += "remove_high = %s \n" % (remove_high)

        book_keeper.save_basic_info(info=config_info)

        print "finished saving individual pairs and chi2 results"
