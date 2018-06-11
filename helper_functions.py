from .util import *
from .mutational import ConstructMutationalMPI, ComputePairMPI
from .configurational import ConstructConfigurationalMPI, ComputeConfigMPI, ConstructConfigIndividualMPI, ComputeConfigIndividualMPI

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
    def __init__(self, native_file, nresidues, savedir=None, use_hbonds=False, relax_native=False, rcutoff=0.6, pcutoff=0.8, mutate_traj=None, repack_radius=10):
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
        self.repack_radius = repack_radius

        self.decoy_avg = None
        self.decoy_sd = None
        self.decoy_list = None
        self.decoy_list_array = None

        self.initialize_rosetta()
        self.prepare_simulations()

    def prepare_simulations(self):
        native_file = self.native_file
        savedir = self.savedir
        use_hbonds = self.use_hbonds
        relax_native = self.relax_native
        rcutoff = self.rcutoff
        pcutoff = self.pcutoff
        nresidues = self.nresidues

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

        native_pose = pyr.pose_from_pdb(native_file)

        if self.mutate_traj is None:
            close_contacts, close_contacts_zero, contacts_scores = determine_close_residues_from_file(native_file, probability_cutoff=pcutoff, radius_cutoff=rcutoff)
        else:
            dump_file = "%s/mutated_pdb_0.pdb" % (scratch_dir)
            if rank == 0:
                for mutation in self.mutate_traj:
                    mut_idx = mutation[0]
                    new_res = mutation[1]

                    mutate_residue(native_pose, mut_idx, new_res, pack_radius=self.repack_radius)

                native_pose.dump_pdb(dump_file)
            self.comm.Barrier()
            native_pose = pyr.pose_from_pdb(dump_file)

            close_contacts, close_contacts_zero, contacts_scores = determine_close_residues_from_file(dump_file, probability_cutoff=pcutoff, radius_cutoff=rcutoff)

        if relax_native:
            relaxer = ClassicRelax()
            relaxer.set_scorefxn(pyrt.get_fa_scorefxn())
            relaxer.apply(native_pose)

        native_pair_E = compute_pairwise(native_pose, scorefxn_custom, order, weights, use_contacts=close_contacts, nresidues=nresidues)

        self.native_pose = native_pose
        self.scorefxn_custom = scorefxn_custom
        self.order = order
        self.weights = weights
        self.close_contacts = close_contacts
        self.scratch_dir = scratch_dir
        self.native_pair_E = native_pair_E

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

    def save_results(self, decoy_avg, decoy_sd):
        self.decoy_avg = decoy_avg
        self.decoy_sd = decoy_sd

        if self.rank == 0:
            np.savetxt("%s/native_pairwise.dat" % self.savedir, self.native_pair_E)
            np.savetxt("%s/decoy_avg.dat" % self.savedir, decoy_avg)
            np.savetxt("%s/decoy_sd.dat" % self.savedir, decoy_sd)

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
            for idx in range(len(decoy_list_array)):
                for jdx in range(len(decoy_list_array)):
                    decoyE = decoy_list_array[idx][jdx]
                    chi, avg, sd = compute_gaussian_and_chi(decoyE)
                    chi_array[idx,jdx] = chi
            np.savetxt("%s/decoy_gaussian_reduced_chi2.dat" % (self.savedir), chi_array)

def compute_mutational_pairwise_mpi(book_keeper, ndecoys=1000, pack_radius=10., mutation_scheme="simple", use_contacts=None, contacts_scores=None, remove_high=None, compute_all_neighbors=False, save_pairs=None):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    pose = book_keeper.native_pose
    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights
    nresidues = book_keeper.nresidues

    analysis_object = ConstructMutationalMPI(nresidues, use_contacts=use_contacts, contacts_scores=contacts_scores)

    analyze_pairs = analysis_object.inputs_collected
    n_analyze = len(analyze_pairs)
    new_computer = ComputePairMPI(rank, analyze_pairs, pose, scorefxn, order, weights, ndecoys, nresidues, pack_radius=pack_radius, mutation_scheme=mutation_scheme, remove_high=remove_high, compute_all_neighbors=compute_all_neighbors)

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
    else:
        comm.send(new_computer.save_q, dest=0, tag=3)

    comm.Barrier()
    # get_saved results
    if rank == 0:
        E_avg, E_std = analysis_object.get_saved_results()
        book_keeper.save_results(E_avg, E_std)

        all_e_list = analysis_object.all_e_list
        book_keeper.analyze_all_pairs(all_e_list)

        if save_pairs is not None:
            for pair in save_pairs:
                np.savetxt("%s/decoy_E_list_%d-%d.npy" % (book_keeper.savedir, pair[0], pair[1]))


def compute_configurational_pairwise_mpi(book_keeper, top_file, configurational_traj_file, configurational_dtraj=None, configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, pcutoff=0.8, native_contacts=None, use_contacts=None, contacts_scores=None, use_config_individual_pairs=False, min_use=10, save_pairs=None, mutate_traj_file=None, repack_radius=10):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights
    scratch_dir = book_keeper.scratch_dir
    nresidues = book_keeper.nresidues

    if mutate_traj_file is None:
        mutate_traj = None
    else:
        mutate_traj = parse_mutations_file(mutate_traj_file)

    if rank == 0:
        use_verbose = True
    else:
        use_verbose = False
    if use_config_individual_pairs:
        analysis_object = ConstructConfigIndividualMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, min_use=min_use, verbose=use_verbose)

        new_computer = ComputeConfigIndividualMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, pcutoff=0.8, rcutoff=analysis_object.decoy_r_cutoff, mutate_traj=mutate_traj, repack_radius=repack_radius)
    else:
        analysis_object = ConstructConfigurationalMPI(nresidues, top_file, configurational_traj_file, configurational_dtraj=configurational_dtraj, configurational_parameters=configurational_parameters, native_contacts=native_contacts, verbose=use_verbose)

        new_computer = ComputeConfigMPI(rank, nresidues, configurational_traj_file, top_file, scorefxn, order, weights, scratch_dir, pcutoff=0.8, rcutoff=analysis_object.decoy_r_cutoff, mutate_traj=mutate_traj, repack_radius=repack_radius)

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
    book_keeper.save_results(E_avg, E_std)


    if rank == 0:
        if not use_config_individual_pairs:
            book_keeper.save_decoy_results(analysis_object.E_list)
            chi, avg, sd = compute_gaussian_and_chi(analysis_object.E_list)
            f = open("%s/chi2.dat" % (book_keeper.savedir), "w")
            f.write("%f   %f   %f\n" % (chi, avg, sd))
            f.close()
        else:
            book_keeper.analyze_all_pairs(analysis_object.E_list)
            if save_pairs is not None:
                book_keeper.save_specific_pairs(analysis_object.E_list, save_pairs)
