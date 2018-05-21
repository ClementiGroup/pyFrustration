from util import *
from mutational import ConstructMutationalMPI, ComputePairMPI
from configurational import ConstructConfigurationalMPI, ComputeConfigMPI, ConstructConfigIndividualMPI, ComputeConfigIndividualMPI

class BookKeeper(object):
    def __init__(native_file, savedir=None, use_hbonds=False):
        self.native_file = native_file
        if savedir is None:
            savedir = os.getcwd()

        self.savedir = savedir
        self.use_hbonds = use_hbonds

        self.decoy_avg = None
        self.decoy_sd = None

        self.decoy_list = None
        self.decoy_list_array = None

        self.prepare_simulations()

    def prepare_simulations(self):
        native_file = self.native_file
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

        native_pose = pyr.pose_from_pdb(native_file)

        close_contacts, close_contacts_zero, contacts_scores = determine_close_residues_from_file(native_file, probability_cutoff=pcutoff, radius_cutoff=rcutoff)

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
                np.save("%s/decoy_E_list_%d-%d.dat" % (self.savedir, pidx, pjdx), this_E_list)

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

def compute_mutational_pairwise_mpi(book_keeper, ndecoys=1000, nresidues=35, pack_radius=10., mutation_scheme="simple", use_contacts=None, contacts_scores=None, remove_high=None):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    pose = book_keeper.native_pose
    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights

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

    book_keeper.save_results(decoy_avg, decoy_sd)

def compute_configurational_pairwise_mpi(book_keeper, top_file, configurational_traj_file, scratch_dir, configurational_dtraj=None, configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, nresidues=35, pcutoff=0.8, native_contacts=None, use_contacts=None, contacts_scores=None, use_config_individual_pairs=False, min_use=10, save_pairs=None):
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    scorefxn = book_keeper.scorefxn_custom
    order = book_keeper.order
    weights = book_keeper.weights

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
    book_keeper.save_results(decoy_avg, decoy_sd)

    if not use_config_individual_pairs:
        book_keeper.save_decoy_results(analysis_object.E_list)
    else:
        if save_pairs is not None:
            book_keeper.save_specific_pairs(analysis_object.E_list, save_pairs)
