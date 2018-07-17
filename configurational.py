from .util import *

class ConstructConfigurationalMPI(object):
    def __init__(self, nresidues, top_file, configurational_traj_file, configurational_dtraj=None, configurational_parameters={"highcutoff":0.9, "lowcutoff":0., "stride_length":10, "decoy_r_cutoff":0.5}, verbose=False, native_contacts=None, remove_high=False):
        self.verbose = verbose
        self.nresidues = nresidues

        self.top_file = top_file
        self.configurational_traj_file = configurational_traj_file
        self.configurational_dtraj = configurational_dtraj
        self.configurational_parameters = configurational_parameters
        self.native_contacts = native_contacts
        self.remove_high = remove_high
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
            if self.remove_high is None:
                count += 1
                self.E_list.append(results)
            else:
                if results < self.remove_high:
                    count += 1
                    self.E_list.append(results)

        if count == 0:
            print "results_q had no results to save... "

        E_avg, E_sd = compute_average_and_sd(self.E_list)

        self.assign_E_results(E_avg, E_sd)

        if self.verbose:
            print "Completed %d saves" % count

    def get_saved_results(self):
        return self.E_avg, self.E_sd

class ComputeConfigMPI(object):
    def __init__(self, thread_number, nresidues, traj_file, top_file, scorefxn, order, weights, scratch_dir, native_fpose, pcutoff=0.8, rcutoff=0.6):
        self.thread_number = thread_number
        print "Thread %d Starting" % self.thread_number

        self.nresidues = nresidues
        self.traj_file = traj_file
        self.top_file = top_file

        self.scorefxn = scorefxn
        self.order = order
        self.weights = weights
        self.pcutoff = pcutoff
        self.rcutoff = rcutoff
        self.scratch_dir = scratch_dir
        self.native_fpose = native_fpose

        self.still_going = True # default action is to keep going
        self.start_time = time.time()
        self.n_jobs_run = 0
        self._initialize_saveq()

        self.mutate_traj = (native_fpose.deletion_ranges is not None) or (native_fpose.mutation_list is not None)
        print self.mutate_traj

        if self.mutate_traj:
            self.clean_and_return_pose = self._clean_and_return_mutate
        else:
            self.clean_and_return_pose = self._clean_and_return_simple

        random.seed(int(time.time()) + int(self.thread_number*1000))

        print self.native_fpose.pose.sequence()
        print self.native_fpose.deletion_ranges
        print self.native_fpose.mutation_list

    def _initialize_saveq(self):
        self.save_q = np.empty((0,))

    def print_status(self):
        print "THREAD%2d --- %6f minutes: %6d Frames Complete" % (self.thread_number, (time.time() - self.start_time)/60., self.n_jobs_run)

    def _clean_and_return_mutate(self, index):
        save_pdb_file_initial = "thread_%d.pdb" % (self.thread_number)
        save_pdb_file_final = "thread_%d_final.pdb" % (self.thread_number)
        rosetta_pdb_file = "thread_%d.clean.pdb" % (self.thread_number)

        traj_initial = md.load_frame(self.traj_file, index, top=self.top_file)
        traj_initial.save(save_pdb_file_initial)
        #print "here"
        cleanATOM(save_pdb_file_initial)

        pose = pyr.pose_from_pdb(rosetta_pdb_file)
        new_fpose = self.native_fpose.duplicate_changes_new_pose(pose)
        new_fpose.dump_to_pdb(save_pdb_file_final)
        traj = md.load(save_pdb_file_final)
        return traj, pose

    def _clean_and_return_simple(self, index):
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
        #enable_print()
        #print this_pose.sequence()
        #block_print()
        # get residue (1-indexed) contacts
        close_contacts, close_contacts_zero, contacts_scores = determine_close_residues(this_traj, probability_cutoff=self.pcutoff, radius_cutoff=self.rcutoff)

        this_pair_E = compute_pairwise(this_pose, self.scorefxn, self.order, self.weights, use_contacts=close_contacts, nresidues=self.nresidues)

        for contact_index in close_contacts_zero:
            idx = contact_index[0] #use the 0-indexed
            jdx = contact_index[1] #use the 0-indexed
            #self.save_q.append(this_pair_E[idx, jdx] )
            self.save_q = np.append(self.save_q, this_pair_E[idx, jdx])

        self.still_going = False
        enable_print()

        self.n_jobs_run += 1

        return

class ConstructConfigIndividualMPI(ConstructConfigurationalMPI):
    def __init__(self, *args, **kwargs):
        specific_args = ["min_use", "count_all_similar"]
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

        if "count_all_similar" in kwargs:
            self.count_all_similar = kwargs["count_all_similar"]
        else:
            self.count_all_similar = False

        self.was_updated = False

    @property
    def E_avg(self):
        if self.was_updated:
            self._save_compute_results()
        return self._E_avg

    @property
    def E_sd(self):
        if self.was_updated:
            self._save_compute_results()

        return self._E_sd

    def _initialize_empty_results(self):
        self.E_list = [[np.empty(0) for i in range(self.nresidues)] for j in range(self.nresidues)]
        self._E_avg = np.zeros((self.nresidues, self.nresidues))
        self._E_sd = np.zeros((self.nresidues, self.nresidues))

    def assign_E_results(self, idx, jdx, avg, std):
        self._E_avg[idx, jdx] = avg
        self._E_sd[idx, jdx] = std

    def append_all_similar(self, idx, jdx, E):
        total_counts = 0
        for i in range(self.nresidues):
            if i != idx and i != jdx:
                self.E_list[i][jdx] = np.append(self.E_list[i][jdx], E)
                self.E_list[jdx][i] = np.append(self.E_list[jdx][i], E)
                self.E_list[i][idx] = np.append(self.E_list[i][idx], E)
                self.E_list[idx][i] = np.append(self.E_list[idx][i], E)
                total_counts += 4

        try:
            assert total_counts == ((self.nresidues * 4) - 8)
        except:
            print "Counted %d pairs, but expected %d" % (total_counts, ((self.nresidues*4)-8))
            raise

    def process_results_q(self, results_q):
        # take a queue as input, and then analyze the results
        # for configurational, anticipate a list of pair energies
        self.was_updated = True
        count = 0
        for i_parse in range(self.nresidues):
            for j_parse in range(i_parse+1, self.nresidues):
                this_array = self.E_list[i_parse][j_parse]
                if np.shape(this_array)[0] > 0:
                    if self.remove_high is None:
                        this_new = results_q[i_parse][j_parse]
                    else:
                        this_new_nocutoff = results_q[i_parse][j_parse]
                        this_new_idxs = np.where(this_new_nocutoff < self.remove_high)
                        this_new = this_new_nocutoff[this_new_idxs]
                    self.E_list[i_parse][j_parse] = np.append(this_array, this_new)
                    self.E_list[j_parse][i_parse] = np.append(this_array, this_new)
                    if self.count_all_similar:
                        self.append_all_similar(i_parse, j_parse, this_new)
                    count += np.shape(this_new)[0]

        print "%d pairs were saved" % (count)

    def _save_compute_results(self):
        zero_count = 0
        found_count = 0
        min_count = 0
        for idx in range(self.nresidues):
            for jdx in range(self.nresidues):
                found_count += 1
                this_list = self.E_list[idx][jdx]
                if np.shape(this_list)[0] == 0:
                    E_avg = 0
                    E_sd = 0
                    zero_count += 1
                elif np.shape(this_list)[0] < self.min_use:
                    E_avg = 0
                    E_sd = 0
                    min_count += 1
                else:
                    E_avg, E_sd = compute_average_and_sd(self.E_list[idx][jdx])
                self.assign_E_results(idx, jdx, E_avg, E_sd)

        self.was_updated = False
        if self.verbose:
            print "%f of the pairs had zero count while %f of the pairs had non-zero counts but were below the minimum threshold of %d" % (float(zero_count)/float(found_count), float(min_count)/float(found_count), self.min_use)

class ComputeConfigIndividualMPI(ComputeConfigMPI):

    def _initialize_saveq(self):
        self.save_q = [[np.empty(0) for i in range(self.nresidues)] for j in range(self.nresidues)]

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
        close_contacts, close_contacts_zero, contacts_scores = determine_close_residues(this_traj, probability_cutoff=self.pcutoff, radius_cutoff=self.rcutoff)

        this_pair_E = compute_pairwise(this_pose, self.scorefxn, self.order, self.weights, use_contacts=close_contacts, nresidues=self.nresidues)

        for contact_index in close_contacts_zero:
            i_first = contact_index[0] #use the 0-indexed
            i_second = contact_index[1] #use the 0-indexed

            if i_first < i_second: #bigger index last
                idx =  i_first
                jdx = i_second
            else:
                idx = i_second
                jdx = i_first

            #save_dict = {"idx":idx, "jdx":jdx, "E": this_pair_E[idx,jdx]}
            #self.save_q.append(save_dict)
            self.save_q[idx][jdx] = np.append(self.save_q[idx][jdx], this_pair_E[idx,jdx])

        self.still_going = False
        enable_print()

        self.n_jobs_run += 1

        return
