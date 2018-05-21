from util import *

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
