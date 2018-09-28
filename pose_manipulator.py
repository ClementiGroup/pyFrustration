""" Define methods and classes for making and manipulating poses """
from .util import *
from pyrosetta.rosetta.protocols.grafting import delete_region

class FrusPose(object):
    def __init__(self, pose, repack_radius=10):
        self.pose = pose
        self.old_nresidues = len(self.pose.sequence())
        self.repack_radius = repack_radius
        self.original_indices = np.arange(self.old_nresidues) # 0-index, keep track of deletions

        self.current_indices = np.arange(self.old_nresidues)

        self.mutation_list = None
        self.deletion_ranges = None

    @property
    def sequence(self):
        return self.pose.sequence()

    @property
    def nresidues(self):
        return len(self.sequence)

    def modify_protein(self, mutation_list=None, deletion_ranges=None):
        print "Warning: FrusPose.modify_protein() not configured to keep track of original indices for mutations"
        if mutation_list is not None:
            for mutation in mutation_list:
                mut_idx = mutation[0]
                new_res = mutation[1]

                self._mutate_residue(mut_idx, new_res, self.repack_radius)

        if deletion_ranges is not None:
            for deletion in deletion_ranges:
                self._delete_region(deletion)

    def add_change_history(self, mutation_list=None, deletion_ranges=None):
        if mutation_list is not None:
            for mutation in mutation_list:
                mut_idx = mutation[0]
                new_res = mutation[1]
                if self.mutation_list is None:
                    self.mutation_list = [[mut_idx, new_res]]
                else:
                    self.mutation_list.append([mut_idx, new_res])

        if deletion_ranges is not None:
            for deletion in deletion_ranges:
                start = deletion[0]
                end = deletion[1]
                if self.deletion_ranges is None:
                    self.deletion_ranges = [[start, end]]
                else:
                    self.deletion_ranges.append([start, end])

    def _mutate_residue(self, mut_idx, new_res, repack_radius):
        mutate_residue(self.pose, mut_idx, new_res, pack_radius=repack_radius)
        if self.mutation_list is None:
            self.mutation_list = [[mut_idx, new_res]]
        else:
            self.mutation_list.append([mut_idx, new_res])

    def _delete_region(self, deletion_range):
        # expect deletion_range is a 1-indexed list for an inclusive range
        start = deletion_range[0]
        end = deletion_range[1]

        true_start = np.where(self.current_indices == start-1)[0][0]
        true_end = np.where(self.current_indices == end-1)[0][0]

        new_indices = np.append(self.current_indices[:true_start], self.current_indices[true_end+1:])

        delete_region(self.pose, true_start+1, true_end+1)
        self.current_indices = new_indices

        if self.deletion_ranges is None:
            self.deletion_ranges = [[start, end]]
        else:
            self.deletion_ranges.append([start, end])

    def _deep_copy(self, new_fpose):
        new_fpose.old_nresidues = self.old_nresidues
        new_fpose.original_indices = self.original_indices
        new_fpose.current_indices = self.current_indices

    def dump_to_pdb(self, file_name):
        self.pose.dump_pdb(file_name)

    def duplicate_changes_new_pose(self, new_pose):
        new_fpose = FrusPose(new_pose, repack_radius=self.repack_radius)
        new_fpose.modify_protein(self.mutation_list, self.deletion_ranges)

        # reset old results:
        self._deep_copy(new_fpose)

        return new_fpose

    def duplicate_nochange_new_pose(self, new_pose):
        new_fpose = FrusPose(new_pose, repack_radius=self.repack_radius)
        new_fpose.deletion_ranges = self.deletion_ranges
        new_fpose.mutation_list = self.mutation_list

        #reset old results
        self._deep_copy(new_fpose)

        return new_fpose
