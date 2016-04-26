"""
Created on Apr 25, 2016

@author: victor
"""
import unittest
import test.data as data
from tools import get_report_steps, add_rejected_to_coordinates,\
    get_report_from_trajectory
from os.path import os
import numpy

class TestAutocorrelationStuff(unittest.TestCase):

    def test_read_report(self):
        report_file = os.path.join(data.__path__[0], "report_7")
        expected_steps = [0, 1, 2, 3, 4, 5, 7, 8, 9, 14, 15, 16, 17] 
        expected_accepted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        steps, accepted = get_report_steps(report_file)
        self.assertSequenceEqual(expected_accepted, accepted)
        self.assertSequenceEqual(expected_steps, steps)
        
    def test_apply_(self):
        accepted_steps = [0, 1, 2, 3, 4, 5, 7, 8, 9, 14, 15, 16, 17] 
        expected_full_coords = [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 8, 8, 9, 10, 11, 12]
        full_coords = add_rejected_to_coordinates(accepted_steps, range(len(accepted_steps)))
        numpy.testing.assert_array_equal(expected_full_coords, full_coords)
        
    def test_get_report_from_trajectory(self):
        self.assertEqual("report_4",  get_report_from_trajectory("traj_4.pdb"))
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_read_report']
    unittest.main()