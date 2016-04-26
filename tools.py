"""
Created on Apr 25, 2016

@author: victor
"""
import numpy

def get_report_from_trajectory(traject_path, report_base):
    """
    The format is usually <name>_<number>.pdb
    """
    name_number = traject_path.split(".")
    _, number = name_number[0].split("_")
    return report_base+"_"+number
    
def get_report_steps(report_file):
    steps = []
    accepted = []
    for line in  open(report_file):
        if line[0] != "#":
            task, step, acc_step, energy = line.split()
            steps.append(int(step))
            accepted.append(int(acc_step))
    return steps, accepted

def add_rejected_to_coordinates(steps, one_traj_coordsets):
    """
     1 3 4 6 <0>
    <0>1 3 4 6
    ------------
     1 2 1 2 -6
      [2 1 2 1]
    """
    
    padded = steps + [0]
    shifted = [0]+steps
    repes = (numpy.array(padded)-numpy.array(shifted))[1:]
    repes[-1] = 1
    
    repe_coordinates = []
    for i, frame_coord in enumerate(one_traj_coordsets):
        for _ in range(repes[i]):
            repe_coordinates.append(one_traj_coordsets[i])
    
    return numpy.array(repe_coordinates)

def add_all_trajectories_rejected_from_report(all_traj_paths, all_traj_coordsets, report_base):
    """
    all_traj_paths and  all_traj_coordsets have the same ordered.
    """
    inflated_coordsets = []
    for i,traj_path in enumerate(all_traj_paths):
        my_report = get_report_from_trajectory(traj_path, report_base)
        steps, _ = get_report_steps(my_report)
        inflated_coordsets.append(add_rejected_to_coordinates(steps,all_traj_coordsets[i]))
    return inflated_coordsets
        
        