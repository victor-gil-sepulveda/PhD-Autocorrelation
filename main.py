import numpy
from pyRMSD.utils.proteinReading import Reader
from pyRMSD.RMSDCalculator import RMSDCalculator
import argparse
import matplotlib.pylab as plt
import prody
from tools import add_all_trajectories_rejected_from_report

# Possible errors were: 
# - When skipping first frame is no longer the same for all trajectories
# - It makes sense to use iterposition (superimposition with mean structure), as we subtract the mean 
# - Each trajectory and coordinate must be analysed separately


def get_coords_and_superimpose_with_prody(trajectories, skip, max_frames, iterpose = True):
    all_coordsets = []
    for traj_path in trajectories:
        trajectory = prody.parsePDB(traj_path, subset='calpha')
        coordinates = trajectory.getCoordsets()

        ensembleTrajectory = prody.PDBEnsemble("Complex")
        ensembleTrajectory.setAtoms(trajectory)
        ensembleTrajectory.addCoordset(coordinates[skip : min(trajectory.numCoordsets(),max_frames+skip)]) 
        ensembleTrajectory.setCoords(coordinates[0]) #reference
        if iterpose:
            print "\t- Using iterposition on trajectory"
            ensembleTrajectory.iterpose()
        else:
            ensembleTrajectory.superpose()
        all_coordsets.append(ensembleTrajectory.getCoordsets())
    return all_coordsets

def get_coordinates(trajectories, skip, max_frames):
    all_coordsets = []
    for traj_path in trajectories:
        coords = Reader().readThisFile(traj_path).gettingOnlyCAs().read()
        all_coordsets.append(coords[skip : min(len(coords),max_frames+skip)])
    return all_coordsets

def superimpose_coordinates(all_coordsets, iterpose = True):
    all_superimposed_coordsets = []
    for coordsets in all_coordsets:
        calculator = RMSDCalculator(calculatorType = "QTRFIT_OMP_CALCULATOR",
                                fittingCoordsets = coordsets)
        calculator.setNumberOfOpenMPThreads(4)
        
        if iterpose:
            print "\t- Using iterposition on trajectory (shape ", coordsets.shape, ")"
            calculator.iterativeSuperposition()
            all_superimposed_coordsets.append(coordsets)
        else:
            print "\t- Superimposing with first trajectory frame (shape ", coordsets.shape, ")"
            _, superimposed_coordsets = calculator.oneVsTheOthers(0, get_superposed_coordinates = True)
            all_superimposed_coordsets.append(superimposed_coordsets)
    return all_superimposed_coordsets

def get_num_frames_per_trajectory(all_coordsets):
    num_frames = []
    for coordsets in all_coordsets:
        num_frames.append(len(coordsets))
    return num_frames

def standardAutocorrelationMethod(all_coordsets, AVG):
    #<r> := avgR
    print 'Computing mean positions ...'
    averages, std_devs = computeMeanCoordinatesForEachTrajectory(all_coordsets)
    
    print 'Computing autocorrelation for each coordinate ...'
    per_trajectory_per_coordinate_autocorrelations = computeEachCoordinateAutocorrelation(all_coordsets, averages, std_devs)

    return per_trajectory_per_coordinate_autocorrelations

def computeMeanCoordinatesForEachTrajectory(all_coordsets):
    """
    Computes per-trajectory averages and std dev.
    """
    averages = [] 
    std_devs = []
    for coordsets in all_coordsets:
        averages.append(coordsets.mean(0))
        std_devs.append(coordsets.std(0))
        
    return numpy.array(averages), numpy.array(std_devs)

def computeEachCoordinateAutocorrelation(all_coordsets, averages, std_devs):
    """
    Computes the autocorrelation with the formula:
        C(k) = 1/[(n-k)] \sum_{t=1}^{n-k} (Xt - mu)(Xt+k - mu)
    To normalise it wihitn [-1:1]
        c(k) = C(k) / var
    When the true mean \mu and variance \sigma^2 are known, this estimate is unbiased.
    """
    per_trajectory_per_coordinate_autocorrelations = []

    # Per trajectory
    for i, trajectory_coordinates in enumerate(all_coordsets):
        per_coordinate_autocorrelation = []
        
#         # Get a coordinate in each row.
#         trajectory_coordinates = trajectory_coordinates.T.copy()
#         
#         # Subtract the mean to each row
#         trajectory_coordinates = (trajectory_coordinates.T - averages[i]).T.copy()
        
        # The same in one line
        trajectory_coordinates = (trajectory_coordinates - averages[i]).T.copy()
        
        # All coordinates have the same dimension
        n = trajectory_coordinates.shape[1]
        
        # At each $\tau$ a different number of operations is done (n-k) in the formula above
        measures = numpy.arange(n, 0., -1.)
        
        # Get the autocorrelation for each coordinate
        for j in range(len(trajectory_coordinates)):
            r = numpy.correlate(trajectory_coordinates[j], trajectory_coordinates[j], mode = 'full')[-n:]
            r /= measures 
            per_coordinate_autocorrelation.append(r)
        per_coordinate_autocorrelation = numpy.array(per_coordinate_autocorrelation)
        
        # Divide per its coordinate variance
        per_coordinate_autocorrelation = (per_coordinate_autocorrelation.T / std_devs[i]).T
    per_trajectory_per_coordinate_autocorrelations.append(numpy.array(per_coordinate_autocorrelation))
    
    return numpy.array(per_trajectory_per_coordinate_autocorrelations)

def average_all_per_coordinate_autocorrelations(ptpc_autocorrelations):
    """
    First dimension is the trajectory level, second the per coordinate level, third contains the per 
    coordinate autocorrelations.
    We need to average them all. 
    """
    num_trajs, num_coordinates, _ = per_trajectory_per_coordinate_autocorrelations.shape
    all_autocorr =  []
    
    for num_traj in range(num_trajs):
        for num_coord in range(num_coordinates):
            all_autocorr.append(ptpc_autocorrelations[num_traj][num_coord])
    
    # Normalize and return (how can we normalize the std?)
    all_autocorr = numpy.array(all_autocorr)
    auto_mean = all_autocorr.mean(0)
    return auto_mean / auto_mean[0], all_autocorr.std(0)
    
def average_per_coordinate_autocorrelations(ptpc_autocorrelations):
    """
    Here we just calculate the avgs and not the std mean
    """
    num_trajs, num_coordinates, _ = per_trajectory_per_coordinate_autocorrelations.shape
    mean_per_coord_autocorrs =  []
    
    for num_coord in range(num_coordinates):
        this_coord_autocorrs = []
        for num_traj in range(num_trajs):
            this_coord_autocorrs.append(ptpc_autocorrelations[num_traj][num_coord])
        this_coord_autocorrs = numpy.array(this_coord_autocorrs)
        mean_per_coord_autocorrs.append(this_coord_autocorrs.mean(0))
    
    return numpy.array([coord_autocorr/coord_autocorr[0] for coord_autocorr in mean_per_coord_autocorrs])

if __name__ == "__main__":
    
    desc = "Calculates the autocorrelation for a set of trajectories and atoms"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-a", "--average", help="Average coordinate correlations", action="store_true")
    parser.add_argument("-i", "--iterpose", help="Perform iterposition instead of superposition with first", action="store_true")
    parser.add_argument("-m", "--max_frames", type=int, help="Maximum number of frames to be used from each trajectory", default=100)
    parser.add_argument("-s", "--skip", type=int, help="Skip first N frames of each trajectory", default=20)
    parser.add_argument("-r", "--report", default=None, help="Searches for report files with the given base path (E.g. '-r results/report')")
    
    parser.add_argument("files", nargs='+', help="PDB trajectory files to use")
    options = parser.parse_args()

    print 'Loading coordinates ...'
    all_coordsets = get_coordinates(options.files, options.skip, options.max_frames)
      
    print 'Superimposing ...'
    all_superimposed_coordsets = superimpose_coordinates(all_coordsets, options.iterpose)
    
#     all_superimposed_coordsets =  get_coords_and_superimpose_with_prody(options.files, options.skip, 
#                                                                         options.max_frames, options.iterpose)

    print all_superimposed_coordsets[0].shape
    if options.report is not None:
        print 'Inflating using report ...'
        all_superimposed_coordsets = add_all_trajectories_rejected_from_report(options.files, 
                                                                               all_superimposed_coordsets, 
                                                                               options.report)
    print all_superimposed_coordsets[0].shape
    
    print 'Reshaping ...'
    for traj_coordsets in all_superimposed_coordsets:
        num_frames, num_atoms, coords_per_atom = traj_coordsets.shape
        traj_coordsets.shape = (num_frames, num_atoms*coords_per_atom)
    
    print 'Computing mean positions ...'
    averages, std_devs = computeMeanCoordinatesForEachTrajectory(all_superimposed_coordsets)
    
    print 'Starting autocorrelation ...'
    per_trajectory_per_coordinate_autocorrelations = computeEachCoordinateAutocorrelation(all_superimposed_coordsets, averages, std_devs)

    if options.average:
        print 'Averaging and normalizing ...'
        autocorr_mean, autocorr_std = average_all_per_coordinate_autocorrelations(per_trajectory_per_coordinate_autocorrelations)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.errorbar(x = range(len(autocorr_mean)), y = autocorr_mean, yerr = autocorr_std)
        plt.show()
    else:
        autocorr_per_c_mean = average_per_coordinate_autocorrelations(per_trajectory_per_coordinate_autocorrelations)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for coord in autocorr_per_c_mean:
            ax.plot(coord)
        plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     if AVG:
#         ax.plot(c[:maxCorrelationTime])
#     else:
#         for i in range(c.shape[1]):
#             ax.plot(c[:maxCorrelationTime,i])
#     #ax.set_yscale('log')
#     plt.show()
# 
# 
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     if AVG:
#         ax.plot(c[:maxCorrelationTime])
#     else:
#         for i in range(c.shape[1]):
#             ax.plot(c[:maxCorrelationTime,i])
#     ax.set_yscale('log')
#     plt.show()
