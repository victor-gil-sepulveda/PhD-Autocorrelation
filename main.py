import numpy
from pyRMSD.utils.proteinReading import Reader
from pyRMSD.RMSDCalculator import RMSDCalculator
import argparse

def get_coordinates(trajectories):
    all_coordsets = []
    for traj_path in trajectories:
        coords = Reader().readThisFile(traj_path).gettingOnlyCAs().read()
        all_coordsets.append(coords)
    return numpy.array(all_coordsets)

def superimpose_coordinates(all_coordsets, iterpose = True):
    all_superimposed_coordsets = []
    for coordsets in all_coordsets:
        calculator = RMSDCalculator(calculatorType = "QTRFIT_OMP_CALCULATOR",
                                fittingCoordsets = coordsets)
        if iterpose:
            superimposed_coordsets = calculator.iterativeSuperposition()
        else:
            superimposed_coordsets = calculator.iterativeSuperposition()
            _, superimposed_coordsets = calculator.oneVsTheOthers(0, get_superposed_coordinates = True)
        all_superimposed_coordsets.append(superimposed_coordsets)
    return numpy.array(all_superimposed_coordsets)

def get_max_num_frames(all_coordsets):
    max_num_frames = 0
    for coordsets in all_coordsets:
        max_num_frames = max(max_num_frames,len(coordsets))
    return max_num_frames

if __name__ == "__main__":
    
    desc = "Program that calculates the autocorrelation for a set of atoms"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-o", "--output", help="Output filename", default="/tmp/autocorr.txt")
    parser.add_argument("-a", "--average", help="Average atoms", action="store_true")
    parser.add_argument("-c", "--choice", default='FULL', choices=[NUMPY, FULL, OWN], type=str.upper, help="Method to compute autocorrelation")
    parser.add_argument("-m", "--max", type=int, default=100)
    parser.add_argument("files", nargs='+', help="Files to add in correlation calculation")
    options = parser.parse_args()

#     return args.files, args.max, args.choice, args.average, args.output
#     trajectories, maxCorrelationTime, choice, AVG, outputFilename = parseArguments()
    
    all_coordsets = get_coordinates(options.files)
    
    all_superimposed_coordsets = superimpose_coordinates(all_coordsets, iterpose = True)
    
    c = standardAutocorrelationMethod(trajectories, AVG)


    np.savetxt(outputFilename, c, newline="\n")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if AVG:
        ax.plot(c[:maxCorrelationTime])
    else:
        for i in range(c.shape[1]):
            ax.plot(c[:maxCorrelationTime,i])
    #ax.set_yscale('log')
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if AVG:
        ax.plot(c[:maxCorrelationTime])
    else:
        for i in range(c.shape[1]):
            ax.plot(c[:maxCorrelationTime,i])
    ax.set_yscale('log')
    plt.show()
