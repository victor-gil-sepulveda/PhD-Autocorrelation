import numpy as np
import prody
import matplotlib.pylab as plt
import sys
import argparse

NUMPY = 'NUMPY'
FULL = 'FULL'
OWN = 'OWN'

INITIAL_FRAME = 20

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python

    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    result = estimate_unnormalised_autocorrelation(x, x.mean())
    variance = x.var()
    #it should be equivalent to result/result[0]
    return result/variance

def estimate_unnormalised_autocorrelation(x, mean):
    """
    http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python

    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    x = x-mean
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(1.*(np.arange(n, 0, -1)))
    return result

def estimate_unnormalised_autocorrelation(x, mean):
    """
    http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python

    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    x = x-mean
    r = np.correlate(x, x, mode = 'full')[-n:]
    measures = (1.*(np.arange(n, 0, -1)))
    return r, measures

def addMatricesAndResizeIfNeeded(a, b):
    #resize if cumulator is smaller
    if a.shape[0] < b.shape[0] and a.shape[1] < b.shape[1]:
        a.resize((b.shape[0], b.shape[1]), refcheck=False)
    if a.shape[0] < b.shape[0]:
        a.resize((b.shape[0], a.shape[1]), refcheck=False)

    #restrict summation if cumulator is larger in size
    if a.shape[0] > b.shape[0]:
        a[:b.shape[0]] += b
        return a

    return a + b

def sumOverTrajectories(matrixOfTrajectories):
    sumMatrix = np.zeros((1, 1), dtype=np.float64)

    for snapshotMatrix in matrixOfTrajectories:
        sumMatrix = addMatricesAndResizeIfNeeded(sumMatrix, snapshotMatrix)

    return sumMatrix

def sumOverAtoms(matrix):
    sumArray = matrix.sum(axis=1)
    return sumArray

def sumOverTimes(matrix):
    sumArray = matrix.sum(axis=0)
    return sumArray

def correlate(x):
    """
    computes rirj for all possible pairs
    """
    n = len(x)
    result = np.correlate(x, x, mode = 'full')[-n:]
    measures = np.arange(n, 0, -1)
    return result, measures

def addEachAtomAutocorrelationMeasures(coordinates, numAtoms):
    """
        Computes sum ri*rj, and ri = coords for a single trajectory. Results are stored in different array indexes for different atoms.
    """
    rirj = np.zeros((1, 3*numAtoms), dtype=np.float64)
    ri = np.zeros((1, 3*numAtoms), dtype=np.float64)
    rirjMeasures = np.zeros((1, 3*numAtoms))
    riMeasures = np.zeros((1, 3*numAtoms))

    #convert to 2d matrix
    coordinates = coordinates.reshape((coordinates.shape[0], 3*coordinates.shape[1]))
    #add rows if necessary
    rirj.resize((coordinates.shape[0], rirj.shape[1]))
    ri.resize((coordinates.shape[0], ri.shape[1]))
    rirjMeasures.resize((coordinates.shape[0], rirjMeasures.shape[1]))
    riMeasures.resize((coordinates.shape[0], riMeasures.shape[1]))
    
    for i in range(3*numAtoms):
        #resize vector and don't initialise up
        result, measures = correlate(coordinates[:,i])
        rirj[:,i] += result
        ri[:,i] += coordinates[:,i]
        rirjMeasures[:,i] += measures
        riMeasures[:,i] += np.ones(len(measures))

    return rirj, ri, rirjMeasures, riMeasures


def addTrajectoryCoordinates(coordinates, numAtoms):
    """
        Stores ri for each atom. Results are stored in different array indexes for different atoms.
    """
    ri = np.zeros((1, 3*numAtoms), dtype=np.float64)
    riMeasures = np.zeros((1, 3*numAtoms))

    #convert to 2d matrix
    coordinates = coordinates.reshape((coordinates.shape[0], 3*coordinates.shape[1]))
    #add rows if necessary
    ri.resize((coordinates.shape[0], ri.shape[1]))
    riMeasures.resize((coordinates.shape[0], riMeasures.shape[1]))
    
    for i in range(3*numAtoms):
        ri[:,i] += coordinates[:,i]
        riMeasures[:,i] += 1

    return ri, riMeasures

def average(ri, riMeasures):
    """
        average over trajectories and times
    """
    ri = sumOverTrajectories(ri)
    ri = sumOverTimes(ri)
    riMeasures = sumOverTrajectories(riMeasures)
    riMeasures = sumOverTimes(riMeasures)

    ri = np.divide(ri, riMeasures)
    return ri

def computeEachAtomAllTrajectoriesMean(trajectories):
    """
        Computes the mean of each atom's position in all the trajectories
    """
    ri = []
    riMeasures = []

    for i, traj in enumerate(trajectories):
        #trajectory = prody.parsePDB(traj)
        trajectory = prody.parsePDB(traj, subset='calpha')
        coordinates = trajectory.getCoordsets()

        ensembleTrajectory = prody.PDBEnsemble("Complex")
        ensembleTrajectory.setAtoms(trajectory)
        ensembleTrajectory.addCoordset(coordinates[INITIAL_FRAME:]) 
        ensembleTrajectory.setCoords(coordinates[0]) #reference
        ensembleTrajectory.superpose()
        #ensembleTrajectory = trajectory
        sri, sriMeasures = addTrajectoryCoordinates(ensembleTrajectory.getCoordsets(), trajectory.numAtoms())

        #sri, sriMeasures = addTrajectoryCoordinates(trajectory.getCoordsets(), trajectory.numAtoms())
        ri.append(sri)
        riMeasures.append(sriMeasures)

    return average(ri, riMeasures)

def computeEachAtomsUnnormalisedAutocorrelationForASingleTrajectory(coordinates, avgR):
    #convert to 2d matrix
    coordinates = coordinates.reshape((coordinates.shape[0], 3*coordinates.shape[1]))
    #add rows if necessary
    rirj = np.zeros((coordinates.shape[0], coordinates.shape[1]), dtype=np.float64)
    measures = np.zeros((coordinates.shape[0], coordinates.shape[1]), dtype=np.float64)

    #print avgR
    for i in range(coordinates.shape[1]):
         #print coordinates[:,i]
         srirj, sMeasures = estimate_unnormalised_autocorrelation(coordinates[:,i], avgR[i])
         rirj[:,i] += srirj
         measures[:,i] += sMeasures

    #print rirj
    return rirj, measures

def computeEachAtomsUnnormalisedAutocorrelation(trajectories, avgR):
    """
        Computes the autocorrelation with the formula:
            C(k) = 1/[(n-k)] \sum_{t=1}^{n-k} (Xt - mu)(Xt+k - mu)
        To normalise it wihitn [-1:1]
            c(k) = C(k) / var
        When the true mean \mu and variance \sigma^2 are known, this estimate is unbiased.
    """
    rirj = []
    rirjMeasures = []

    for i, traj in enumerate(trajectories):
        #trajectory = prody.parsePDB(traj)
        trajectory = prody.parsePDB(traj, subset='calpha')
        coordinates = trajectory.getCoordsets()

        #superpose
        ensembleTrajectory = prody.PDBEnsemble("Complex")
        ensembleTrajectory.setAtoms(trajectory)
        ensembleTrajectory.addCoordset(coordinates[INITIAL_FRAME:])
        ensembleTrajectory.setCoords(coordinates[0]) #reference
        ensembleTrajectory.superpose()

        #ensembleTrajectory = trajectory

        srirj, sMeasures = computeEachAtomsUnnormalisedAutocorrelationForASingleTrajectory(ensembleTrajectory.getCoordsets(), avgR)
        rirj.append(srirj)
        rirjMeasures.append(sMeasures)

    rirj = sumOverTrajectories(rirj)
    rirjMeasures = sumOverTrajectories(rirjMeasures)

    return rirj/rirjMeasures

def standardAutocorrelationMethod(trajectories, AVG):
    """
        AVG: averages over all atoms
    """
    #<r> := avgR
    print 'Computing mean positions...'
    avgR = computeEachAtomAllTrajectoriesMean(trajectories)
    print 'Computing Unnormalised Autocorrelation for each atom...'
    rirj = computeEachAtomsUnnormalisedAutocorrelation(trajectories, avgR)

    print 'Normalising...'
    if AVG:
        rirj = sumOverAtoms(rirj)
        normalisation = rirj[0]
        rirj /= normalisation
    else:
        for i in range(rirj.shape[1]):
            normalisation = rirj[0][i]
            rirj[:,i]/=normalisation

    return rirj

def computeEachAtomAllTrajectoriesAutocorrelationMeasures(trajectories):
    """
        Computes sum ri*rj, and ri = coords for a set of trajectories. 
        Results are stored in different array indexes for different atoms, and in different array indexes for different snapshots
    """

    rirj = []
    ri = []
    rirjMeasures = []
    riMeasures = []

    for i, traj in enumerate(trajectories):
        #trajectory = prody.parsePDB(traj)
        trajectory = prody.parsePDB(traj, subset='calpha')
        natoms = trajectory.numAtoms()
        coordinates = trajectory.getCoordsets()

        
        #To superpose
        """
        ensembleTrajectory = prody.PDBEnsemble("Complex")
        ensembleTrajectory.setAtoms(trajectory)
        ensembleTrajectory.addCoordset(coordinates)
        ensembleTrajectory.setCoords(coordinates[0]) #reference
        ensembleTrajectory.superpose()
        """
        
        #coordinates = trajectory.getCoordsets()
        #print coordinates

        srirj, sri, srirjMeasures, sriMeasures = addEachAtomAutocorrelationMeasures(coordinates, natoms)
        rirj.append(srirj)
        ri.append(sri)
        rirjMeasures.append(srirjMeasures)
        riMeasures.append(sriMeasures)

    """
    print '***'
    print ri
    print rirj
    print rirjMeasures
    print riMeasures
    print '***'
    """

    return rirj, ri, rirjMeasures, riMeasures



def parseArguments():
    desc = "Program that calculates the autocorrelation for a set of atoms"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-o", "--output", help="Output filename", default="/tmp/autocorr.txt")
    parser.add_argument("-a", "--average", help="Average atoms", action="store_true")
    parser.add_argument("-c", "--choice", default='FULL', choices=[NUMPY, FULL, OWN], type=str.upper, help="Method to compute autocorrelation")
    parser.add_argument("-m", "--max", type=int, default=100)
    parser.add_argument("files", nargs='+', help="Files to add in correlation calculation")
    args = parser.parse_args()

    return args.files, args.max, args.choice, args.average, args.output


def main(trajectories, maxCorrelationTime, choice, AVG, outputFilename):
    #maxCorrelationTime = -1
    
    c = standardAutocorrelationMethod(trajectories, AVG)

    """
    if AVG:
        for i in c: print i
    else:
        print c
    """

    #sortedIndices = np.argsort(c[25,:])
    #print sortedIndices[:5]
    #print sortedIndices[-5:]


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

if __name__ == "__main__":
    trajectories, maxCorrelationTime, choice, AVG, outputFilename = parseArguments()
    main(trajectories, maxCorrelationTime, choice, AVG, outputFilename)
