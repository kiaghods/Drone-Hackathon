import numpy as np
import copy
import sys
import cv2
import random
from scipy import ndimage
from Visualization import darp_area_visualization
import time
import random
import os
from numba import njit
np.set_printoptions(threshold=sys.maxsize)

random.seed(1)
os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)

#We create the assignation matrix A, from the distance matrices MetricMatrix[i] = E_i
#BWlist[i] has value 1 in the cells attributed to the drone i, 0 elsewhere
@njit
def assign(droneNo, rows, cols, initial_positions, GridEnv, MetricMatrix, A, poids_matrice):
    BWlist = np.zeros((droneNo, rows, cols))
    for r in range(droneNo):
        BWlist[r, initial_positions[r][0], initial_positions[r][1]] = 1

    ArrayOfElements = np.zeros(droneNo)
    for i in range(rows):
        for j in range(cols):
            if GridEnv[i, j] == -1:
                minV = MetricMatrix[0, i, j]
                indMin = 0
                for r in range(droneNo):
                    if MetricMatrix[r, i, j] < minV:
                        minV = MetricMatrix[r, i, j]
                        indMin = r

                A[i][j] = indMin
                BWlist[indMin, i, j] = 1
                #ArrayOfElements counts the k_i, now pondered by  the vertices' weights
                ArrayOfElements[indMin] += poids_matrice[i,j]

            elif GridEnv[i, j] == -2:
                A[i, j] = droneNo
    return BWlist, A, ArrayOfElements

#generates, for the robot beginning in robo_start_point, the binary assignation values over the grid's cells
@njit
def constructBinaryImages(A, robo_start_point, rows, cols):
    BinaryRobot = np.copy(A)
    BinaryNonRobot = np.copy(A)
    for i in range(rows):
        for j in range(cols):
            if A[i, j] == A[robo_start_point]:
                BinaryRobot[i, j] = 1
                BinaryNonRobot[i, j] = 0
            elif A[i, j] != 0:
                BinaryRobot[i, j] = 0
                BinaryNonRobot[i, j] = 1

    return BinaryRobot, BinaryNonRobot

@njit
def CalcConnectedMultiplier(rows, cols, dist1, dist2, CCvariation):
    returnM = np.zeros((rows, cols))
    MaxV = 0
    MinV = 2**30

    for i in range(rows):
        for j in range(cols):
            returnM[i, j] = dist1[i, j] - dist2[i, j]
            if MaxV < returnM[i, j]:
                MaxV = returnM[i, j]
            if MinV > returnM[i, j]:
                MinV = returnM[i, j]

    #we split proportionnally the differences between 1-CCvariation and 1+CCvariation
    for i in range(rows):
        for j in range(cols):
            returnM[i, j] = (returnM[i, j]-MinV)*((2*CCvariation)/(MaxV - MinV)) + (1-CCvariation)

    return returnM

#utility for Djikstra, as priority aren't natively implemented in Python (not with the ability to change priorities)
#   I resorted to using lists instead, as it is only a cost taking place once
def min_unvisited_from_list(list, unvisited):
    index, minima = (-1,-1), 2**30
    rows, cols = len(list), len(list[0])
    for x in range(rows):
        for y in range(cols):
            if unvisited[x,y]:
                d = list[x,y]
                if d<minima:
                    index=(x,y)
                    minima = d
    return index, minima

#updating the distances of neighbours in the grid when visiting (x,y) with distance dist_u
def exploration_neighbours(rows, cols, poids_matrice, distances_from_robots, r, dist_u, ux, uy):
    if uy != 0:
        alternative_path = dist_u + poids_matrice[ux, uy-1]
        if alternative_path < distances_from_robots[r,ux, uy-1]:
            distances_from_robots[r,ux, uy-1] = alternative_path
    if uy != cols-1:
        alternative_path = dist_u + poids_matrice[ux, uy+1]
        if alternative_path < distances_from_robots[r,ux, uy+1]:
            distances_from_robots[r,ux, uy+1] = alternative_path
    if ux != 0:
        alternative_path = dist_u + poids_matrice[ux-1, uy]
        if alternative_path < distances_from_robots[r,ux-1, uy]:
            distances_from_robots[r,ux-1, uy] = alternative_path
    if ux != rows-1:
        alternative_path = dist_u + poids_matrice[ux+1, uy]
        if alternative_path < distances_from_robots[r,ux+1, uy]:
            distances_from_robots[r,ux+1, uy] = alternative_path

#djikstra's algorithm, except it's implemented with lists instead of priority queues
def bad_djikstra(poids_matrice, rows, cols, GridEnv, initial_positions):
    droneNo = len(initial_positions)
    vertexesNo = rows*cols
    distances_from_robots = np.full((droneNo, rows, cols), 2**30)

    for idx, (x,y) in enumerate(initial_positions):
        distances_from_robots[idx,x,y] = 0
    
    for r in range(droneNo):
        rx, ry = initial_positions[r][0], initial_positions[r][1]
        unvisited_set = np.full((rows, cols), True)
        unvisited_set[rx, ry] = 0

        ux, uy, dist_u = rx, ry, 0
        
        while not ux==-1:
            #we mark that vertex as visited
            unvisited_set[ux, uy]=False
            #We can leave from any empty tile or from ourself, but not from obstacles nor from other robots
            if GridEnv[ux, uy]== -1 or GridEnv[ux, uy]==r:
                #We thus enumerate our (possible) neighbours
                exploration_neighbours(rows, cols, poids_matrice, distances_from_robots, r, dist_u, ux, uy)

            (ux, uy), dist_u = min_unvisited_from_list(distances_from_robots[r], unvisited_set)
    return distances_from_robots


class DARP():
    def __init__(self, nx, ny, notEqualPortions, given_initial_positions, given_portions, obstacles_positions,
                 visualization, MaxIter=80000, CCvariation=0.01,
                 randomLevel=0.0001, dcells=2,
                 importance=False, poids = [], tps_affichage = 0.05):

        self.rows = nx
        self.cols = ny

        print("dimensions :", nx, ny)

        #convenience boolean value to accelerate the computation in the uniform case
        self.poids_uniforme = True
        if poids != []:
            self.poids_uniforme = False

        self.initial_positions, self.obstacles_positions, self.poids_positions, self.portions = self.sanity_check(given_initial_positions, given_portions, obstacles_positions, notEqualPortions, poids)

        self.visualization = visualization
        self.MaxIter = MaxIter
        self.CCvariation = CCvariation
        self.randomLevel = randomLevel
        self.dcells = dcells
        self.importance = importance
        self.notEqualPortions = notEqualPortions
        self.tps_affichage = tps_affichage
    

        print("\nInitial Conditions Defined:")
        print("Grid Dimensions:", nx, ny)
        print("Number of Robots:", len(self.initial_positions))
        print("Initial Robots' positions", self.initial_positions)
        print("Portions for each Robot:", self.portions)
        print("maximum number of iterations allowed:", MaxIter, "\n")


        self.empty_space = []
        if self.rows > self.cols:
            for j in range(self.cols, self.rows):
                for i in range(self.rows):
                    self.empty_space.append((i, j))
            self.cols = self.rows
        elif self.cols > self.rows:
            for j in range(self.rows, self.cols):
                for i in range(self.cols):
                    self.empty_space.append((j, i))
            self.rows = self.cols

        self.droneNo = len(self.initial_positions)
        self.A = np.zeros((self.rows, self.cols))
        self.poids_matrice = np.full((self.rows, self.cols), 1)
        self.defineGridEnv()
   
        self.connectivity = np.zeros((self.droneNo, self.rows, self.cols))
        self.BinaryRobotRegions = np.zeros((self.droneNo, self.rows, self.cols), dtype=bool)

        self.AllDistances, self.termThr, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance= self.construct_Assignment_Matrix()
        self.MetricMatrix = copy.deepcopy(self.AllDistances)
        self.BWlist = np.zeros((self.droneNo, self.rows, self.cols))
        self.ArrayOfElements = np.zeros(self.droneNo)
        self.color = []

        for r in range(self.droneNo):
            self.color.append(list(np.random.choice(range(256), size=3)))

        if self.visualization:
            self.assignment_matrix_visualization = darp_area_visualization(self.A, self.droneNo, self.color, self.initial_positions)

    #Checking that no trivial incoherent input has been given
    def sanity_check(self, given_initial_positions, given_portions, obs_pos, notEqualPortions, poids):
        
        initial_positions = []
        for position in given_initial_positions:
            if position < 0 or position >= self.rows * self.cols:
                print("Initial positions should be inside the Grid.")
                sys.exit(1)
            initial_positions.append((position // self.cols, position % self.cols))

        obstacles_positions = []
        for obstacle in obs_pos:
            if obstacle < 0 or obstacle >= self.rows * self.cols:
                print("Obstacles should be inside the Grid.")
                sys.exit(2)
            obstacles_positions.append((obstacle // self.cols, obstacle % self.cols))

        #Checking similarly that the weights' positions are correct
        poids_positions = []
        for (cell, weight) in poids:
            if cell < 0 or obstacle >= self.rows * self.cols:
                print("Weighted vertexes should be inside the Grid.")
                sys.exit(6)
            if weight <= 0:
                print("Weighted vertexes should have (strictly) positive weight.")
                sys.exit(7)
            poids_positions.append((cell // self.cols, cell%self.cols, weight))

        portions = []
        if notEqualPortions:
            portions = given_portions
        else:
            for drone in range(len(initial_positions)):
                portions.append(1 / len(initial_positions))

        if len(initial_positions) != len(portions):
            print("Portions should be defined for each drone")
            sys.exit(3)

        s = sum(portions)
        if abs(s - 1) >= 0.0001:
            print("Sum of portions should be equal to 1.")
            sys.exit(4)

        for position in initial_positions:
            for obstacle in obstacles_positions:
                if position[0] == obstacle[0] and position[1] == obstacle[1]:
                    print("Initial positions should not be on obstacles")
                    sys.exit(5)

        return initial_positions, obstacles_positions, poids_positions, portions
          
    def defineGridEnv(self):
        self.GridEnv = np.full(shape=(self.rows, self.cols), fill_value=-1)  # create non obstacle map with value -1

        # obstacle tiles value is -2
        for idx, obstacle_pos in enumerate(self.obstacles_positions):
            self.GridEnv[obstacle_pos[0], obstacle_pos[1]] = -2
        for idx, es_pos in enumerate(self.empty_space):
            self.GridEnv[es_pos] = -2
        connectivity = np.zeros((self.rows, self.cols))

        #adding the weights
        for x,y,weight in self.poids_positions:
            self.poids_matrice[x,y]=weight
        
        #defining regions' connectivity
        mask = np.where(self.GridEnv == -1)
        connectivity[mask[0], mask[1]] = 255
        image = np.uint8(connectivity)
        num_labels, labels_im = cv2.connectedComponents(image, connectivity=4)

        if num_labels > 2:
            print("The environment grid MUST not have unreachable and/or closed shape regions")
            sys.exit(6)
        
        # initial robot tiles will have their array.index as value
        for idx, robot in enumerate(self.initial_positions):
            self.GridEnv[robot] = idx
            self.A[robot] = idx

        return

    def divideRegions(self):
        success = False
        cancelled = False
        criterionMatrix = np.zeros((self.rows, self.cols))

        while self.termThr <= self.dcells and not success and not cancelled:
            downThres = (self.Notiles - self.termThr*(self.droneNo-1))/(self.Notiles*self.droneNo)
            upperThres = (self.Notiles + self.termThr)/(self.Notiles*self.droneNo)

            success = True

            # Main optimization loop
            iteration = 0

            while iteration <= self.MaxIter and not cancelled:
                self.BWlist, self.A, self.ArrayOfElements = assign(self.droneNo,
                                                                   self.rows,
                                                                   self.cols,
                                                                   self.initial_positions,
                                                                   self.GridEnv,
                                                                   self.MetricMatrix,
                                                                   self.A,
                                                                   self.poids_matrice)
                ConnectedMultiplierList = np.ones((self.droneNo, self.rows, self.cols))
                ConnectedRobotRegions = np.zeros(self.droneNo)
                plainErrors = np.zeros((self.droneNo))
                #same as plainErrors, but reduced by the eventual allowed threshold
                divFairError = np.zeros((self.droneNo))

                for r in range(self.droneNo):
                    ConnectedMultiplier = np.ones((self.rows, self.cols))
                    ConnectedRobotRegions[r] = True
                    self.update_connectivity()
                    image = np.uint8(self.connectivity[r, :, :])
                    num_labels, labels_im = cv2.connectedComponents(image, connectivity=4)
                    if num_labels > 2:
                        ConnectedRobotRegions[r] = False
                        BinaryRobot, BinaryNonRobot = constructBinaryImages(labels_im, self.initial_positions[r], self.rows, self.cols)
                        ConnectedMultiplier = CalcConnectedMultiplier(self.rows, self.cols,
                                                                      self.NormalizedEuclideanDistanceBinary(True, BinaryRobot, BinaryNonRobot),
                                                                      self.NormalizedEuclideanDistanceBinary(False, BinaryRobot, BinaryNonRobot),self.CCvariation)
                    ConnectedMultiplierList[r, :, :] = ConnectedMultiplier
                    plainErrors[r] = self.ArrayOfElements[r]/(self.DesireableAssign[r]*self.droneNo)
                    if plainErrors[r] < downThres:
                        divFairError[r] = downThres - plainErrors[r]
                    elif plainErrors[r] > upperThres:
                        divFairError[r] = upperThres - plainErrors[r]

                if self.IsThisAGoalState(self.termThr, ConnectedRobotRegions):
                    break

                TotalNegPerc = 0
                totalNegPlainErrors = 0
                correctionMult = np.zeros(self.droneNo)

                for r in range(self.droneNo):
                    if divFairError[r] < 0:
                        TotalNegPerc += np.absolute(divFairError[r])
                        totalNegPlainErrors += plainErrors[r]

                    correctionMult[r] = 1

                for r in range(self.droneNo):
                    if totalNegPlainErrors != 0:
                        # This conditions seems useless to me : we are adding the ratios plainErrors[r], which are thus always >0
                        if divFairError[r] < 0:
                            correctionMult[r] = 1 + (plainErrors[r]/totalNegPlainErrors)*(TotalNegPerc/2)
                        else:
                            correctionMult[r] = 1 - (plainErrors[r]/totalNegPlainErrors)*(TotalNegPerc/2)

                        criterionMatrix = self.calculateCriterionMatrix(
                                self.TilesImportance[r],
                                self.MinimumImportance[r],
                                self.MaximumImportance[r],
                                correctionMult[r],
                                divFairError[r] < 0)

                    #the random matrix only shifts things by a small difference to 1 (per default <= e-4)
                    self.MetricMatrix[r] = self.FinalUpdateOnMetricMatrix(
                            criterionMatrix,
                            self.generateRandomMatrix(),
                            self.MetricMatrix[r],
                            ConnectedMultiplierList[r, :, :])

                iteration += 1
                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(self.A, iteration_number=iteration)
                    time.sleep(self.tps_affichage)

            if iteration >= self.MaxIter:
                self.MaxIter = self.MaxIter/2
                success = False
                self.termThr += 1

        self.getBinaryRobotRegions()
        return success, iteration

    def getBinaryRobotRegions(self):
        ind = np.where(self.A < self.droneNo)
        temp = (self.A[ind].astype(int),)+ind
        self.BinaryRobotRegions[temp] = True

    def generateRandomMatrix(self):
        RandomMatrix = np.zeros((self.rows, self.cols))
        RandomMatrix = 2*self.randomLevel*np.random.uniform(0, 1,size=RandomMatrix.shape) + (1 - self.randomLevel)
        return RandomMatrix

    def FinalUpdateOnMetricMatrix(self, CM, RM, currentOne, CC):
        MMnew = np.zeros((self.rows, self.cols))
        MMnew = currentOne*CM*RM*CC

        return MMnew

    def IsThisAGoalState(self, thresh, connectedRobotRegions):
        for r in range(self.droneNo):
            if np.absolute(self.DesireableAssign[r] - self.ArrayOfElements[r]) > thresh or not connectedRobotRegions[r]:
                return False
        return True

    def update_connectivity(self):
        self.connectivity = np.zeros((self.droneNo, self.rows, self.cols))
        for i in range(self.droneNo):
            mask = np.where(self.A == i)
            self.connectivity[i, mask[0], mask[1]] = 255


    #Defining the distances to robots, as well as the desirable number of tiles for each robot
    def construct_Assignment_Matrix(self):
        Notiles = self.rows*self.cols
        fair_division = 1/self.droneNo
        effectiveSize = 0
        if self.poids_uniforme:
            effectiveSize = Notiles - self.droneNo - len(self.obstacles_positions) - len(self.empty_space)
        else:
            #the desirable assign takes into account the vertices' weights
            for x in range(self.rows):
                for y in range(self.rows):
                    if self.GridEnv[x,y] ==-1:
                        effectiveSize += self.poids_matrice[x,y]
        
        termThr = 0
        #possible issue if the weights are all devisible by 2, for instance
        if effectiveSize % self.droneNo != 0:
            termThr = 1

        DesireableAssign = np.zeros(self.droneNo)
        MaximunDist = np.zeros(self.droneNo)
        MaximumImportance = np.zeros(self.droneNo)
        MinimumImportance = np.zeros(self.droneNo)

        for i in range(self.droneNo):
            DesireableAssign[i] = effectiveSize * self.portions[i]
            MinimumImportance[i] = sys.float_info.max
            if (DesireableAssign[i] != int(DesireableAssign[i]) and termThr != 1):
                termThr = 1

        AllDistances = bad_djikstra(self.poids_matrice, self.rows, self.cols, self.GridEnv, self.initial_positions)
        TilesImportance = np.zeros((self.droneNo, self.rows, self.cols))


        for x in range(self.rows):
            for y in range(self.cols):
                tempSum = 0
                for r in range(self.droneNo):
                    if AllDistances[r, x, y] > MaximunDist[r] and AllDistances[r,x,y]<2**30:
                        MaximunDist[r] = AllDistances[r, x, y]
                    tempSum += AllDistances[r, x, y]

                for r in range(self.droneNo):
                    if tempSum - AllDistances[r, x, y] != 0:
                        TilesImportance[r, x, y] = 1/(tempSum - AllDistances[r, x, y])
                    else:
                        TilesImportance[r, x, y] = 1
                    # Todo FixMe!
                    if TilesImportance[r, x, y] > MaximumImportance[r]:
                        MaximumImportance[r] = TilesImportance[r, x, y]

                    if TilesImportance[r, x, y] < MinimumImportance[r]:
                        MinimumImportance[r] = TilesImportance[r, x, y]

        return AllDistances, termThr, Notiles, DesireableAssign, TilesImportance, MinimumImportance, MaximumImportance

    def calculateCriterionMatrix(self, TilesImportance, MinimumImportance, MaximumImportance, correctionMult, smallerthan_zero,):
        returnCrit = np.zeros((self.rows, self.cols))
        if self.importance:
            if smallerthan_zero:
                returnCrit = (TilesImportance- MinimumImportance)*((correctionMult-1)/(MaximumImportance-MinimumImportance)) + 1
            else:
                returnCrit = (TilesImportance- MinimumImportance)*((1-correctionMult)/(MaximumImportance-MinimumImportance)) + correctionMult
        else:
            returnCrit[:, :] = correctionMult

        return returnCrit

    def NormalizedEuclideanDistanceBinary(self, RobotR, BinaryRobot, BinaryNonRobot):
        if RobotR:
            distRobot = ndimage.morphology.distance_transform_edt(np.logical_not(BinaryRobot))
        else:
            distRobot = ndimage.morphology.distance_transform_edt(np.logical_not(BinaryNonRobot))

        MaxV = np.max(distRobot)
        MinV = np.min(distRobot)

        #Normalization
        if RobotR:
            distRobot = (distRobot - MinV)*(1/(MaxV-MinV)) + 1
        else:
            distRobot = (distRobot - MinV)*(1/(MaxV-MinV))

        return distRobot

