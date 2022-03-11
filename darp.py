import numpy as np
import copy
import sys
import cv2
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
def assign(droneNo, rows, cols, initial_positions, GridEnv, MetricMatrix, A, poids_matrice, passage):

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
                weight = poids_matrice[i,j]
                A[i][j] = indMin
                #ArrayOfElements counts the k_i, now pondered by  the vertices' weights
                if weight > 0:
                    ArrayOfElements[indMin] += weight
                else:
                    #when this tile is a passage tile, we want to make the passage taxing
                    ArrayOfElements[indMin] -= weight

            elif GridEnv[i, j] == -2:
                A[i, j] = droneNo
    return A, ArrayOfElements

@njit(fastmath=True)
def inverse_binary_map_as_uint8(BinaryMap):
    # cv2.distanceTransform needs input of dtype unit8 (8bit)
    return np.logical_not(BinaryMap).astype(np.uint8)

@njit(fastmath=True)
def euclidian_distance_points2d(array1: np.array, array2: np.array) -> np.float_:
    # this runs much faster than the (numba) np.linalg.norm and is totally enough for our purpose
    return (
                   ((array1[0] - array2[0]) ** 2) +
                   ((array1[1] - array2[1]) ** 2)
           ) ** 0.5

@njit(fastmath=True)
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


def ConnectedComponentWarpDistance(num_labels, labels_im, poids_matrice, r_position, passage_positions, max_coeff):
    weights = np.zeros(num_labels-1)
    total_weight = 0
    max_weight = -1
    connected = True
    initial_label = labels_im[r_position[0], r_position[1]]
    rows, cols = np.shape(poids_matrice)

    distance_matrix = WarpDistanceToRegion(labels_im, poids_matrice, passage_positions, r_position, initial_label)

    for label in range(1,num_labels):
        #revoir cette portion, qu'elle soit plus efficace
        min_distance_path_to_label = 2**30
        for x in range(rows):
            for y in range(cols):
                if labels_im[x,y] == label:
                    min_distance_path_to_label = min(min_distance_path_to_label, distance_matrix[x,y])
        weight = min_distance_path_to_label

        weights[label-1] = weight
        if weight < 2**30:
            if weight >0:
                total_weight += weight
            if weight > max_weight:
                max_weight = weight
        else:
            connected = False

    print(weights)

    tab_coeffs = np.ones(num_labels)
    if max_weight>0:
        for label in range(num_labels-1):
            if weights[label]<2**30:
                tab_coeffs[label+1] = max_coeff * ((weights[label]**3)/(max_weight**3))
    return tab_coeffs, total_weight, connected

<<<<<<< HEAD
#utility for Djikstra, as priority aren't natively implemented in Python (not with the ability to change priorities)
#   I resorted to using lists instead, as it is only a cost taking place once
def min_unvisited_from_dict(distances, unvisited_passages):
    index, minima = (-1,-1), 2**30
    for (x,y) in unvisited_passages.keys():
        d = distances[x,y]
        if d<minima:
            index=(x,y)
            minima = d
    return index, minima

#updating the distances of neighbours in the grid when visiting (x,y) with distance dist_u
def exploration_neighbours(rows, cols, poids_matrice, distances_from_robots, dist_u, ux, uy):
    if uy != 0:
        if poids_matrice[ux, uy-1] < 0:
            alternative_path = dist_u - poids_matrice[ux, uy-1]
            if alternative_path < distances_from_robots[ux, uy-1]:
                distances_from_robots[ux, uy-1] = alternative_path
        else: #we only look at the river-crossing cost
            distances_from_robots[ux, uy-1] = dist_u
    if uy != cols-1:
        if poids_matrice[ux, uy+1] < 0:
            alternative_path = dist_u - poids_matrice[ux, uy+1]
            if alternative_path < distances_from_robots[ux, uy+1]:
                distances_from_robots[ux, uy+1] = alternative_path
        else:
            distances_from_robots[ux, uy+1] = dist_u
    if ux != 0:
        if poids_matrice[ux-1, uy] <0:
            alternative_path = dist_u - poids_matrice[ux-1, uy]
            if alternative_path < distances_from_robots[ux-1, uy]:
                distances_from_robots[ux-1, uy] = alternative_path
        else:
            distances_from_robots[ux-1, uy] = dist_u
    if ux != rows-1:
        if poids_matrice[ux+1, uy] <0:
            alternative_path = dist_u - poids_matrice[ux+1, uy]
            if alternative_path < distances_from_robots[ux+1, uy]:
                distances_from_robots[ux+1, uy] = alternative_path
        else:
            distances_from_robots[ux+1, uy] = dist_u


#djikstra's algorithm, except it's implemented with lists instead of priority queues
#We are only looking at paths through passage tiles
def WarpDistanceToRegion(labels_im, poids_matrice, passage_positions, initial_position, initial_label):
    rows, cols = np.shape(poids_matrice)
    distances_from_robot = np.full((rows, cols), 2**30)
    rx, ry = initial_position[0], initial_position[1]

    unvisited_set = {}
    for (x,y,weight) in passage_positions:
        unvisited_set[x,y]=True

    ux, uy, dist_u = rx, ry, 0

    #We look at the distance from the initial connected component, so we put this whole component at distance 0
    for x in range(rows):
        for y in range(cols):
            if labels_im[x,y] == initial_label:
                exploration_neighbours(rows, cols, poids_matrice, distances_from_robot, 0, x, y)

    distances_from_robot[rx, ry] = 0
    
    (ux, uy), dist_u = min_unvisited_from_dict(distances_from_robot, unvisited_set)

    while not ux==-1:
        unvisited_set.pop((ux,uy))
        #We thus enumerate our (possible) neighbours
        exploration_neighbours(rows, cols, poids_matrice, distances_from_robot, dist_u, ux, uy)

        (ux, uy), dist_u = min_unvisited_from_dict(distances_from_robot, unvisited_set)

    return distances_from_robot
=======
>>>>>>> main


class DARP:
    def __init__(self, nx, ny, notEqualPortions, given_initial_positions, given_portions, obstacles_positions,
                 visualization, MaxIter=80000, CCvariation=0.01,
                 randomLevel=0.0001, dcells=2,
<<<<<<< HEAD
                 importance=False, poids = [], tps_affichage = 0.05, given_passage = []):
                 #given_passage corresponds to the list of tiles that you don't need to explore, but can pass throgh.
                 #  they can be seen as "semi-obstacles"
=======
                 importance=False, poids = [], tps_affichage = 0.05, reduction_step_power = 8):
>>>>>>> main

        self.rows = nx
        self.cols = ny

        print("dimensions :", nx, ny)

        #convenience boolean value to accelerate the computation in the uniform case
        self.poids_uniforme = True
        if poids != []:
            self.poids_uniforme = False

        

        self.initial_positions, self.obstacles_positions, self.poids_positions, self.passage_positions, self.portions = self.sanity_check(given_initial_positions, given_portions, obstacles_positions,
                                                                                                                                         notEqualPortions, poids, given_passage)


        self.droneNo = len(self.initial_positions)
        self.passage = False
        self.PassageNo = 0
        if given_passage != []:
            self.passage = True
            self.passageNo = len(given_passage)

        self.visualization = visualization
        self.MaxIter = MaxIter
        self.CCvariation = CCvariation
        self.randomLevel = randomLevel
        self.dcells = dcells
        self.importance = importance
        self.notEqualPortions = notEqualPortions
        self.tps_affichage = tps_affichage
        self.reduction_step = 1-10**(-reduction_step_power)
        self.current_reduction = 1
    

        print("\nInitial Conditions Defined:")
        print("Grid Dimensions:", nx, ny)
        print("Number of Robots:", len(self.initial_positions))
        if self.passage:
            print("with the presence of an additional false robot")
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

        self.AllDistances, self.termThr, dcells_weight, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance= self.construct_Assignment_Matrix()
        self.MetricMatrix = copy.deepcopy(self.AllDistances)
        self.ArrayOfElements = np.zeros(self.droneNo)
        self.color = []

        if self.dcells ==2: #that is, the default value
            self.dcells = dcells_weight
            #There's the slight issue if you want dcells to be EXACTLY 2 despiste weights, but well, you can always put 2.01 then

        print("The minimum deviation is", self.termThr, "and it will go up to", self.termThr+self.dcells)

        for r in range(self.droneNo):
            np.random.seed(r)
            self.color.append(list(np.random.choice(range(256), size=3)))
        
        np.random.seed(1)
        if self.visualization:
            self.assignment_matrix_visualization = darp_area_visualization(self.A, self.droneNo, self.color, self.initial_positions)

    #Checking that no trivial incoherent input has been given
    def sanity_check(self, given_initial_positions, given_portions, obs_pos, notEqualPortions, poids, given_passages):
        
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

        passage_positions = []
        for (position, weight) in given_passages:
            if position < 0 or position >= self.rows * self.cols:
                print("Initial positions should be inside the Grid.")
                sys.exit(1)
            passage_positions.append((position // self.cols, position % self.cols, weight))

        #Checking similarly that the weights' positions are correct
        poids_positions = []
        for (cell, weight) in poids:
            if cell < 0 or obstacle >= self.rows * self.cols:
                print("Weighted vertexes should be inside the Grid.")
                sys.exit(6)
            if weight == 0:
                print("Weighted vertexes should have non-null weight.")
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

        for position in passage_positions:
            for obstacle in obstacles_positions:
                if position[0] == obstacle[0] and position[1] == obstacle[1]:
                    print("passage tiles should not be on obstacles")
                    sys.exit(5)

        return initial_positions, obstacles_positions, poids_positions, passage_positions, portions
          
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

        #We add the passage tiles as obstacles after this verification
        for x,y, weight in self.passage_positions:
            self.poids_matrice[x,y]= -weight
            self.GridEnv[x,y]=-2
        
        # initial robot tiles will have their array.index as value
        for idx, robot in enumerate(self.initial_positions):
            self.GridEnv[robot] = idx
            self.A[robot] = idx

        return

    def divideRegions(self):
        success = False
        cancelled = False
        criterionMatrix = np.zeros((self.rows, self.cols))
<<<<<<< HEAD
        total_iteration = 0

        #as it is supposed to be defined out of the while loop for the last return
        iteration=0
=======
        iteration = 0
        total_iteration=0
>>>>>>> main

        while self.termThr <= self.dcells and not success and not cancelled:
            downThres = (self.Notiles - self.termThr*(self.droneNo-1))/(self.Notiles*self.droneNo)
            upperThres = (self.Notiles + self.termThr)/(self.Notiles*self.droneNo)

            success = True

            # Main optimization loop

            iteration=0

            while iteration <= self.MaxIter and not cancelled:
                self.A, self.ArrayOfElements = assign(self.droneNo,
                                                                   self.rows,
                                                                   self.cols,
                                                                   self.initial_positions,
                                                                   self.GridEnv,
                                                                   self.MetricMatrix,
                                                                   self.A,
                                                                   self.poids_matrice,
                                                                   self.passage)
                #here however we only look at the droneNo "true" drones, as we have no connectivity constraint on the false one
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
                        BinaryRobot, BinaryNonRobot = constructBinaryImages(labels_im, self.initial_positions[r], self.rows, self.cols)
                        ConnectedMultiplier, added_weight, connected = self.CalcConnectedMultiplier(self.rows, self.cols,
                                                                      self.NormalizedEuclideanDistanceBinary(True, BinaryRobot, BinaryNonRobot),
<<<<<<< HEAD
                                                                      self.NormalizedEuclideanDistanceBinary(False, BinaryRobot, BinaryNonRobot),self.CCvariation,
                                                                      num_labels, labels_im, r)
                        ConnectedRobotRegions[r] = connected
=======
                                                                      self.NormalizedEuclideanDistanceBinary(False, BinaryRobot, BinaryNonRobot),self.CCvariation*self.current_reduction)
>>>>>>> main
                    ConnectedMultiplierList[r, :, :] = ConnectedMultiplier
                    self.ArrayOfElements[r]+=added_weight

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

                #old_metric = np.zeros((self.droneNo, self.rows, self.cols))
                for r in range(self.droneNo):
                    if totalNegPlainErrors != 0:
                        # This conditions seems useless to me : we are adding the ratios plainErrors[r], which are thus always >0
                        if divFairError[r] < 0:
                            correctionMult[r] = 1 + (plainErrors[r]/totalNegPlainErrors)*(TotalNegPerc/2)*self.current_reduction
                        else:
                            correctionMult[r] = 1 - (plainErrors[r]/totalNegPlainErrors)*(TotalNegPerc/2)*self.current_reduction

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

<<<<<<< HEAD
                #loop to keep values in check : we have no need for values spanning from 1e-50 to 1e+50
                if total_iteration % 30 == 0:
                    for x in range(self.rows):
                        for y in range(self.cols):
                            for r in range(self.droneNo):
                                self.MetricMatrix[r,x,y] = np.power(self.MetricMatrix[r,x,y], 0.95)

                total_iteration +=1
=======
                """
                for r in range(self.droneNo):
                    for i in range(min(10, self.rows)):
                        for j in range(self.cols):
                            if old_metric[r,i,j] != self.MetricMatrix[r,i,j]:
                                print(r,i,j, " : old = ", old_metric[r,i,j], "new =", self.MetricMatrix[r,i,j])
                print()
                """

>>>>>>> main
                iteration += 1
                total_iteration +=1
                #we reduce the size of our steps every so often
                if total_iteration %30 ==0:
                    self.current_reduction *= self.reduction_step
                    print("current weakening :", self.current_reduction)
                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(self.A, iteration_number=iteration)
                    time.sleep(self.tps_affichage)

            if iteration >= self.MaxIter:
                self.MaxIter = self.MaxIter/2
                success = False
                self.termThr += 1

        print("exiting with a deviation of", self.termThr)
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
        """
        print("current one", currentOne)
        print("criterion", CM)
        print("random", RM)
        print("connected multiplier", CC, "\n")
        """
        MMnew = currentOne*CM*RM*CC

        return MMnew

    def IsThisAGoalState(self, thresh, connectedRobotRegions):
        print("goal verification : weight repartition", self.ArrayOfElements, "and connectedness :", connectedRobotRegions)
        for r in range(self.droneNo):
            if np.absolute(self.DesireableAssign[r] - self.ArrayOfElements[r]) > thresh or not connectedRobotRegions[r]:
                return False
        #For the fake drone, we do not add any success condition : we just want it to be here to compete with other drones on passage tiles
        return True

    def update_connectivity(self):
        self.connectivity = np.zeros((self.droneNo, self.rows, self.cols))
        for i in range(self.droneNo):
            mask = np.where(self.A == i)
            self.connectivity[i, mask[0], mask[1]] = 255


    #Defining the distances to robots, as well as the desirable number of tiles for each robot
    def construct_Assignment_Matrix(self):
        Notiles = self.rows*self.cols
        effectiveSize = 0
        total_passage_weight= 0
        if self.poids_uniforme:
            effectiveSize = Notiles - self.droneNo - len(self.obstacles_positions) - len(self.empty_space)
        else:
            max_weight = -1
            min_weight = 2**30
            #the desirable assign takes into account the vertices' weights
            for x in range(self.rows):
                for y in range(self.rows):
                    if self.GridEnv[x,y] ==-1:
                        weight = self.poids_matrice[x,y]
                        max_weight = max(max_weight, abs(weight))
                        min_weight = min(min_weight, abs(weight))
                        if weight >0:
                            #which allows to not consider passage tiles in our desirable weight
                            effectiveSize += weight
                    elif self.GridEnv[x,y]==-2 and self.poids_matrice[x,y] <0:
                        total_passage_weight -= self.poids_matrice[x,y]
        print("effective size :", effectiveSize)
        
        termThr = 0
        dcells =2
        #possible issue if the weights are all devisible by 2, for instance
        if effectiveSize % self.droneNo != 0:
            termThr = 1
        if not self.poids_uniforme:
            termThr = min_weight
            dcells = max_weight + (total_passage_weight // self.droneNo)
            print("max_weight =", max_weight, ", dcells =", dcells)

        DesireableAssign = np.zeros(self.droneNo)
        MaximunDist = np.zeros(self.droneNo)
        MaximumImportance = np.zeros(self.droneNo)
        MinimumImportance = np.zeros(self.droneNo)

        for i in range(self.droneNo):
            DesireableAssign[i] = effectiveSize * self.portions[i]
            MinimumImportance[i] = sys.float_info.max
            if (DesireableAssign[i] != int(DesireableAssign[i]) and termThr != 1):
                termThr = 1

        AllDistances = np.zeros((self.droneNo, self.rows, self.cols))
        TilesImportance = np.zeros((self.droneNo, self.rows, self.cols))


        for x in range(self.rows):
            for y in range(self.cols):
                tempSum = 0
                for r in range(self.droneNo):
                    AllDistances[r, x, y] = euclidian_distance_points2d(np.array(self.initial_positions[r]), np.array((x, y))) # E!
                    if AllDistances[r, x, y] > MaximunDist[r]:
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

        return AllDistances, termThr, dcells, Notiles, DesireableAssign, TilesImportance, MinimumImportance, MaximumImportance

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

    def CalcConnectedMultiplier(self, rows, cols, dist1, dist2, CCvariation, num_labels, labels_im, r):
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

        coeffs_labels, total_weight, connected = ConnectedComponentWarpDistance(num_labels, labels_im, self.poids_matrice, 
                                                    self.initial_positions[r], self.passage_positions, 1)

        #we split proportionnally the differences between 1-CCvariation and 1+CCvariation
        for i in range(rows):
            for j in range(cols):
                #in the case of a disconnected component
                if returnM[i,j]>0:
                    returnM[i,j]*= coeffs_labels[labels_im[i,j]]
                returnM[i, j] = (returnM[i, j]-MinV)*((2*CCvariation)/(MaxV - MinV)) + (1-CCvariation)

        return returnM, total_weight, connected

    def NormalizedEuclideanDistanceBinary(self, RobotR, BinaryRobot, BinaryNonRobot):
        if RobotR:
            distRobot = cv2.distanceTransform(inverse_binary_map_as_uint8(BinaryRobot), distanceType=2, maskSize=0, dstType=5)
        else:
            distRobot = cv2.distanceTransform(inverse_binary_map_as_uint8(BinaryNonRobot), distanceType=2, maskSize=0, dstType=5)

        MaxV = np.max(distRobot)
        MinV = np.min(distRobot)

        #Normalization
        if RobotR:
            distRobot = (distRobot - MinV)*(1/(MaxV-MinV)) + 1
        else:
            distRobot = (distRobot - MinV)*(1/(MaxV-MinV))

        return distRobot

