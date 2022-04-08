from ast import Return
import math
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

BLINKING_LAPSE = 10
BLINKING_THRES = 3*10**-3

def connectivity_function(a,b):
    return a**0.75-b**0.75

#We create the assignation matrix A, from the distance matrices MetricMatrix[i] = E_i
#BWlist[i] has value 1 in the cells attributed to the drone i, 0 elsewhere
@njit
def assign(droneNo, rows, cols, GridEnv, MetricMatrix, A, poids_matrice):
    blinking = np.full((rows, cols), False)
    priorities = np.zeros((rows, cols))
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
                if indMin != A[i][j]:
                    blinking[i, j] = True
                A[i][j] = indMin
                priorities[i][j] = minV
                #ArrayOfElements counts the k_i, now pondered by  the vertices' weights
                if weight > 0:
                    ArrayOfElements[indMin] += weight
                else:
                    #when this tile is a passage tile, we want to make the passage taxing
                    ArrayOfElements[indMin] -= weight

            elif GridEnv[i, j] == -2:
                A[i, j] = droneNo
    return A, ArrayOfElements, priorities, blinking

@njit(fastmath=True)
def decreasing_factor_for_gradient(min_priorities, metric_matrix, effective_size, poids_matrice):
    rows, cols = np.shape(min_priorities)
    sum = 0
    for x in range(rows):
        for y in range(cols):
            ecart_a_combler = min_priorities[x][y] - metric_matrix[x][y]
            sum+= poids_matrice[x][y] * math.e**(-(ecart_a_combler**2)) / np.sqrt(2*math.pi)
    return sum/np.sqrt(effective_size)

def blinking_frequency(blinking, x, y):
    changes = 0
    for i in range(BLINKING_LAPSE):
        if blinking[i,x,y]:
            changes +=1
    return changes

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

def find_back_path(distance_matrix, poids_matrice, x, y):
    rows, cols = np.shape(distance_matrix)
    dist = distance_matrix[x,y]
    u_x, u_y = x,y
    n_x, n_y = x,y
    list_crossed_tiles = []
    new_dist = dist
    while dist>0:
        #We can only find the smallest distance, as all edges to u_x, u_y have the same weight
        if u_y != 0:
            #we want to be able to join back the river, of same weight as we are
            if distance_matrix[u_x, u_y-1] <  new_dist or (distance_matrix[u_x, u_y-1] == new_dist 
            and poids_matrice[u_x, u_y]>0 and poids_matrice[u_x, u_y-1]<0):
                new_dist = distance_matrix[u_x, u_y-1]
                n_x, n_y = u_x, u_y-1
        if u_y != cols-1:
            if distance_matrix[u_x, u_y+1] <  new_dist or (distance_matrix[u_x, u_y+1] == new_dist and 
            poids_matrice[u_x, u_y]>0 and poids_matrice[u_x, u_y+1]<0):
                    new_dist = distance_matrix[u_x, u_y+1]
                    n_x, n_y = u_x, u_y+1
        if u_x != 0:
            if distance_matrix[u_x-1, u_y] <  new_dist or (distance_matrix[u_x-1, u_y] == new_dist and
            poids_matrice[u_x, u_y]>0 and poids_matrice[u_x-1, u_y]>0):
                new_dist = distance_matrix[u_x-1, u_y]
                n_x, n_y = u_x-1, u_y
        if u_x != cols-1:
            if distance_matrix[u_x+1, u_y] <  new_dist or (distance_matrix[u_x+1, u_y] == new_dist and
            poids_matrice[u_x, u_y]>0 and poids_matrice[u_x+1, u_y]<0):
                    new_dist = distance_matrix[u_x+1, u_y]
                    n_x, n_y = u_x+1, u_y
        dist = new_dist
        list_crossed_tiles.append((n_x, n_y))
        u_x, u_y = n_x, n_y
    return list_crossed_tiles

def ConnectedComponentWarpDistance(num_labels, labels_im, poids_matrice, r_position, passage_positions, max_coeff):
    total_weight = 0
    max_weight = -1
    connected = True
    initial_label = labels_im[r_position[0], r_position[1]]
    rows, cols = np.shape(poids_matrice)

    distance_matrix = WarpDistanceToRegion(labels_im, poids_matrice, passage_positions, r_position, initial_label)
    paths_to_labels = [[] for i in range(num_labels)]
    min_distance_paths_to_labels = np.full(num_labels, 2**30)
    min_distance_paths_to_labels[initial_label] = 0
    closest_cell_per_label = [(-1,-1)for r in range(num_labels)]
    closest_cell_per_label[initial_label] = (r_position[0], r_position[1])

    for x in range(rows):
        for y in range(cols):
            label = labels_im[x,y]
            if distance_matrix[x,y] < min_distance_paths_to_labels[label]:
                min_distance_paths_to_labels[label] = distance_matrix[x,y]
                closest_cell_per_label[label] = x, y

    #We thus exclude 0, which is the label of all the "other" cells
    for label in range(1, num_labels):
        x,y = closest_cell_per_label[label]
        if min_distance_paths_to_labels[label]<2**30:
            paths_to_labels[label] = find_back_path(distance_matrix, poids_matrice, x, y)

        distance = min_distance_paths_to_labels[label]

        if distance < 2**30:
            if distance >0:
                total_weight += distance
            if distance > max_weight:
                max_weight = distance
        else:
            connected = False

    used_path_cells = []
    for label in range(num_labels):
        for x,y in paths_to_labels[label]:
            used_path_cells.append((x,y))

    tab_coeffs = np.ones(num_labels)
    if max_weight>0:
        for label in range(1, num_labels):
            if min_distance_paths_to_labels[label]<2**30:
                tab_coeffs[label] = max_coeff * ((min_distance_paths_to_labels[label]**3)/(max_weight**3))
    return tab_coeffs, total_weight, connected, used_path_cells

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
            distances_from_robots[ux, uy-1] = min(dist_u, distances_from_robots[ux, uy-1])
    if uy != cols-1:
        if poids_matrice[ux, uy+1] < 0:
            alternative_path = dist_u - poids_matrice[ux, uy+1]
            if alternative_path < distances_from_robots[ux, uy+1]:
                distances_from_robots[ux, uy+1] = alternative_path
        else:
            distances_from_robots[ux, uy+1] = min(dist_u, distances_from_robots[ux, uy+1])
    if ux != 0:
        if poids_matrice[ux-1, uy] <0:
            alternative_path = dist_u - poids_matrice[ux-1, uy]
            if alternative_path < distances_from_robots[ux-1, uy]:
                distances_from_robots[ux-1, uy] = alternative_path
        else:
            distances_from_robots[ux-1, uy] = min(dist_u, distances_from_robots[ux-1, uy])
    if ux != rows-1:
        if poids_matrice[ux+1, uy] <0:
            alternative_path = dist_u - poids_matrice[ux+1, uy]
            if alternative_path < distances_from_robots[ux+1, uy]:
                distances_from_robots[ux+1, uy] = alternative_path
        else:
            distances_from_robots[ux+1, uy] = min(dist_u, distances_from_robots[ux+1, uy])


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
                distances_from_robot[x,y]=0
                exploration_neighbours(rows, cols, poids_matrice, distances_from_robot, 0, x, y)

    distances_from_robot[rx, ry] = 0
    
    (ux, uy), dist_u = min_unvisited_from_dict(distances_from_robot, unvisited_set)

    while not ux==-1:
        unvisited_set.pop((ux,uy))
        #We thus enumerate our (possible) neighbours
        exploration_neighbours(rows, cols, poids_matrice, distances_from_robot, dist_u, ux, uy)

        (ux, uy), dist_u = min_unvisited_from_dict(distances_from_robot, unvisited_set)

    return distances_from_robot

def printing_metrics(MetricMatrix):
    droneNo, rows, cols = np.shape(MetricMatrix)
    for x in range(rows):
        print(x, ": [", end="")
        for y in range(cols):
            print(y, "(", end="")
            for r in range(droneNo):
                print(MetricMatrix[r, x, y], end=" ")
            print(")")
        print("]\n")

class DARP:
    def __init__(self, nx, ny, notEqualPortions, given_initial_positions, given_portions, obstacles_positions,
                 visualization, MaxIter=80000, CCvariation=0.01,
                 randomLevel=0.00001, dcells=2,
                 importance=False, poids = [], tps_affichage = 0.05, given_passage = [], reduction_step_power = 8,
                 scale_down = False, output="dump_results.txt", filename =  "no file specified"):
                 #given_passage corresponds to the list of tiles that you don't need to explore, but can pass throgh.
                 #  they can be seen as "semi-obstacles"

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
        self.used_passages = [[] for r in range(self.droneNo)]
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
        self.scale_down = scale_down
        self.output = output
        self.output_string = filename
    

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
        self.min_priorities = np.ones((self.rows, self.cols))
        self.poids_matrice = np.full((self.rows, self.cols), 1)
        self.defineGridEnv()
   
        self.connectivity = np.zeros((self.droneNo, self.rows, self.cols))
        self.BinaryRobotRegions = np.zeros((self.droneNo, self.rows, self.cols), dtype=bool)

        self.AllDistances, self.termThr, dcells_weight, self.Notiles, self.DesireableAssign, self.TilesImportance, self.MinimumImportance, self.MaximumImportance, self.effectiveSize= self.construct_Assignment_Matrix()
        self.MetricMatrix = copy.deepcopy(self.AllDistances)
        self.ArrayOfElements = np.zeros(self.droneNo)
        self.color = []
        self.blinking = np.full((10, self.rows, self.cols), False)

        if self.dcells ==2: #that is, the default value
            self.dcells = dcells_weight
            #There's the slight issue if you want dcells to be EXACTLY 2 despiste weights, but well, you can always put 2.01 then

        print("The minimum deviation is", self.termThr, "and it will go up to", self.dcells)

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
        total_iteration = 0

        #as it is supposed to be defined out of the while loop for the last return
        iteration=0

        while self.termThr <= self.dcells and not success and not cancelled:
            downThres = (self.Notiles - self.termThr*(self.droneNo-1))/(self.Notiles*self.droneNo)
            upperThres = (self.Notiles + self.termThr)/(self.Notiles*self.droneNo)

            success = True

            # Main optimization loop

            iteration=0

            while iteration <= self.MaxIter and not cancelled:
                self.A, self.ArrayOfElements, self.min_priorities, self.blinking[total_iteration % 10]= assign(self.droneNo,
                                                                   self.rows,
                                                                   self.cols,
                                                                   self.GridEnv,
                                                                   self.MetricMatrix,
                                                                   self.A,
                                                                   self.poids_matrice)
                #here however we only look at the droneNo "true" drones, as we have no connectivity constraint on the false one
                ConnectedMultiplierList = np.ones((self.droneNo, self.rows, self.cols))
                ConnectedRobotRegions = np.zeros(self.droneNo)
                plainErrors = np.zeros((self.droneNo))
                #same as plainErrors, but reduced by the eventual allowed threshold
                divFairError = np.zeros((self.droneNo))
                div_updated_error = np.zeros((self.droneNo))

                for r in range(self.droneNo):
                    derivation_coeff = decreasing_factor_for_gradient(self.min_priorities, self.MetricMatrix[r],self.effectiveSize, self.poids_matrice)

                    ConnectedMultiplier = np.ones((self.rows, self.cols))
                    ConnectedRobotRegions[r] = True
                    self.update_connectivity()
                    image = np.uint8(self.connectivity[r, :, :])
                    added_weight=0
                    num_labels, labels_im = cv2.connectedComponents(image, connectivity=4)
                    if num_labels > 2:
                        BinaryRobot, BinaryNonRobot = constructBinaryImages(labels_im, self.initial_positions[r], self.rows, self.cols)
                        ConnectedMultiplier, added_weight, connected, used_path_cells = self.CalcConnectedMultiplier(self.rows, self.cols,
                                                                      self.NormalizedEuclideanDistanceBinary(True, BinaryRobot, BinaryNonRobot),
                                                                      self.NormalizedEuclideanDistanceBinary(False, BinaryRobot, BinaryNonRobot),self.CCvariation,
                                                                      num_labels, labels_im, r, derivation_coeff)
                        ConnectedRobotRegions[r] = connected
                        self.used_passages[r] = used_path_cells
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
                            correctionMult[r] = 1 + (plainErrors[r]/totalNegPlainErrors)*(TotalNegPerc/2)*self.current_reduction*derivation_coeff
                        else:
                            correctionMult[r] = 1 - (plainErrors[r]/totalNegPlainErrors)*(TotalNegPerc/2)*self.current_reduction*derivation_coeff

                        criterionMatrix = self.calculateCriterionMatrix(
                                self.TilesImportance[r],
                                self.MinimumImportance[r],
                                self.MaximumImportance[r],
                                correctionMult[r],
                                divFairError[r] < 0)

                    #the random matrix only shifts things by a small difference to 1 (per default <= e-4)
                    self.MetricMatrix[r] = self.FinalUpdateOnMetricMatrix(
                            criterionMatrix,
                            self.generateRandomMatrix(r),
                            self.MetricMatrix[r],
                            ConnectedMultiplierList[r, :, :])
                #loop to keep values in check : we have no need for values spanning from 1e-50 to 1e+50
                if self.scale_down and total_iteration % 30 == 0:
                    for x in range(self.rows):
                        for y in range(self.cols):
                            for r in range(self.droneNo):
                                self.MetricMatrix[r,x,y] = np.power(self.MetricMatrix[r,x,y], 0.95)

                total_iteration +=1
                iteration += 1
                if total_iteration%30==0:
                    printing_metrics(self.MetricMatrix)
                #we reduce the size of our steps every so often
                if total_iteration %100 ==0:
                #    print(ConnectedMultiplierList)
                    print(self.A)
                #    self.current_reduction *= self.reduction_step
                #    print("current weakening :", self.current_reduction)
                if self.visualization:
                    self.assignment_matrix_visualization.placeCells(self.A, iteration_number=iteration)
                    time.sleep(self.tps_affichage)

            if iteration >= self.MaxIter:
                self.MaxIter = self.MaxIter/2
                success = False
                self.termThr += 1

        print("exiting with a deviation of", self.termThr, "after", total_iteration, "iterations")
        self.getBinaryRobotRegions()
        return success, total_iteration

    def getBinaryRobotRegions(self):
        ind = np.where(self.A < self.droneNo)
        temp = (self.A[ind].astype(int),)+ind
        temp_r, temp_x, temp_y = temp[0].tolist(), temp[1].tolist(), temp[2].tolist()
        for r in range(self.droneNo):
            for x,y in self.used_passages[r]:
                temp_r.append(r)
                temp_x.append(x)
                temp_y.append(y)
        temp_with_rivers = np.asarray(temp_r), np.asarray(temp_x), np.asarray(temp_y)
        self.BinaryRobotRegions[temp_with_rivers] = True

    def generateRandomMatrix(self, r):
        RandomMatrix = np.zeros((self.rows, self.cols))
        RandomMatrix = 2*self.randomLevel*np.random.uniform(0, 1,size=RandomMatrix.shape) + (1 - self.randomLevel)
        for x in range(self.rows):
            for y in range(self.cols):
                if blinking_frequency(self.blinking, x, y) > 4 and np.random.uniform(0, 1) < BLINKING_THRES:
                    RandomMatrix[x,y] = 0.5
                    print("blinking in", x, y)
        return RandomMatrix

    def FinalUpdateOnMetricMatrix(self, CM, RM, currentOne, CC):
        MMnew = np.zeros((self.rows, self.cols))
        MMnew = currentOne*CM*RM*CC

        return MMnew

    def IsThisAGoalState(self, thresh, connectedRobotRegions):
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
            dcells = max(2, max_weight + (total_passage_weight // self.droneNo))
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

        return AllDistances, termThr, dcells, Notiles, DesireableAssign, TilesImportance, MinimumImportance, MaximumImportance, effectiveSize

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

    def CalcConnectedMultiplier(self, rows, cols, dist1, dist2, CCvariation, num_labels, labels_im, r, coeff_derivation):
        returnM = np.zeros((rows, cols))
        MaxV = 0
        MinV = 2**30

        for i in range(rows):
            for j in range(cols):
                a, b = dist1[i, j], dist2[i, j]
                returnM[i, j] = connectivity_function(a,b)
                if MaxV < returnM[i, j]:
                    MaxV = returnM[i, j]
                if MinV > returnM[i, j]:
                    MinV = returnM[i, j]

        coeffs_labels, total_weight, connected, used_path_cells = ConnectedComponentWarpDistance(num_labels, labels_im, self.poids_matrice, 
                                                    self.initial_positions[r], self.passage_positions, 1)

        #we split proportionnally the differences between 1-CCvariation and 1+CCvariation
        for i in range(rows):
            for j in range(cols):
                #in the case of a disconnected component
                connected_incentive = (returnM[i, j]-MinV)*((2*CCvariation)/(MaxV - MinV)) - CCvariation
                if connected_incentive>0:
                    connected_incentive*= coeffs_labels[labels_im[i,j]]*coeff_derivation
                returnM[i, j] = 1+connected_incentive

        return returnM, total_weight, connected, used_path_cells

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

