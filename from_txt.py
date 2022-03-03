from multiRobotPathPlanner import MultiRobotPathPlanner
import sys
import argparse

#returns the content of the file as a list of strings, one per line
def readfile (filename):
    print(filename)
    #filename = sys.argv[0]
    file = open(filename, "r")
    strings_input = file.readlines()
    file.close()
    return strings_input

if __name__ == '__main__':
    #We define again the "other" options, the ones that do not directly appear in the grid
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-nep',
        action='store_true',
        help='Not Equal Portions shared between the Robots in the Grid (default: False)')
    argparser.add_argument(
        '-portions',
        default=[0.2, 0.3, 0.5],
        nargs='*',
        type=float,
        help='Portion for each Robot in the Grid (default: (0.2, 0.7, 0.1))')
    argparser.add_argument(
        '-file',
        default="tests_txt/couloir_avec_bureaux.txt",
        nargs = '?',
        type=str,
        help='File containing the desired input (default : tests_txt/couloir_avec_bureaux.txt)')
    argparser.add_argument(
        '-vis',
        default=False,
        action='store_true',
        help='Visualize results (default: False)')
    argparser.add_argument(
        '-iter',
        default=80000,
        nargs = '?',
        type = int,
        help='maximum number of iterations (default: 80000)')
    argparser.add_argument(
        '-show',
        default=0.05,
        nargs = '?',
        type = float,
        help='time for each iteration to be shown (default: 0.05)')
    args = argparser.parse_args()
    
    strings_input = readfile(args.file)

    rows = len(strings_input)
    list_robots = []
    list_obstacles = []
    list_poids = []
    list_passage = []

    cols = len(strings_input[0].split())
    current_tile = 0

    for i in range(rows):
        list_tiles = strings_input[i].split()
        if len(list_tiles) != cols:
            print('Not the same number of items at every line, line ', i, 'has', len(list_tiles), "columns while the 0-th has", cols, "elements")
            sys.exit(1)

        #building the grid parsing through the input txt
        for elt in list_tiles:
            if elt == "@":
                list_robots.append(current_tile)
            elif elt == "#":
                list_obstacles.append(current_tile)
            elif elt[0] == "-":
                try : 
                    list_passage.append((current_tile, float(elt[1:])))
                except :
                    print("Not a recognized value at position", current_tile, ":", elt[1:], "where a float is expected after '-'")
                    sys.exit(3)
            else:
                try :
                   list_poids.append((current_tile, float(elt)))
                except :
                    print("Not a recognized value at position", current_tile, ": only floats, '@' and '#' accepted, '", elt, "' submitted")
                    sys.exit(3)
            current_tile+=1

    MultiRobotPathPlanner( rows, cols, args.nep, list_robots,  args.portions, list_obstacles, args.vis, 
                                                list_poids, MaxIter=args.iter, tps_affichage=args.show,
                                                passage=list_passage)