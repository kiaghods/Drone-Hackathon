from black import out
from numpy import average
from multiRobotPathPlanner import MultiRobotPathPlanner
import sys
import argparse
import os

#returns the content of the file as a list of strings, one per line
def readfile (filename):
    print(filename)
    #filename = sys.argv[0]
    file = open(filename, "r")
    strings_input = file.readlines()
    file.close()
    return strings_input

#calls DARP on the said file, with the options precised in args
def darp_call_file(args, filename):
    strings_input = readfile(filename)
    rows = len(strings_input)
    list_robots = []
    list_obstacles = []
    list_poids = []
    list_passage = []

    cols = len(strings_input[0].split())
    current_tile = 0

    for i in range(rows):
        list_tiles = strings_input[i].split()
        #check if the input file has the right format
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
                    value = float(elt)
                    if value != 1:
                        list_poids.append((current_tile, float(elt)))
                except :
                    print("Not a recognized value at position", current_tile, ": only floats, '@' and '#' accepted, '", elt, "' submitted")
                    sys.exit(3)
            current_tile+=1

    instance = MultiRobotPathPlanner( rows, cols, args.nep, list_robots,  args.portions, list_obstacles, args.vis, 
                                                list_poids, MaxIter=args.iter, tps_affichage=args.show,
                                                passage=list_passage, reduction_step_power=args.slow, scale_down=args.root)
    if instance.DARP_success:
        return instance.iterations
    else:
        return -1

#encapsulation of the darp calls to print an averaged output
def run_on_file(filename, args, output="dump.txt"):
    output_string = filename
    if args.average == 1:
        iter = darp_call_file(args, filename)
        if iter<0:
            output_string += " : inf\n"
        else:
            output_string += " : "+ str(iter)+"\n"
    else:
        total_sum = 0
        nb_infinity = 0
        for i in range(args.average):
            iter = darp_call_file(args, filename)
            if iter >=0:
                total_sum+=iter
            else:
                nb_infinity +=1
        if args.average == nb_infinity:
            output_string += " : inf\n"
        else:
            average = total_sum / (args.average - nb_infinity)
            output_string += " : "+ str(average)+ " avec "+ str(nb_infinity)+ " divergences\n"
        
    output_file = open(output, "a")  # append mode
    output_file.write(output_string)
    output_file.close()
    print(output_string)

#Main function. We parse the options, then choose the appropriate behaviour
if __name__ == '__main__':
    #We define again the "other" options, the ones that do not directly define the grid
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
        type=str,
        help='File containing the desired input')
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
    argparser.add_argument(
        '-slow',
        default=8,
        nargs = '?',
        type = int,
        help='the power of 10 by which the gradient step will reduce every 30 iterations (default: 8)')
    argparser.add_argument(
        '-root',
        default=False,
        action='store_true',
        help='Apply a reduction to a power < 1 every 30 iterations (default: False)')
    argparser.add_argument(
        '-tests',
        type=str,
        help='applies a standard iteration of tests, with output in the given file')
    argparser.add_argument(
        '-average',
        default=1,
        nargs = '?',
        type = int,
        help='runs darp X times, outputs the average of the iteration numbers (default: 1)')
    args = argparser.parse_args()

    filename = args.file
    if filename != None:
        print("we run darp on the input", filename)
        run_on_file(filename, args)
    else:
        output_file = args.tests
        if output_file != None:
            #We are running all the tests in tests_txt/
            with open(output_file,'w') as f:
                pass
            print("running tests")
            path = "./tests_txt"
            list_files = os.listdir(path)
            list_files.sort()
            for filename in list_files:
                if filename.endswith(".txt"):
                    extended_name = "tests_txt/"+filename
                    run_on_file(extended_name, args, output_file)
        else:
            print("no behaviour specified, exiting")
            sys.exit(9)
