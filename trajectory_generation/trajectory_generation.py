import os
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import numpy as np
import cv2
import tqdm
from maze_planner import MazeStateValidityChecker

# only print error messages
ou.setLogLevel(ou.LOG_ERROR)

def draw_maze(maze_img, traj_set):
    '''
    Draw the trajectory on the maze image.
    input:
        maze_img: a 2d numpy array, which represents the maze. 0 means the obstacle, 255 means the free space.
        traj_set: a list of trajectories. Each trajectory is a 2d numpy array.
    output:
        result_img: a 2d numpy array, which represents the maze with trajectories. 0 means the obstacle, 255 means the free space, 127 means the trajectory.
    '''
    result_img = maze_img.copy()
    for traj in traj_set:
        for i in range(traj.shape[0]):
            result_img[int(traj[i][0])][int(traj[i][1])] = 127
    return result_img

def save_traj_to_file(traj_set, file_name):
    '''
    Save the trajectory to a file.
    input:
        traj_set: a list of trajectories. Each trajectory is a 2d numpy array.
        file_name: the name of the file to save the trajectories.
    output:
        None
    '''
    with open(file_name, 'w') as f:
        for traj in traj_set:
            for i in range(traj.shape[0]):
                f.write(str(traj[i][0]) + ' ' + str(traj[i][1]) + '\n')
            f.write('---\n')

def generate_solution_for_on_maze(maze_img, number_of_traj=1, planning_time=0.3):
    '''
    Generate a solution for the maze.
    input:
        maze_img: a 2d numpy array, which represents the maze. 0 means the obstacle, 255 means the free space.
        number_of_traj: the number of trajectories to generate
    output:
        traj_set: a list of trajectories. Each trajectory is a 2d numpy array.
    '''

    # get size of img
    height, width = maze_img.shape

    # create an 2d state space
    space = ob.RealVectorStateSpace(2)

    # set bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, 0)
    bounds.setHigh(0, width)
    bounds.setLow(1, 0)
    bounds.setHigh(1, height)
    space.setBounds(bounds)

    # create a simple setup object
    ss = og.SimpleSetup(space)

    # set state validity checker
    stateValidityChecker = MazeStateValidityChecker(ss.getSpaceInformation())
    stateValidityChecker.setMaze(maze_img)
    ss.setStateValidityChecker(stateValidityChecker)

    # use rrt* planner
    planner = og.RRTstar(ss.getSpaceInformation())
    ss.setPlanner(planner)

    solution_traj = []
    for i in range(number_of_traj):

        # set start and goal
        start = ob.State(space)
        start.random()
        # check if the start is valid for 100 times, if not, then quit
        for j in range(100):
            if stateValidityChecker.isValid(start):
                break
            start.random()
        if not stateValidityChecker.isValid(start):
            print("Start is not valid")
            return

        goal = ob.State(space)
        goal.random()
        # check if the goal is valid for 100 times, if not, then quit
        for j in range(100):
            if stateValidityChecker.isValid(goal):
                break
            goal.random()
        if not stateValidityChecker.isValid(goal):
            print("Goal is not valid")
            return

        # set the start and goal
        ss.setStartAndGoalStates(start, goal)

        # try to solve the problem
        solved = ss.solve(planning_time)

        if solved:
            # print("Found solution")
            # get the solution path
            path = ss.getSolutionPath()
            path.interpolate(1000)
            traj = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                traj.append([state[0], state[1]])
            traj = np.array(traj)
            solution_traj.append(traj)
        # else:
        #     print("No solution found")
        ss.clear()

    return solution_traj
    
# main function
if __name__ == "__main__":

    ############################## Parameters #################################
    # maze_path = '../maze/src/datasets/rectangular_mazes_1709525567' # rectanglar maze
    maze_path = '../maze' # random maze
    number_of_traj_for_each_maze = 10
    time_limit_for_each_start_goal_pair = 0.3
    ###########################################################################
    maze_images_path = maze_path + '/images/'
    maze_solution_path = maze_path + '/images_result/'
    maze_img_path_set = []

    # check if maze_path exists
    if not os.path.exists(maze_path):
        print("The dataset path does not exist")
        exit()
    
    # print all files' name in the directory
    for root, dirs, files in os.walk(maze_images_path):
        for file in files:
            maze_img_path_set.append(os.path.join(root, file))

    # create the result directory. If it exists, then remove it and create a new one
    if os.path.exists(maze_solution_path):
        import shutil
        shutil.rmtree(maze_solution_path)
    os.makedirs(maze_solution_path)

    # load the image
    for img_path in tqdm.tqdm(maze_img_path_set):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # generate the solution
        result_traj = generate_solution_for_on_maze(img, number_of_traj_for_each_maze, time_limit_for_each_start_goal_pair)

        # draw the result
        result_img = draw_maze(img, result_traj)

        # get the name of the file
        file_name = img_path.split('/')[-1]

        # get name only without extension
        file_name_without_extension = file_name.split('.')[0]

        save_traj_to_file(result_traj, maze_solution_path + file_name_without_extension + '.txt')

        # save the result
        cv2.imwrite(maze_solution_path + file_name, result_img)