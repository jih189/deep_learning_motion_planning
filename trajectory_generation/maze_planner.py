from ompl import base as ob
from ompl import geometric as og
import numpy as np
import cv2

class MazeStateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si):
        super(MazeStateValidityChecker, self).__init__(si)
        self.si = si
        self.maze = None

    def isValid(self, state):
        if self.maze is None:
            # raise an exception if the maze is not set. 
            raise Exception("Maze is not set")
        
        # Check if the position is inside the maze
        if self.maze[int(state[0])][int(state[1])] == 0:
            return False
        else:
            return True

    def setMaze(self, maze):
        self.maze = maze

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

def generate_solution_for_on_maze(maze_img, number_of_traj=1):
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
        solved = ss.solve(0.3)

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
    maze_path = '../maze/src/datasets/rectangular_mazes_1709523634'
    maze_dataset_path = maze_path + '/images/'
    maze_solution_path = maze_path + '/images_result/'
    maze_img_path_set = []

    # print all files' name in the directory
    import os
    for root, dirs, files in os.walk(maze_dataset_path):
        for file in files:
            maze_img_path_set.append(os.path.join(root, file))

    # create the result directory. If it exists, then remove it and create a new one
    if os.path.exists(maze_solution_path):
        import shutil
        shutil.rmtree(maze_solution_path)
    os.makedirs(maze_solution_path)

    # load the image
    for img_path in maze_img_path_set:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # generate the solution
        result_traj = generate_solution_for_on_maze(img, 10)

        # draw the result
        result_img = draw_maze(img, result_traj)

        # get the name of the file
        file_name = img_path.split('/')[-1]

        save_traj_to_file(result_traj, maze_solution_path + file_name + '.txt')

        # save the result
        cv2.imwrite(maze_solution_path + file_name, result_img)