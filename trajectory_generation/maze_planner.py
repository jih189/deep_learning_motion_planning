from ompl import base as ob
from ompl import geometric as og

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