import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

import enum

import cv2

class DatasetType(enum.Enum):
    TRAIN = 1
    VALID = 2
    TEST = 3

class MazeDataset(Dataset):
    '''
    Custom dataset class for the maze dataset
    Each sample in the dataset is a maze image with a corresponding solution
    '''
    def __init__(
        self, 
        data_dir: str, 
        dataset_type: DatasetType
        ):
        self.data_dir = data_dir
        self.dataset_type = dataset_type

        # get the sample names
        self.sample_names = os.listdir(data_dir)
        maze_images_dir = os.path.join(data_dir, "images")
        solution_dir = os.path.join(data_dir, "images_result")

        # get the maze images and solutions
        self.maze_image_names = os.listdir(maze_images_dir)
        self.solution_names = os.listdir(solution_dir)
    
        self.data_block_size = 10

        # check if each maze image has a corresponding solution
        assert len(self.maze_image_names) == len(self.solution_names), "Number of maze images and solutions do not match"
    
    def __len__(self):
        return len(self.maze_image_names)
    
    def __getitem__(self, idx):
        '''
        Get the maze image and solution at the given index
        '''
        maze_image_name = self.maze_image_names[idx]
        solution_name = self.solution_names[idx]

        maze_image_path = os.path.join(self.data_dir, "images", maze_image_name)
        solution_path = os.path.join(self.data_dir, "images_result", solution_name)

        # maze images are png files
        maze_image = cv2.imread(maze_image_path, cv2.IMREAD_GRAYSCALE)

        # scale the image to 100x100
        maze_image = cv2.resize(maze_image, (100, 100))
        maze_image = maze_image / 255.0
        maze_image = torch.tensor(maze_image, dtype=torch.float32)

        # solutions are a list of points separated by ---
        solutions = []
        with open(solution_path, "r") as f:
            input_data = f.read().strip().split("\n")
            current_solution = []
            for line in input_data:
                if line == "---":
                    solutions.append(current_solution)
                    current_solution = []
                else:
                    # convert the line to a list of integers
                    point = list(map(float, line.split()))
                    current_solution.append(point)

        start_goal_block = []
        solution_block = []

        for _ in range(self.data_block_size):
            # randomly select on solution from solutions list
            solution = solutions[np.random.randint(len(solutions))]

            # get the start and goal points from this solution
            start_point = solution[0]
            goal_point = solution[-1]

            # randomly select a point from the solution
            solution_point = solution[np.random.randint(len(solution))]

            # add the data to the data block
            # concatenate the start and goal points
            start_goal_point = start_point + goal_point
            start_goal_block.append(start_goal_point)
            solution_block.append(solution_point)

        start_goal_block = np.array(start_goal_block)
        solution_block = np.array(solution_block)

        # convert the data block to a tensor
        start_goal_block = torch.tensor(start_goal_block, dtype=torch.float32)
        solution_block = torch.tensor(solution_block, dtype=torch.float32)

        maze_image = maze_image.repeat(start_goal_block.shape[0], 1, 1)

        return maze_image, start_goal_block, solution_block

# in the main function, we test the data loader
if __name__ == "__main__":
    # data_dir = Path("data")
    # dataset = MazeDataset(data_dir, DatasetType.TRAIN)
    # print(f"Number of samples in the dataset: {len(dataset)}")
    # print(f"Sample data: {dataset[0]}")

    print("Testing data loader...")
    dataset_dir = "/root/deep_learning_motion_planning/rectangular_mazes"
    dataset = MazeDataset(dataset_dir, DatasetType.TRAIN)
    print(f"Number of samples in the dataset: {len(dataset)}")
    # print(f"Sample data: {dataset[0]}")
    dataset[0]
    print("Data loader test passed!")
