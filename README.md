# Deep learning Based Motion Planning in 2D Maze

## Description
This days, deep learning is used in many fields. Speically, motion planning becomes more and more popular with deep learning. However, there is no comparison between deep learning and traditional motion planning methods. Most of those deep learning based motion planning methods did not provide convincing evidence about why they are better. Thus, to analyze the performance of deep learning based motion planning, I am using 2D mazes. 2D mazes are simple and easy to understand. In this project, I am using deep learning to solve 2D mazes. Thus, we can have some insights about how deep learning based motion planning actually works in this field.

## Getting Started

### Installing
We have provided a docker file to install all the dependencies, such as ompl. You can use the following command to build the docker image.
```
cd [where DEEP_LEARNING_MOTION_PLANNING is located]/docker_image
sh build.sh
xhost +
sh run.sh
```
You can create a new terminal and use the following command to enter the docker container.
```
sh enter_lastest_container.sh
```
## Executing program
There are mainly two parts in this project. One is to generate 2D mazes and the other is to use deep learning to solve the mazes. 

### Generate 2D mazes
You can use the following command to generate 2D mazes.
```
cd [where DEEP_LEARNING_MOTION_PLANNING is located]/maze/src/datasets
python3.8 generate_rect_dataset.py -rows 10 -width 10 -items 10
```
The rows and width are the size of the maze. The items is the number of mazes you want to generate. After you run the code, there should be a directory called **rectangular_mazes_...** generated in the same directory. All the mazes are saved as the png files in the images directory of the **rectangular_mazes_...** directory.

### Generate solution trajectories.
In here, to train the deep learning model, we need to generate the solution trajectories for the mazes. Thus, we will use ompl to first generate the solution trajectories for the mazes. 

TODO: Add the code to generate the solution trajectories.