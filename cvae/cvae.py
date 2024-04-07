'''
The module contains the implementation of the Conditional Variational Autoencoder (CVAE) model. 
In motion planning, the CVAE model is a generative model that learns to generate sample points 
for solving a maze based on the conditional information of the maze image and start and goal points.

The maze image is 100x100 pixels and is represented as a grayscale image. The start and goal points
are represented as two points in the maze image such as (start_x, start_y) and (goal_x, goal_y). 
The solution path is a list of points that connect the start and goal points.
'''

import torch

from data_loader import DatasetType, MazeDataset

class Encoder(torch.nn.Module):
    '''
    Encoder network of the CVAE model. The encoder network takes point x on the solution path 
    and condition y (maze image and start and goal points) as input and outputs the mean and
    log variance of the latent distribution. For the conditional information, the encoder network
    it uses convolutional layers to process the maze image and concatenate start and goal points
    with the point x.
    '''
    def __init__(self, input_dim, latent_dim):
        '''
        Initialize the encoder network
        '''
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # convolutional layers for processing the maze image
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # fully connected layers for processing the start and goal points
        self.fc1 = torch.nn.Linear(4, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        
        # fully connected layers for processing the point x
        self.fc3 = torch.nn.Linear(input_dim, 64)
        
        # fully connected layers for processing the concatenated features
        # of the maze image, start and goal points, and point x
        self.fc4 = torch.nn.Linear(20000 + 64 + 64, self.latent_dim * 2)

    def forward(self, x, maze_image, start_goal_points):
        '''
        Forward pass of the encoder network.
        x: (BATCH_SIZE, input_dim) - point on the solution path
        maze_image: (BATCH_SIZE, 100, 100) - maze image
        start_goal_points: (BATCH_SIZE, 2 * input_dim) - start and goal points
        '''
        # process the maze image
        maze_image = torch.relu(self.conv1(maze_image.unsqueeze(1)))
        maze_image = self.pool(maze_image)
        maze_image = torch.relu(self.conv2(maze_image))
        maze_image = self.pool(maze_image)
        maze_image = torch.relu(self.conv3(maze_image))
        maze_image = self.pool(maze_image)
        maze_image = maze_image.view(maze_image.size(0), -1)
        # now maze image size should be (BATCH_SIZE, 20000)

        # process the start and goal points
        start_goal_points = torch.relu(self.fc1(start_goal_points))
        start_goal_points = torch.relu(self.fc2(start_goal_points))

        # process the point x
        x = torch.relu(self.fc3(x))

        # concatenate the features
        features = torch.cat([maze_image, start_goal_points, x], dim=1)

        # process the concatenated features to get the mean and log variance
        features = torch.relu(self.fc4(features))
        z_mean = features[:, :self.latent_dim]
        z_log_var = features[:, self.latent_dim:]

        return z_mean, z_log_var

class Decoder(torch.nn.Module):
    '''
    Similar to the encoder network, the decoder network takes the latent representation z and
    condition y (maze image and start and goal points) as input and outputs the reconstructed
    point x on the solution path. The decoder network uses convolutional layers to process the
    maze image and concatenate start and goal points with the latent representation z.
    '''
    def __init__(self, input_dim, latent_dim):
        '''
        Initialize the decoder network
        '''
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # convolutional layers for processing the maze image
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        # fully connected layers for processing the start and goal points
        self.fc1 = torch.nn.Linear(4, 32)
        self.fc2 = torch.nn.Linear(32, 64)

        # fully connected layers for processing the latent representation z
        self.fc3 = torch.nn.Linear(self.latent_dim, 64)

        # fully connected layers for processing the concatenated features
        # of the maze image, start and goal points, and latent representation z
        self.fc4 = torch.nn.Linear(20000 + 64 + 64, self.input_dim)

    def forward(self, z, maze_image, start_goal_points):
        '''
        Forward pass of the decoder network
        z: (BATCH_SIZE, latent_dim) - latent representation
        maze_image: (BATCH_SIZE, 100, 100) - maze image
        start_goal_points: (BATCH_SIZE, 2 * input_dim) - start and goal points
        '''
        # process the maze image
        maze_image = torch.relu(self.conv1(maze_image.unsqueeze(1)))
        maze_image = self.pool(maze_image)
        maze_image = torch.relu(self.conv2(maze_image))
        maze_image = self.pool(maze_image)
        maze_image = torch.relu(self.conv3(maze_image))
        maze_image = self.pool(maze_image)
        maze_image = maze_image.view(maze_image.size(0), -1)

        # process the start and goal points
        start_goal_points = torch.relu(self.fc1(start_goal_points))
        start_goal_points = torch.relu(self.fc2(start_goal_points))

        # process the latent representation z
        z = torch.relu(self.fc3(z))

        # concatenate the features
        features = torch.cat([maze_image, start_goal_points, z], dim=1)

        # process the concatenated features to get the reconstructed point x
        x_hat = torch.sigmoid(self.fc4(features))

        return x_hat

class CVAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        '''
        Initialize the CVAE model
        '''
        super(CVAE, self).__init__()
        
        # encoder network
        self.encoder = Encoder(input_dim, latent_dim)
        
        # decoder network
        self.decoder = Decoder(input_dim, latent_dim)
        
    def forward(self, x, maze_image, start_goal_points):
        '''
        Forward pass of the CVAE model
        '''
        # encode one point x on solution path and condition (maze image and start and goal points)
        z_mean, z_log_var = self.encoder(x, maze_image, start_goal_points)
        
        # sample from the latent distribution
        z = self.sample_latent(z_mean, z_log_var)
        
        # decode the latent representation
        x_hat = self.decoder(z, maze_image, start_goal_points)
        
        return x_hat, z_mean, z_log_var
    
    def sample_latent(self, z_mean, z_log_var):
        '''
        Sample from the latent distribution
        '''
        # sample from the normal distribution
        epsilon = torch.randn_like(z_mean)
        
        # reparameterization trick
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
        return z

# main function to test the CVAE model
if __name__ == "__main__":
    # test encoder
    # load a sample maze image and solution path
    dataset_dir = "/root/deep_learning_motion_planning/rectangular_mazes"
    dataset = MazeDataset(dataset_dir, DatasetType.TRAIN)
    maze_image, start_goal_block, solution_block = dataset[0]

    # (solution_block, maze_image, start_goal_block)

    # test cvae model
    input_dim = 2
    latent_dim = 2
    cvae = CVAE(input_dim, latent_dim)

    x_hat, z_mean, z_log_var = cvae(solution_block, maze_image, start_goal_block)

    print("Output shape:", x_hat.shape)
    print("Mean shape:", z_mean.shape)
    print("Log variance shape:", z_log_var.shape)