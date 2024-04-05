import torch

from cvae import CVAE
from data_loader import MazeDataset, DatasetType

def train_cvae(
    cvae: CVAE,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device
    ):
    '''
    Train the CVAE model
    '''
    # set the model to training mode
    cvae.train()
    
    # set the optimizer
    optimizer = torch.optim.Adam(cvae.parameters(), lr=learning_rate)
    
    # loss function
    loss_fn = torch.nn.MSELoss()
    
    # train the model
    for epoch in range(num_epochs):
        for i, (maze_image, start_goal_points, x) in enumerate(train_loader):

            maze_image = maze_image.view(-1, 100, 100)
            start_goal_points = start_goal_points.view(-1, 4)
            x = x.view(-1, 2)
        
            # move the data to the device
            x = x.to(device)
            maze_image = maze_image.to(device)
            start_goal_points = start_goal_points.to(device)
            
            # forward pass
            x_hat, z_mean, z_log_var = cvae(x, maze_image, start_goal_points)

            # compute the loss
            loss = loss_fn(x_hat, x)

            # KL divergence
            kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

            # total loss
            loss += kl_divergence

            # zero the gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            # print the loss
            if i % 100 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

    print("Training complete.")

if __name__ == "__main__":

    # load a sample maze image and solution path
    dataset_dir = "/root/deep_learning_motion_planning/rectangular_mazes"
    dataset = MazeDataset(dataset_dir, DatasetType.TRAIN)

    # test cvae model
    input_dim = 2
    latent_dim = 2
    cvae = CVAE(input_dim, latent_dim)

    # move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae.to(device)

    # create a data loader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)

    # train the cvae model
    train_cvae(cvae, train_loader, num_epochs=10, learning_rate=0.001, device=device)

    # save the model
    torch.save(cvae.state_dict(), "cvae.pth")

