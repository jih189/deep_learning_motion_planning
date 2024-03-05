import os
import cv2
import numpy as np
import random
import math
import argparse

def draw_square(image, origin, size):
    top_left = (origin[0], origin[1])
    bottom_right = (origin[0] + size, origin[1] + size)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)

def draw_equilateral_triangle(image, center, size):
    height = size * math.sqrt(3) / 2
    vertices = np.array([
        [[center[0], int(center[1] - 2 * height / 3)]],
        [[int(center[0] - size / 2), int(center[1] + height / 3)]],
        [[int(center[0] + size / 2), int(center[1] + height / 3)]]
    ], np.int32)
    cv2.fillPoly(image, [vertices], (0, 0, 0))

def draw_circle(image, center, diameter):
    cv2.circle(image, center, diameter // 2, (0, 0, 0), -1)

# main function
if __name__ == "__main__":

    # get the image parameters from user input with argparse
    parser = argparse.ArgumentParser(description='Create a random obstacle image')
    parser.add_argument('--image_width', type=int, default=512, help='Width of the image')
    parser.add_argument('--image_height', type=int, default=512, help='Height of the image')
    parser.add_argument('--number_of_obstacles', type=int, default=40, help='Number of obstacles')
    parser.add_argument('--number_of_images', type=int, default=10, help='Number of images to create')

    args = parser.parse_args()

    # Image parameters
    image_width = args.image_width
    image_height = args.image_height
    obstacle_size = min(image_width, image_height) // 10
    number_of_obstacles = args.number_of_obstacles
    number_of_images = args.number_of_images

    output_folder = 'images'

    # if the output folder does exist, delete it
    if os.path.exists(output_folder):
        import shutil
        shutil.rmtree(output_folder)
    
    # create the output folder
    os.makedirs(output_folder)

    for i in range(number_of_images):
        # Create a white canvas
        canvas = np.ones((image_width, image_height, 3), dtype=np.uint8) * 255

        for _ in range(number_of_obstacles):
            # random select a point in in the image
            center = (random.randint(0, image_width - 1), random.randint(0, image_height - 1))

            size = random.randint(int(obstacle_size * 0.5), int(obstacle_size * 1.5))

            shape_type = random.choice(['square', 'triangle', 'circle'])

            if shape_type == 'square':
                origin = (center[0] - size // 2, center[1] - size // 2)
                draw_square(canvas, origin, size)
            elif shape_type == 'triangle':
                draw_equilateral_triangle(canvas, center, size)
            else:  # Circle
                draw_circle(canvas, center, size)

        # Save the image
        image_path = os.path.join(output_folder, f'{i}.png')
        cv2.imwrite(image_path, canvas)
