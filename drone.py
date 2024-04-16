import random
import pygame 
import numpy
import math
import cv2
from time import sleep


drone_max_speed = 8 # ms

drone_accel = 100 # ms^2

GREEN = (0, 128, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
FOV_COLOR = pygame.Color(255, 0, 0, a = 0)

max_distance_comunication =100 
object_size = 3

def deg_to_rad (deg):
    return deg * math.pi / 180

## 
class Tree:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size_ = size 

    def draw(self, screen): 
        pygame.draw.circle(screen, GREEN, (self.x, self.y), self.size_)

    @property
    def pos(self):
        return [self.x, self.y]

    @property 
    def size(self):
        return self.size_

    @size.setter
    def size(self, size):
        self.size_ = size
    
class TreeCluster:
    def __init__(self, x, y, num_trees, radius_range):
        self.x = x
        self.y = y

        self.trees =  [Tree(random.uniform(-1, 1) * radius_range + x, random.uniform(-1, 1) * radius_range + y, random.randint(5, 15)) for _ in range(num_trees)]

class Forest: 
    ## forest is a list of clusters of trees
    def __init__(self, num_tree_clusters, mean_cluster_tree_size):
        self.tree_clusters = []
        self.num_tree_clusters = num_tree_clusters
        self.mean_cluster_tree_size = mean_cluster_tree_size

        self.clusters = [TreeCluster(random.randint(0, 800), random.randint(0, 600), random.randint(5, 15), 30) for _ in range(num_tree_clusters)]
    
    ## forest is a list of separate tree 
    def __init__(self, num_tree_clusters):
        self.trees =  [Tree(random.randint(0, 800), random.randint(0, 600), random.randint(5, 15)) for _ in range(num_tree_clusters)]

    def get_tree_position(self): 
        return [tree.pos for tree in self.trees]

    def get_tree_radius(self):
        return [tree.size for tree in self.trees]

    def draw(self, screen):
        for tree in self.trees:
            tree.draw(screen)

## The class drone only make sense inside the context of the swarn of drones
class Drone:
    def __init__(self, pos_x, pos_y, speed):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.speed_ = speed
        self.drone_size = 5
        self.orientation = random.randint(45, 135) 
        self.rotation = 1
        self.orientation_rad = deg_to_rad(self.orientation)

        self.camera_fov_deg = 90
        self.camera_fov_rad = deg_to_rad(self.camera_fov_deg)
        self.camera_dist = 50

        self.position_diff = {}

    def set_direction (self, direction):
        self.direction = direction

    def move(self, obstacles, simulation_step):

        if self.orientation > 135:
            self.rotation = -1

        if self.orientation < 45:
            self.rotation = 1


        self.orientation += 0.1 * self.rotation
        self.orientation_rad = deg_to_rad(self.orientation)

        obstacles_pos = obstacles[0]
        obstacles_radius = obstacles[1]        
        obstacles_dist = []        

        self.speed_ += drone_accel * simulation_step
        self.speed_ = min(self.speed_, drone_max_speed)

        for p, r in zip (obstacles_pos, obstacles_radius):
            dist = numpy.linalg.norm(numpy.array(p) - numpy.array([self.pos_x, self.pos_y]))
            ang = math.atan2(self.pos_y - p[1], self.pos_x - p[0])

            obstacles_dist.append(dist)

            
            if dist < r * 1.5 + self.drone_size:
                self.pos_x += self.speed_ * math.cos(ang) * simulation_step * 3

                self.pos_y += self.speed_ * math.sin(ang) * simulation_step * 0

        self.pos_y -= self.speed_ * simulation_step
        return obstacles_dist
        
    def draw(self, screen): 
        x_movemnt =   self.camera_dist * math.sin(self.camera_fov_rad / 2)
        y_movemnt =   self.camera_dist * math.cos(self.camera_fov_rad / 2)

        pygame.draw.arc(screen, FOV_COLOR, 
                        ((self.pos_x - self.camera_dist , self.pos_y - self.camera_dist), (  self.camera_dist*2, self.camera_dist*2)), 
                        (self.orientation_rad - self.camera_fov_rad / 2 ), 
                        (self.orientation_rad + self.camera_fov_rad / 2) , 2)

        pygame.draw.line(screen, FOV_COLOR, (self.pos_x, self.pos_y), (self.pos_x + self.camera_dist * math.cos(self.orientation_rad - self.camera_fov_rad / 2), 
                                                                        self.pos_y - self.camera_dist * math.sin(self.orientation_rad - self.camera_fov_rad / 2)), 2)

        pygame.draw.line(screen, FOV_COLOR, (self.pos_x, self.pos_y), (self.pos_x + self.camera_dist * math.cos(self.orientation_rad + self.camera_fov_rad / 2),
                                                                        self.pos_y - self.camera_dist * math.sin(self.orientation_rad + self.camera_fov_rad / 2)), 2)     


        pygame.draw.circle(screen, BLUE, (self.pos_x, self.pos_y), self.drone_size)

    def is_inside_fov(self, objective_x, objective_y):
        ang = math.atan2(self.pos_y - objective_y, self.pos_x - objective_x)
        if ang < self.camera_fov_rad / 2 and ang > - self.camera_fov_rad / 2:
            print("ang ", ang)
            return True
        return False

    def check_objective(self, objective_x, objective_y):
        if numpy.linalg.norm(numpy.array([self.pos_x, self.pos_y]) - numpy.array([objective_x, objective_y])) <  self.camera_dist:
            if self.is_inside_fov(objective_x, objective_y):
                return True
    @property
    def pos(self):
        return (self.pos_x, self.pos_y, )
    
    @property
    def speed(self):
        return self.speed_


class Swarn:
    def __init__(self, num_drones): 
        self.num_drones = num_drones
        self.drones  = []   
        self.drone_position = None

        self.distance_to_obstacles = {}
        self.distance_to_drones = {}
        
        for  i in range(num_drones):
            self.distance_to_obstacles[i] = []

        for i in range(num_drones):
            self.distance_to_drones[i] = []

        ## dict with distances         

    ## ! Make speed have bounds as well
    def create_drones(self, x_range, y_range, tree_pos, tree_radius):
        x_range_min = x_range[0]
        x_range_max = x_range[1]
        y_range_min = y_range[0]
        y_range_max = y_range[1]

        x = numpy.linspace(x_range_min, x_range_max, num= self.num_drones)
        index = 0

        drones_location = []
        drone_radius = 3
        for i in range(self.num_drones):
            
            available_pos = False
            x_pos = None
            y_pos = None
            x_pos = x[index]
            index = index + 1
            while not available_pos:
                y_pos = random.randint(y_range_min, y_range_max)

                in_tree =  False
                in_drone = False

                for p, r in zip (tree_pos, tree_radius): 
                    if numpy.linalg.norm(numpy.array(p) - numpy.array([x_pos, y_pos])) < r:
                        in_tree = True
                        break

                
                for p in drones_location:
                    if  numpy.linalg.norm(numpy.array(p) - numpy.array([x_pos, y_pos])) < r :
                        in_drone = True
                        break

                if not in_tree and not in_drone:
                    available_pos = True

            drone = Drone (x_pos, y_pos, 0)      
            self.drones.append(drone)
            drones_location.append(drone.pos)

    def move(self, obstacles, simulation_step):
        drone_position = []
        index = 0

        for  drone in self.drones:
            obstacle_distance = drone.move(obstacles, simulation_step)
            drone_position.append(drone.pos)
            
            self.distance_to_obstacles[index].append(obstacle_distance)


            index = index + 1

        self.drone_position = numpy.array(drone_position)

    def draw(self, screen): 
        index = 0


        for drone in self.drones:
            
            drone_pos =  self.drone_position[index, :]
            drone_dist = numpy.linalg.norm(drone_pos - self.drone_position, axis=1)
            
            indexes = numpy.where( drone_dist < max_distance_comunication) 

            for i in indexes[0]:
                pygame.draw.line(screen, YELLOW, (drone_pos[0], drone_pos[1]), (self.drone_position[i, 0], self.drone_position[i, 1]), 2)

            self.distance_to_drones[index].append(drone_dist)


            drone.draw(screen)
            index = index + 1

    def check_objective(self, objective_x, objective_y):
        for drone in self.drones:
            if drone.check_objective(objective_x, objective_y):
                return True

        return False
## 
# @brief The simulation class is the main class that will run the simulation
class Simulation: 
    def __init__(self, screen_width  : int , screen_height :  int , num_drones : int, Simulation_time :  int, num_trees :  int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_drones = num_drones
        self.simulation_time = Simulation_time

        self.forest = Forest(num_trees)


        self.swarn = Swarn(num_drones)
        self.swarn.create_drones((0, 800), (500, 600), self.forest.get_tree_position(), self.forest.get_tree_radius())

        self.current_time = 0.00
        self.create_objective ()
        self.output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (screen_width, screen_height))
   


    def create_map(self, num_obstacles):
        for i in range(num_obstacles):
            obstacle = Obstacle(random.randint(0, self.screen_width), random.randint(0, self.screen_height), random.randint(0, 100))
            self.obstacles.append(obstacle)
        
    def create_obstacle(self, pos_x, pos_y, size):
        obstacle = Obstacle(pos_x, pos_y, size)
        self.obstacles.append(obstacle)

    def create_objective(self): 
        self.objective_x = 300
        self.objective_y = 300

    def simulation_step(self, screen):
        self.current_time += 0.1

        self.swarn.move((self.forest.get_tree_position(), self.forest.get_tree_radius()), 0.01)
        self.forest.draw(screen)
        self.swarn.draw(screen)
        
        pygame.draw.line(screen, ORANGE, (self.objective_x + 10 * math.sin(math.pi / 4), 
            self.objective_y + 10 * math.cos(math.pi / 4)), (self.objective_x - 10 * math.sin(math.pi / 4), 
            self.objective_y - 10 * math.cos(math.pi / 4)), 
            2)

        pygame.draw.line(screen, ORANGE, (self.objective_x + 10 * math.sin(math.pi / 4), 
            self.objective_y - 10 * math.cos(math.pi / 4)), 
            (self.objective_x - 10 * math.sin(math.pi / 4), 
            self.objective_y + 10 * math.cos(math.pi / 4)), 
            2)

        surf = pygame.display.get_surface()

        # Convert Pygame surface to numpy array
        frame = pygame.surfarray.array3d(surf)
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        self.output_video.write(pygame.surfarray.array3d(screen))

    def plot (self, index):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axs = plt.subplots(2, 1)
        fig.suptitle('Drone distance to obstacles')

        obstacles = self.swarn.distance_to_obstacles[index]
        obstacles = np.array(obstacles)
        obstacles = np.min(obstacles, axis=1) 

        axs[0].plot(obstacles)
        axs[0].set_title('Drone distance to obstacles')
        axs[1].plot(self.swarn.distance_to_drones[index])
        axs[1].set_title('Drone distance to drones')

        plt.show()

    @property
    def simulation_end(self) -> bool:

        if self.swarn.check_objective(self.objective_x, self.objective_y):
            return True
        
        return self.current_time >= self.simulation_time

        


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    simul = Simulation(800, 600, 20, 1000, 100)
    sleep(2)


    while not simul.simulation_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        screen.fill((0 ,0 , 0))
        simul.simulation_step(screen)
        pygame.display.flip()

    simul.plot(3)
    simul.output_video.release()

    pygame.quit()

if __name__ == "__main__":
    main()
    