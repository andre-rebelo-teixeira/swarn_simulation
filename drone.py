import random
import pygame 
import numpy
import math


GREEN = (0, 128, 0)
RED = (255, 0, 0)

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
    def __init__(self, pos_x, pos_y,  speed_x, speed_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.direction = random.choice([-1, 1])

    def set_direction (self, direction):
        self.direction = direction

    def move(self, obstacles, simulation_step):
        obstacles_pos = obstacles[0]
        obstacles_radius = obstacles[1]        

        for p, r in zip (obstacles_pos, obstacles_radius):
            dist = numpy.linalg.norm(numpy.array(p) - numpy.array([self.pos_x, self.pos_y]))
            ang = math.atan2(self.pos_y - p[1], self.pos_x - p[0])
            
            if dist < r * 1.5:
                self.pos_x += self.speed_x * math.cos(ang) * simulation_step 
                self.pos_y += self.speed_y * math.sin(ang) * simulation_step

        self.pos_y -= self.speed_y * simulation_step
        pass
        
    def draw(self, screen): 
        pygame.draw.circle(screen, RED, (self.pos_x, self.pos_y), 5)
    
    @property
    def pos(self):
        return (self.pos_x, self.pos_y, )
    
    @property
    def speed(self):
        return (self.speed_x, self.speed_y, self.speed_z)

class Swarn:
    def __init__(self, num_drones): 
        self.num_drones = num_drones
        self.drones  = []   

    ## ! Make speed have bounds as well
    def create_drones(self, x_range, y_range, tree_pos, tree_radius):
        x_range_min = x_range[0]
        x_range_max = x_range[1]
        y_range_min = y_range[0]
        y_range_max = y_range[1]

        drones_location = []
        drone_radius = 3
        for i in range(self.num_drones):
            
            available_pos = False
            x_pos = None
            y_pos = None
            while not available_pos:
                x_pos = random.randint(x_range_min, x_range_max)
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

            drone = Drone (x_pos, y_pos, random.randint(1, 10), random.randint(1, 10))      
            self.drones.append(drone)
            drones_location.append(drone.pos)

    def move(self, obstacles, simulation_step):
        for  drone in self.drones:
            drone.move(obstacles, simulation_step)

    def draw(self, screen): 
        for drone in self.drones:
            drone.draw(screen)
        
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

    def create_map(self, num_obstacles):
        for i in range(num_obstacles):
            obstacle = Obstacle(random.randint(0, self.screen_width), random.randint(0, self.screen_height), random.randint(0, 100))
            self.obstacles.append(obstacle)
        
    def create_obstacle(self, pos_x, pos_y, size):
        obstacle = Obstacle(pos_x, pos_y, size)
        self.obstacles.append(obstacle)

    def simulation_step(self, screen):
        self.current_time += 0.01

        self.swarn.move((self.forest.get_tree_position(), self.forest.get_tree_radius()), 0.01)
        self.forest.draw(screen)
        self.swarn.draw(screen)

    @property
    def simulation_end(self) -> bool:
        return self.current_time >= self.simulation_time


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    simul = Simulation(800, 600, 20, 10000, 100)

    while not simul.simulation_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((255, 255, 255))
        simul.simulation_step(screen)
        pygame.display.flip()



if __name__ == "__main__":
    main()
    