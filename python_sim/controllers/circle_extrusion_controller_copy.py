
import numpy as np
import numpy as np
#from math import pi
import pdb
from controllers.fillet_rule import calculate_centroid
import random

def direction_to_centroid(obs_grid):
    centroid = calculate_centroid(obs_grid, stigmergic = False)
    direction_centroid = centroid - np.floor((obs_grid.shape[0]-1)/2)*np.ones((2,))
        
    return direction_centroid
        
def check_density(obs_grid):
    '''
    Arguments:
        obs_grid : occupancy grid of the bots
    Returns:
        total_density: density of 1's and 2's combined
        density_ones: density of 1's only
        density_twos: density of 2's only
    '''
    t = obs_grid
    
    total_density = np.count_nonzero(t)/t.size
    
    t1 = (t == 1).astype(int)
    density_ones = np.count_nonzero(t1)/t1.size
    
    t2 = (t == 2).astype(int)
    density_twos = np.count_nonzero(t2)/t2.size
    
    return total_density, density_ones, density_twos

class CircleExtrusionController:
    
<<<<<<< HEAD
    def __init__(self, line_length=10, slowdown_alpha=0.5, circle_radius=25, gradient_mode='distance', seed_flag = 1, num_seed_agents = 2):
=======
    def __init__(self, line_length=10, slowdown_alpha=0.5, circle_radius=25, gradient_mode='distance',
                 min_circle_radius=5, distance_gradient_parameter=-0.1):
>>>>>>> main
        self.n_agents = None
        self.deposition = None
        self.actions_list = None
        self.prev_x = None
        self.t = 0
        self.line_length = line_length
        self.jump_direction = 1
        self.slowdown_alpha = slowdown_alpha
        self.z_vector = np.array([0,0,1])
        self.radius = circle_radius
        self.min_circle_radius = min_circle_radius
        self.circle_timer = None
        self.vector_to_disturbance = None
        self.gradient_mode = gradient_mode
<<<<<<< HEAD
        self.seed = seed_flag
        self.seed_agents = num_seed_agents
=======
        self.distance_gradient_parameter = distance_gradient_parameter
        self.print_warning = True
>>>>>>> main
        
    def update_robot_radius(self, i, x, gradient_mode, radius):
        
        if(gradient_mode == 'distance'):
            #self.robot_wise_radius[i] = self.radius
            if self.print_warning:
                print("WARNING: assuming fixed environment size of 550 px on a side. todo generalize to other env sizes.")
                self.print_warning=False

            dist_to_center = np.linalg.norm([275,275]-x)
<<<<<<< HEAD
            ##self.robot_wise_radius[i] = (1-np.min([(dist_to_center/300),0.9]))*self.init_radius
            self.robot_wise_radius[i] = radius - (dist_to_center/250)*(radius-15)
            
=======
            self.robot_wise_radius[i] = min(max(self.min_circle_radius, self.radius + self.distance_gradient_parameter*dist_to_center), self.radius)

>>>>>>> main
        elif(gradient_mode == 'time'):
            if(self.robot_timer_counter[i] > 0):
                self.robot_wise_radius[i] = self.radius - int(self.robot_timer_counter[i]/400) #change this value to figit with it
                if(self.robot_wise_radius[i] <= self.min_circle_radius):
                    self.robot_wise_radius[i] = int(self.min_circle_radius)

                        
        elif(gradient_mode == 'circle_counter'):
            if(self.count_circles[i] > 0):
<<<<<<< HEAD
                if(self.robot_wise_radius[i] <= self.radius*0.2):
                    self.robot_wise_radius[i] = int(self.radius*0.2)
                else:
                    self.robot_wise_radius[i] = self.radius - int(self.count_circles[i]/5) #change this value to figit
                    #pdb.set_trace()
            
    def calculate_radius(self, i, x, direction_centroid_norm):
        present_position = x
        present_position = np.append(present_position, 0)
        centre = present_position + self.robot_wise_radius[i] * (-direction_centroid_norm)
        self.circle_centre[i] = centre[0:2]
        
        #calculate how many steps the circle should take
        n_circle_steps = np.floor((2*np.pi*self.robot_wise_radius[i])/self.slowdown_alpha)
        self.circle_timer[i] = n_circle_steps
=======
                self.robot_wise_radius[i] = self.radius - int(self.count_circles[i]/5) #change this value to figit
                if(self.robot_wise_radius[i] <= self.min_circle_radius):
                    self.robot_wise_radius[i] = int(self.min_circle_radius)

>>>>>>> main
            
    def seek_material_func(self, i, x, total_density, direction_centroid):

        direction_centroid_norm = direction_centroid/np.linalg.norm(direction_centroid)
        if self.circle_start_delay_timer[i]==0:
            if ((total_density > 0) and (total_density <= 0.5)):
                self.actions_list[i] = (direction_centroid_norm[0:2])
                self.deposition_action[i] = 0

            elif ((total_density > 0.5) and (total_density <= 0.75)):
                self.actions_list[i] = (-direction_centroid_norm[0:2])
                self.deposition_action[i] = 0
            else:
                self.actions_list[i] = ((np.random.uniform(low = -1, high = 1, size = (2,)))*10)*10
                self.deposition_action[i] = 0

            if ((total_density > 0) and (total_density <= 0.75)):
                distance_from_centroid = np.linalg.norm(direction_centroid[0:2])
                if (distance_from_centroid <= 4):       #can tweek distance_from_centroid to make the bot extrude circle closer or farther from detected material
                    self.actions_list[i] = direction_centroid_norm[0:2]*3
                    self.circle_start_delay_timer[i] = 1

        elif self.circle_start_delay_timer[i] <= 10:
            self.actions_list[i] = np.array([0,0])
            self.deposition_action[i] = 1
            self.circle_start_delay_timer[i] += 1

        else:
<<<<<<< HEAD
            self.actions_list[i] = ((np.random.uniform(low = -1, high = 1, size = (2,)))*10)*10
            self.deposition_action[i] = 0
            
        if ((total_density > 0) and (total_density <= 0.75)):
            distance_from_centroid = np.linalg.norm(direction_centroid[0:2])
            if (distance_from_centroid <= 4):       #can tweek distance_from_centroid to make the bot extrude circle closer or farther from detected material
                self.actions_list[i] = np.array([0,0])
                self.circle_extrusion[i] = 1
                self.seek_material[i] = 0
                    #self.flag[i] = 1
                    
                self.update_robot_radius(i, x, self.gradient_mode, self.radius)
                
                self.calculate_radius(i, x, direction_centroid_norm)
                
=======
            self.circle_start_delay_timer[i] = 0
            self.circle_extrusion[i] = 1
            self.seek_material[i] = 0
                #self.flag[i] = 1

            self.update_robot_radius(i, x, self.gradient_mode, self.radius)

            present_position = x
            present_position = np.append(present_position, 0)
            centre = present_position + self.robot_wise_radius[i] * (-direction_centroid_norm)
            self.circle_centre[i] = centre[0:2]

            #calculate how many steps the circle should take
            n_circle_steps = np.floor((2*np.pi*self.robot_wise_radius[i])/self.slowdown_alpha)
            self.circle_timer[i] = n_circle_steps

>>>>>>> main
                
    def circle_trajectory(self, i, x):
        vector_towards_centre = self.circle_centre[i] - x
        vector_towards_centre = np.append(vector_towards_centre, 0)
        movement_direction = np.cross(vector_towards_centre, self.z_vector)
        
        dist_from_centre = np.linalg.norm(vector_towards_centre)
        
        if (dist_from_centre > self.robot_wise_radius[i]):
            movement_direction = 0.9*movement_direction + 0.1*vector_towards_centre
            
        movement_norm = movement_direction/np.linalg.norm(movement_direction)
        
        self.actions_list[i] = movement_norm[0:2]*10*self.slowdown_alpha
        self.deposition_action[i] = 1
        
    def take_no_action(self, i, x):
        self.one_time = 0
        self.circle_extrusion[i] = 0
        self.seek_material[i] = 1
        self.actions_list[i] = np.array([0,0])
        self.deposition_action[i] = 0
        
        
        
    def get_action(self, obs, i):
        
        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_x = np.zeros((self.n_agents,10,2)) ##
            self.seek_material = np.ones((self.n_agents,1))  
            self.deposition_flag = np.zeros((self.n_agents,))
            self.circle_extrusion = np.zeros((self.n_agents,))
            
            self.circle_centre = np.zeros((self.n_agents,2))
            #self.circle_start = np.zeros((self.n_agents,2))
            #self.wall_follow = np.ones((self.n_agents,))
            
            self.blinders = np.zeros((self.n_agents,))
            self.random = np.zeros((self.n_agents,))
            self.previous_centroid = np.zeros((self.n_agents,2))
            self.stop = np.zeros((self.n_agents,))
            self.count_circles = np.zeros((self.n_agents,))
            self.robot_wise_radius = np.ones((self.n_agents,))*self.radius
            self.circle_timer = np.zeros((self.n_agents,))
            self.vector_to_disturbance = np.zeros((self.n_agents,2))
            
            #self.centroid_dist_history = [[],[],[]]
            
            self.jump_delay = np.zeros((self.n_agents,))
            
            self.deposition_action = np.zeros((self.n_agents,))
            
            self.robot_timer_counter = np.zeros((self.n_agents,))
<<<<<<< HEAD
            self.seed_bot_list = []
            self.seed_counter = 0
            self.one_time = 0
            self.distance_from_centre = np.zeros((self.n_agents,))
            
=======

            self.circle_start_delay_timer = np.zeros((self.n_agents,))
>>>>>>> main
        self.robot_timer_counter[i] += 1
        
        x = obs['x']
                    
        obs = sensor_reading[i, :, :]
        
        direction_centroid = direction_to_centroid(obs)
        direction_centroid = np.append(direction_centroid, 0)
                  
        direction_centroid_norm = direction_centroid/np.linalg.norm(direction_centroid)
        
        total_density, density_ones, density_twos = check_density(obs)
        #one_time = 0
        if((self.seed == 1) and (self.one_time == 0) and (self.t == self.n_agents)):
            closest_index = self.distance_from_centre.argsort()
            self.seed_bot_list = closest_index[0:self.seed_agents]
            self.one_time += 1
            self.seed_counter = self.seed_agents
            #pdb.set_trace()
            

        if((self.seed_counter >= 0) and (self.t == self.n_agents) and (self.seed == 1)):
           if((i == self.seed_bot_list).any()):
                direction_centroid_norm = np.array([0, -1, 0])
                self.calculate_radius(i, x[i], direction_centroid_norm)
                self.seed_counter = self.seed_counter - 1
                self.circle_extrusion[i] = 1
                self.seek_material[i] = 0
                
            
        # if(self.seed_counter == 0):
        #     pdb.set_trace()
            
        
        ##seek_material: check centroid distance from robot's current position to determine if it's close enough to the wall to 
        ##start circle extrusion
        if ((self.circle_extrusion[i] == 0) and (self.stop[i] == 0) and (self.seek_material[i] == 1)):
     
            self.seek_material_func(i, x[i], total_density, direction_centroid)

                
        ##begin circle extrusion            
        if ((self.circle_extrusion[i] == 1) and (self.seek_material[i] == 0)):
       
            self.circle_trajectory(i, x[i])
            
            self.circle_timer[i] -= 1
            self.blinders[i] += 1   ##blinders to avoid jumping of bot at the start of circle
            
        ##keeping track of distance from previoud centroid to determine when to stop circle extrusion
        dist_from_prev_centroid = np.linalg.norm(self.previous_centroid[i] - direction_centroid[0:2])
        
        ##This if statement takes care of the bot repeating circles or moving further into the wall
        if ((total_density > 0.75) and (self.circle_extrusion[i] == 1)):
            self.stop[i] = 1
            self.jump_delay[i] = 50

        if self.circle_timer[i] < -2:
            self.stop[i] = 1
            self.circle_timer[i] = 0
            self.jump_delay[i] = 50
        
        ## Checking if change in centroid exceeds threshold in order to set stop flag    
        if ((dist_from_prev_centroid > (1.5/self.slowdown_alpha)) and (self.circle_extrusion[i] == 1) and (self.blinders[i] > 30)):
            self.circle_extrusion[i] = 0
            self.stop[i] = 1
            self.blinders[i] = 0
            self.vector_to_disturbance[i,:] = (self.previous_centroid[i] - direction_centroid[0:2])/dist_from_prev_centroid
            
            #pdb.set_trace()

        delay_time = 10
        ##Delaying the stopping of the bot using jump_delay so as to let the bot meet the opposite material
        if ((self.stop[i] == 1) and (self.jump_delay[i] <= delay_time)):
            self.jump_delay[i] += 1
            self.circle_trajectory(i, x[i])
            
            
        elif ((self.stop[i] == 1) and (self.jump_delay[i] <= delay_time+5)):
            self.jump_delay[i] += 1
            self.actions_list[i] = self.vector_to_disturbance[i,:]
        elif ((self.stop[i] == 1) and (self.jump_delay[i] > delay_time+5)):
            self.jump_delay[i] = 0
            self.seek_material[i] = 0
            self.stop[i] = 0
            self.actions_list[i] = (-direction_centroid[0:2])*50
            self.deposition_action[i] = 0
            
            
        ##making the bot jump big random steps for 15 steps so that the bot moves a bit farther from existing circle or wall    
        if((self.stop[i] == 0) and (self.seek_material[i] == 0) and (self.circle_extrusion[i] == 0) and (self.random[i] <= 15)):
            self.random[i] += 1
            self.actions_list[i] = ((np.random.uniform(low = -1, high = 1, size = (2,)))*10)*20
            self.deposition_action[i] = 0
        elif((self.stop[i] == 0) and (self.seek_material[i] == 0) and (self.circle_extrusion[i] == 0) and (self.random[i] > 15)):
            self.random[i] = 0
            self.actions_list[i] = np.array([0,0])
            self.deposition_action[i] = 0
            self.seek_material[i] = 1
            self.count_circles[i] += 1  ##to keep count of number of circles extruded by each bot
             
        self.previous_centroid[i] = direction_centroid[0:2]
        
        
        ##self.t incremented for debugging purposes
        if (self.t < self.n_agents):
            self.t += 1
            self.take_no_action(i, x[i])
            self.distance_from_centre[i] = np.linalg.norm([275,275]-x[i])
            
        
        # if(self.t == 990):
        #     pdb.set_trace()

        # if np.isnan(self.actions_list[i]).any():
        #     import pdb; pdb.set_trace()
        
        return self.actions_list[i], self.deposition_action[i]
    # def set_env(self, env):
    #     self.env = env
        
        
    
    
