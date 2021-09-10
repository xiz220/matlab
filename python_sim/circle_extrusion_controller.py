import numpy as np
import numpy as np
#from math import pi
import pdb
from fillet_rule import calculate_centroid

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
    
    def __init__(self, line_length=10, slowdown_alpha=0.5, circle_radius=25):
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
        self.init_radius = circle_radius
        self.circle_timer = None
        self.vector_to_disturbance = None
        
    def update_robot_radius(self, i, x):
        #self.robot_wise_radius[i] = self.radius
        dist_to_center = np.linalg.norm([275,275]-x)
        self.robot_wise_radius[i] = (1-np.min([(dist_to_center/300),0.9]))*self.init_radius
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
            
        
        x = obs['x']
                    
        obs = sensor_reading[i, :, :]
        
        direction_centroid = direction_to_centroid(obs)
        direction_centroid = np.append(direction_centroid, 0)
                  
        direction_parallel_wall = np.cross(direction_centroid, self.z_vector)
        
        direction_centroid_norm = direction_centroid/np.linalg.norm(direction_centroid)
        
        
        total_density, density_ones, density_twos = check_density(obs)
        
        ##self.t incremented for debugging purposes
        if (i == (self.n_agents - 1)):
            self.t += 1
        
        ##seek_material: check centroid distance from robot's current position to determine if it's close enough to the wall to 
        ##start circle extrusion
        if ((self.circle_extrusion[i] == 0) and (self.stop[i] == 0) and (self.seek_material[i] == 1)):
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
                    self.actions_list[i] = np.array([0,0])
                    self.circle_extrusion[i] = 1
                    self.seek_material[i] = 0
                    self.update_robot_radius(i, x[i])

                    # find circle center
                    present_position = x[i]
                    present_position = np.append(present_position, 0)
                    centre = present_position + self.robot_wise_radius[i] * (-direction_centroid_norm)
                    self.circle_centre[i] = centre[0:2]

                    #calculate how many steps the circle should take
                    n_circle_steps = np.floor((2*np.pi*self.robot_wise_radius[i]))
                    self.circle_timer[i] = n_circle_steps
                    #import pdb; pdb.set_trace()
                    #pdb.set_trace()

                
        ##begin circle extrusion            
        if ((self.circle_extrusion[i] == 1) and (self.seek_material[i] == 0)):



            vector_towards_centre = self.circle_centre[i] - x[i]
            vector_towards_centre = np.append(vector_towards_centre, 0)
            movement_direction = np.cross(vector_towards_centre, self.z_vector)
            
            dist_from_centre = np.linalg.norm(vector_towards_centre)
            
            if (dist_from_centre > self.robot_wise_radius[i]):
                movement_direction = 0.9*movement_direction + 0.1*vector_towards_centre
                
            movement_norm = movement_direction/np.linalg.norm(movement_direction)
            self.circle_timer[i] -= 1
            self.actions_list[i] = movement_norm[0:2]*10
            self.deposition_action[i] = 1
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
        if ((dist_from_prev_centroid > 1.5) and (self.circle_extrusion[i] == 1) and (self.blinders[i] > 30)):
            self.circle_extrusion[i] = 0
            self.stop[i] = 1
            self.blinders[i] = 0
            self.vector_to_disturbance[i,:] = (self.previous_centroid[i] - direction_centroid[0:2])/dist_from_prev_centroid
            #pdb.set_trace()

        delay_time = 10
        ##Delaying the stopping of the bot using jump_delay so as to let the bot meet the opposite material
        if ((self.stop[i] == 1) and (self.jump_delay[i] <= delay_time)):
            self.jump_delay[i] += 1
            vector_towards_centre = self.circle_centre[i] - x[i]
            vector_towards_centre = np.append(vector_towards_centre, 0)
            movement_direction = np.cross(vector_towards_centre, self.z_vector)
            
            dist_from_centre = np.linalg.norm(vector_towards_centre)
            
            if (dist_from_centre > self.robot_wise_radius[i]):
                movement_direction = 0.9*movement_direction + 0.1*vector_towards_centre
                
            movement_norm = movement_direction/np.linalg.norm(movement_direction)
            self.actions_list[i] = movement_norm[0:2]*10
            
            self.deposition_action[i] = 1
        elif ((self.stop[i] == 1) and (self.jump_delay[i] <= delay_time + 5)):
            self.jump_delay[i] += 1
            self.actions_list[i] = self.vector_to_disturbance[i,:]
        elif ((self.stop[i] == 1) and (self.jump_delay[i] > delay_time + 5)):
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
        
        
        # if(self.t == 990):
        #     pdb.set_trace()

        # if np.isnan(self.actions_list[i]).any():
        #     import pdb; pdb.set_trace()
        
        return self.actions_list[i], self.deposition_action[i]
    # def set_env(self, env):
    #     self.env = env
        
        
    
    
