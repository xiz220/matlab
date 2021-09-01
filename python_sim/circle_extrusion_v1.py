import numpy as np
#from math import pi
#import pdb
from fillet_rule import calculate_centroid


class CircleExtrusion_v1:

    def __init__(self, line_length=10, slowdown_alpha=0.5, angle=45, circle_radius=25):
        self.n_agents = None
        self.deposition = None
        self.actions_list = None
        self.prev_x = None
        self.t = 0
        self.line_length = line_length
        self.jump_direction = 1
        self.angle = angle
        self.slowdown_alpha = slowdown_alpha
        self.z_vector = np.array([0,0,1])
        self.radius = circle_radius
        
    def direction_to_centroid(self, obs_grid):
        centroid = calculate_centroid(obs_grid, stigmergic = False)
        direction_centroid = centroid - np.floor((obs_grid.shape[0]-1)/2)*np.ones((2,))
        
        return direction_centroid
        
    def check_density(self, obs_grid):
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
    
        
    def get_action(self, obs):
        
        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_x = np.zeros((self.n_agents,10,2)) ##
            self.counter = np.zeros((self.n_agents,1))  
            self.deposition_flag = np.zeros((self.n_agents,))
            self.circle_extrusion = np.zeros((self.n_agents,))
            self.stop = np.zeros((self.n_agents,))
            self.circle_centre = np.zeros((self.n_agents,2))
            self.circle_start = np.zeros((self.n_agents,2))
            
            self.blinders = np.zeros((self.n_agents,))
            self.counter = np.zeros((self.n_agents,))
            
        deposition_action = []
        x = obs['x']
        
        for i in range(self.n_agents):
            
            obs = sensor_reading[i, :, :]
            
            direction_centroid = self.direction_to_centroid(obs)
            direction_centroid = np.append(direction_centroid, 0)
                    
            direction_parallel_wall = np.cross(direction_centroid, self.z_vector)
                    
            direction_centroid_norm = direction_centroid/np.linalg.norm(direction_centroid)
            
            direct_parallel_wall = direction_parallel_wall/np.linalg.norm(direction_parallel_wall)
            
            if (np.isnan(direct_parallel_wall).any()):
                direct_parallel_wall = direction_parallel_wall
            
            total_density, density_ones, density_twos = self.check_density(obs)
            
            
            if ((self.circle_extrusion[i] == 0) and (self.stop[i] == 0)):
                
                if ((total_density > 0) and (total_density < 0.45)):
                    #Bot is near some material, action is to move closer to centroid and also move parallel to the wall
                    temp_movement_vector = (1-total_density)*direction_centroid_norm[0:2] + direct_parallel_wall[0:2] 
                    self.actions_list[i] = ((temp_movement_vector/np.linalg.norm(temp_movement_vector))*self.slowdown_alpha)*10
                    deposition_action.append(0)
                    self.deposition_flag[i] = 0
                    
                    
                elif ((total_density >= 0.45) and (total_density <= 0.55)):
                    #bot is just at the wall so action is to move parallel to it
                    self.actions_list[i] = direct_parallel_wall[0:2]
                    deposition_action.append(1)
                    self.actions_list[i] = (self.actions_list[i]*self.slowdown_alpha)*10
                    self.deposition_flag[i] = 1
                    #pdb.set_trace()
                    
                    
                elif ((total_density > 0.55) and (total_density <=0.85)):
                    #bot is inside the wall, action is to move parallel to wall but also away from centroid
                    temp_movement_vector = total_density*direct_parallel_wall[0:2] + (-1+total_density)*direction_centroid_norm[0:2]
                    
                    self.actions_list[i] = ((temp_movement_vector/np.linalg.norm(temp_movement_vector))*self.slowdown_alpha)*10
                    deposition_action.append(1)
                    self.deposition_flag[i] = 1
                    
                elif ((total_density > 0.85) and (total_density < 1)):
                    #bot is almost completely inside the wall so move in the opposite direction of centroid
                    self.actions_list[i] = ((total_density*(-direction_centroid_norm[0:2])) + ((1-total_density)*direct_parallel_wall[0:2]))*10
                    deposition_action.append(0)
                    self.deposition_flag[i] = 0
                     
                else:
                    #if inside a wall completely or in an empty area then jump randomly
                    self.actions_list[i] = ((np.random.uniform(low = -1, high = 1, size = (2,)))*15)*10
                    deposition_action.append(0)
                    self.deposition_flag[i] = 0
                    
                if ((total_density >= 0.45) and (total_density < 0.75)):
                ####
                    #setting the centre of circle for respective bot for circle extrusion
                    self.circle_extrusion[i] = 1
                    present_position = x[i]
                    present_position = np.append(present_position, 0)
                    centre = present_position + self.radius*(-direction_centroid_norm)
                    self.circle_centre[i] = centre[0:2]
                    
                    self.circle_start[i] = x[i]
                    
                    if hasattr(self,'env'):
                        self.env.set_flag(x[i,0],x[i,1])
                        self.env.set_flag(centre[0], centre[1])
                
                    
                    
            
            elif ((self.circle_extrusion[i] == 1) and (self.stop[i] == 0)):
                ###
                #Circle extrusion
                
                vector_towards_centre = self.circle_centre[i] - x[i]
                vector_towards_centre = np.append(vector_towards_centre, 0)
                movement_direction = np.cross(vector_towards_centre, self.z_vector)
                
                dist_from_centre = np.linalg.norm(vector_towards_centre)
                
                if (dist_from_centre > 25):
                    movement_direction = 0.9*movement_direction + 0.1*vector_towards_centre
                    
                movement_norm = movement_direction/np.linalg.norm(movement_direction)
                self.actions_list[i] = movement_norm[0:2]*10
                deposition_action.append(1)
                ##blinders to protect the bot from stopping at the start of circle extrusion
                self.blinders[i] = self.blinders[i] + 1
                
            #calculating the distance of the bot from the starting coordinate of the circle for respective bot    
            dist_from_start = np.linalg.norm(x[i] - self.circle_start[i])
            
            #calculating when to stop and jumping randomly for 25 steps
            if ((dist_from_start < 0.5) and (self.blinders[i] > 25) and (self.stop[i] == 0)):
                self.stop[i] = 1
                self.blinders[i] = 0
                self.circle_extrusion[i] = 0
                deposition_action.pop(-1)
                
            if ((self.stop[i] == 1) and (self.counter[i] < 25)):
                self.actions_list[i] = ((np.random.uniform(low = -1, high = 1, size = (2,)))*20)*10
                self.counter[i] = self.counter[i] + 1
                deposition_action.append(0)
            elif((self.stop[i] == 1) and (self.counter[i] >= 25)):
                self.stop[i] = 0
                self.actions_list[i] = ((np.random.uniform(low = -1, high = 1, size = (2,)))*20)*10
                deposition_action.append(0)
                
                
            #deposition_action.append(0)
        
        return np.concatenate((np.array(self.actions_list), np.array(deposition_action).reshape(self.n_agents, 1)),
                              axis=1)

    
    def set_env(self, env):
        self.env = env
