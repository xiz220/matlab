import numpy as np
from math import pi
import pdb
from fillet_rule import calculate_centroid

class WallFollow_v2:

    def __init__(self, line_length=10, turn_delay_max=5, slowdown_alpha=0.5, angle=45):
        self.n_agents = None
        self.deposition = None
        self.actions_list = None
        self.prev_x = [None, None]
        self.t = 0
        self.line_length = line_length
        self.jump_direction = 1
        self.random_jump = 0
        self.angle = angle
        self.radian = np.deg2rad(self.angle)
        self.blinders = None
        self.turn_delay = None
        self.turn_delay_actions = None
        self.turn_delay_max = turn_delay_max
        self.slowdown_alpha = slowdown_alpha
        self.z_vector = np.array([0,0,1])
    
    def check_directional_and_total_density(self, obs_grid):
        t = obs_grid
        row = t.shape[0]
        col = t.shape[1]
        
        total_density = np.count_nonzero(t)/t.size
        
        #north density
        n_count = np.count_nonzero(t[0:int(np.ceil(row/2)), :])
        n_size = float(t[0:int(np.ceil(row/2)), :].size)
        n_density = n_count/n_size
        
        #south density
        s_count = np.count_nonzero(t[int(np.floor(row/2)):row, :])
        s_size = float(t[int(np.floor(row/2)):row, :].size)
        s_density = s_count/s_size
        
        #east density
        e_count = np.count_nonzero(t[:, int(np.floor(col/2)):col])
        e_size = float(t[:, int(np.floor(col/2)):col].size)
        e_density = e_count/e_size
        
        #west density
        w_count = np.count_nonzero(t[:, 0:int(np.ceil(col/2))])
        w_size = float(t[:, 0:int(np.ceil(col/2))].size)
        w_density = w_count/w_size
        
        direction_wise_density = np.array([n_density, s_density, e_density, w_density])
        direction_wise_count = np.array([n_count, s_count, e_count, w_count])
        direction_wise_size = np.array([n_size, s_size, e_size, w_size])
        
        return direction_wise_density, direction_wise_count, direction_wise_size, total_density
    
    def direction_to_centroid(self, obs_grid):
        centroid = calculate_centroid(obs_grid)
        direction_centroid = centroid - np.floor((obs_grid.shape[0]-1)/2)*np.ones((2,))
        
        return direction_centroid
    
    
    def get_action(self, obs):
        
        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition = np.zeros((self.n_agents,))
            ##self.actions_list = []
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_jump = np.zeros((self.n_agents,2))
            self.counter = np.zeros((self.n_agents,1))
            self.turn_delay = np.zeros((self.n_agents,1))
            self.turn_delay_actions = np.zeros((self.n_agents,2))
        deposition_action = []
        x = obs['x']
        threshold = 0.8
        
        for i in range(self.n_agents):
            obs = sensor_reading[i, :, :]
            
            direction_centroid = self.direction_to_centroid(obs)
            direction_centroid = np.append(direction_centroid, 0)
                    
            direction_parallel_wall = np.cross(direction_centroid, self.z_vector)
                    
            direct_parallel_wall = direction_parallel_wall/np.linalg.norm(direction_parallel_wall)
            
            if (np.isnan(direct_parallel_wall).any()):
                direct_parallel_wall = direction_parallel_wall
            
            direction_wise_density, direction_wise_count, direction_wise_size, total_density = self.check_directional_and_total_density(obs)
            
            
            if ((total_density >= 0.45) and (total_density <= 0.55)):
                #checking if a sufficient mass of the wall is detected
                self.actions_list[i] = direct_parallel_wall[0:2]
                deposition_action.append(1)
                self.actions_list[i] = self.actions_list[i]*self.slowdown_alpha
            
            elif ((total_density > 0) and (total_density < 0.45)):
                #checking if there is a wall near by and should move in direction of wall
                self.actions_list[i] = direction_centroid[0:2]
                deposition_action.append(0)
                self.actions_list[i] = self.actions_list[i]*self.slowdown_alpha*1.25
                
            elif ((total_density > 0.55) and (total_density != 1)):
                #checking if the bot moved into the wall or is moving on the wall, then move direction opposite to wall
                self.actions_list[i] = -direction_centroid[0:2]
                deposition_action.append(1)
                self.actions_list[i] = self.actions_list[i]*self.slowdown_alpha*1.5
                
            else:
                #if inside a wall completely or in an empty area then jump randomly
                self.actions_list[i] = (np.random.rand(2)-0.5)*10
                deposition_action.append(0)
                
            
            # if self.counter[i] == 25:
            #     self.actions_list[i] = (np.random.rand(2)-0.5)*25
            #     self.counter[i] = 0
                
            # if ((self.prev_jump[i] == self.actions_list[i]).all()):
            #     self.counter[i] = self.counter[i] + 1
            # else:
            #     self.counter[i] = 0
                
            # #if self.counter[i] == 5:
            #  #   pdb.set_trace()
                
            # self.prev_jump[i] = self.actions_list[i]
        
                
                
        return np.concatenate((np.array(self.actions_list), np.array(deposition_action).reshape(self.n_agents, 1)),
                              axis=1)
