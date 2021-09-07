
import numpy as np
import pdb
from wall_follow_controller import WallFollowController
from angle_beam_controller import AngleBeamController
from circle_extrusion_controller import CircleExtrusionController


class P4Rule:

    def __init__(self, wall_follow_kwargs, angle_beam_kwargs, circle_extrusion_kwargs, extrude_after_stop_time=10):
        self.n_agents = None
        self.deposition_action = None
        self.actions_list = None
        self.x = None
        self.t = 0
        self.extrude_after_stop_time = extrude_after_stop_time
        self.extrude_after_stop_counter = None

        self.robot_state = None

        # STOP CONDIITON
        self.n_corners = None

        # init controller modules
        self.wall_follow_controller = WallFollowController(**wall_follow_kwargs)
        self.angle_beam_controller = AngleBeamController(**angle_beam_kwargs)
        self.circle_extrusion_controller = CircleExtrusionController(**circle_extrusion_kwargs)

    def get_action(self, obs):
        """
        Takes in an observation and returns the next action: move randomly, deposit if adjacent to existing material.
        :param obs: observation dict {'x': robot_states (n_agents x 2 np array), 'sensor_readings': (n_agents x row_dim x col_dim) np array}
        :return: n_agents x 3 np array, where row i represents the i'th robot's: [x_action, y_action, deposition_action]
        """

        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition_action = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.robot_state = np.zeros((self.n_agents,))
            self.n_corners = np.zeros((self.n_agents,))
            self.extrude_after_stop_counter = np.ones((self.n_agents,)) * self.extrude_after_stop_time

        self.x = obs['x']
        print('time: ', self.t)
        print('state: ', self.robot_state)
        print(self.angle_beam_controller.turn_counter)
        for i in range(self.n_agents):

            ############### WALLFOLLOW STATE ####################
            # if self.robot_state[i] == 0:
            #     self.actions_list[i], self.deposition_action[i] = self.wall_follow_controller.get_action(obs, i)

            #     if self.wall_follow_controller.corner_counter[i] >= 1:
            #         self.wall_follow_controller.corner_counter[i] = 0
            #         self.robot_state[i] = 1
            #         if hasattr(self, 'env'):
            #             self.env.set_flag(self.x[i, 0], self.x[i, 1])

            # ############### ANGLE BEAM STATE ####################
            # if self.robot_state[i] == 1:  # ANGLE BEAM
            #     self.actions_list[i], self.deposition_action[i] = self.angle_beam_controller.get_action(obs, i)

            #     # if it is SECOND time turning, then change robot to wall follow state, increment number of corners
            #     if self.angle_beam_controller.turn_counter[i] >= 2:
            #         self.angle_beam_controller.reset_agent_i(i)
            #         self.robot_state[i] = 0
            #         self.n_corners[i] += 1
            #         if hasattr(self, 'env'):
            #             self.env.set_flag(self.x[i, 0], self.x[i, 1])
            
            self.actions_list[i], self.deposition_action[i] = self.circle_extrusion_controller.get_action(obs, i)
            
        self.t += 1
        #self.check_stop_position()
        return np.concatenate((np.array(self.actions_list), np.array(self.deposition_action).reshape(self.n_agents, 1)),
                              axis=1)

    def check_stop_position(self):
        for i in range(self.n_agents):
            if self.n_corners[i] >= 2:
                self.actions_list[i] = np.array([0, 0])
                if self.extrude_after_stop_counter[i] >= 8:
                    self.deposition_action[i] = 0
                else:
                    self.deposition_action[i] = 1
                    self.extrude_after_stop_counter[i] += 1
        # print(self.n_corners) TODO: add in this print conditional on -test flag

    def set_env(self, env):
        self.env = env
