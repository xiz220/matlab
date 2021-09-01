import numpy as np
from fillet_rule import calculate_centroid


def direction_to_centroid(obs_grid):
    """
     Parameters
     ----------
     obs_grid : TYPE - np array
         DESCRIPTION. - sensor_readings of dimensions [row_dim, col_dim]

     Returns
     -------
     numpy (1,2) array:
     unnormalized vector from the robot position to the centroid of the observation
     """

    centroid = calculate_centroid(obs_grid)
    direction_centroid = centroid - np.floor((obs_grid.shape[0] - 1) / 2) * np.ones((2,))

    return direction_centroid


class AngleBeamController:

    def __init__(self, turn_delay_max=5, angle=45, slowdown_alpha=0.5):

        self.n_agents = None
        self.deposition_action = None
        self.actions_list = None
        self.t = 0

        # ANGLE BEAM EXTRUSION VARIABLES
        self.angle = angle
        self.radian = np.deg2rad(self.angle)
        self.blinders = None
        self.turn_delay = None
        self.turn_delay_actions = None
        self.turn_delay_max = turn_delay_max/slowdown_alpha
        self.slowdown_alpha = slowdown_alpha
        self.prev_jump = None
        self.turn_counter = None

    def get_action(self, obs, i):

        sensor_reading = obs['sensor_readings']
        if self.n_agents is None:
            self.n_agents = len(sensor_reading)
            self.deposition_action = np.zeros((self.n_agents,))
            self.actions_list = [np.zeros((2,)) for _ in range(self.n_agents)]
            self.prev_x = np.zeros((self.n_agents, 7, 2))  ##
            self.turn_delay = np.zeros((self.n_agents, 1))
            self.turn_delay_actions = np.zeros((self.n_agents, 2))
            self.prev_jump = np.zeros((self.n_agents, 2))
            self.blinders = np.zeros((self.n_agents, 1))
            self.turn_delay = np.zeros((self.n_agents, 1))
            self.turn_delay_actions = np.zeros((self.n_agents, 2))
            self.turn_counter = np.zeros((self.n_agents, 1))


        jump_dir = - direction_to_centroid(sensor_reading[i])
        threshold = 0.8

        if self.turn_delay[i] == 0:  # if we are not in turn delay state
            # TODO are turn_delay and blinders redundant?? draw out FSM diagram, see if we can simplify this
            # also, eventually make this a FSM with straightforward state
            # if observation is all zeros or all ones
            if (jump_dir == np.array([0, 0])).all():

                # move in random direction, take blinders off, do not deposit material
                self.actions_list[i] = (np.random.rand(2) - 0.5) * 100
                self.prev_jump[i] = self.actions_list[i] * 0.2  # todo do we need to update this anywhere else??
                self.blinders[i] = 0
                self.deposition_action[i] = 0
            else:  # else, we have some observation centroid

                # normalize
                jump_dir = jump_dir / np.linalg.norm(jump_dir)
                dir_diff = (1 / 10) * self.prev_jump[i] - jump_dir
                # if the difference between old mvmt dir and new movement dir is above threshold and the blinders
                # are off
                if np.linalg.norm(dir_diff) > threshold and self.blinders[i] == 0:
                    self.turn_delay[i] = self.turn_delay_max
                    self.turn_delay_actions[i, :] = self.prev_jump[i]
                    self.blinders[i] = 10 + self.turn_delay_max
                    # find the nearest angle to snap to
                    cand_vectors = np.array([[np.cos(self.radian), np.sin(self.radian)],
                                             [np.cos(self.radian), -np.sin(self.radian)],
                                             [-np.cos(self.radian), np.sin(self.radian)],
                                             [-np.cos(self.radian), -np.sin(self.radian)]])

                    dot_products = dir_diff @ cand_vectors.T
                    snapped_jump_dir = cand_vectors[np.argmin(dot_products), :]

                    # move in the direction opposite the new disturbance
                    self.actions_list[i] = 10 * snapped_jump_dir * self.slowdown_alpha
                    self.deposition_action[i] = 1
                    self.prev_jump[i] = 10 * snapped_jump_dir

                    # import pdb; pdb.set_trace()

                else:  # else, no significant new disturbance (or blinders on)
                    if self.blinders[i] != 0:  # decrement blinders if they are on
                        self.blinders[i] = self.blinders[i] - 1
                    self.actions_list[i] = self.prev_jump[i] * self.slowdown_alpha
                    self.deposition_action[i] = 1

        else:  # if in turn delay state
            self.turn_delay[i] = self.turn_delay[i] - 1
            self.actions_list[i] = self.turn_delay_actions[i, :] * self.slowdown_alpha
            self.deposition_action[i] = 1

            #increment turn counter on exit of turn delay --> into turn and put blinders on state
            if self.turn_delay[i] == 0:
                self.turn_counter[i] += 1

        return self.actions_list[i], self.deposition_action[i]

    def reset_agent_i(self, i):
        self.turn_delay[i] = 0
        self.turn_delay_actions[i,:] = np.zeros((1, 2))
        self.prev_jump[i,:] = np.zeros((1, 2))
        self.blinders[i] = 0
        self.turn_counter[i] = 0