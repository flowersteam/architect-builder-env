import numpy as np
import sys

sys.path.append('..')
from gym import Env, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from env_comem.utils import render_image

## AGENT ACTIONS

# 0 : DO_NOTHING
# 1 : LEFT
# 2 : RIGHT
# 3 : UP
# 4 : DOWN
ACTIONS_DIRECTION_DICT = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
PRINT_ACTION_DICT = {0: 'x', 1: '\u2190', 2: '\u2192', 3: '\u2191', 4: '\u2193'}

SET_OF_REWARD_TYPE = set(['manhattan', 'sparse', 'progress'])


class Gridworld(Env):
    num_env = 0

    def __init__(self, n_objects, grid_size, reward_type, obs_type, change_goal, verbose, seed):
        # states are (x,y) coordinate

        if not isinstance(grid_size, (list, tuple)) or not len(grid_size) == 2:
            raise ValueError("grid_size argument must be a list/tuple of length 2")

        reward_type = reward_type.lower()
        if not reward_type in SET_OF_REWARD_TYPE:
            raise ValueError(f"reward_type must be in {SET_OF_REWARD_TYPE}")

        self._action_direction_dict = ACTIONS_DIRECTION_DICT
        self.action_space = spaces.Discrete(len(self._action_direction_dict))

        self._grid_size = tuple(grid_size)
        # x_lim and y_lim are one above maximum possible values such that x < x_lim and y < y_lim
        self._x_lim, self._y_lim = self._grid_size

        self.obs_type = obs_type  # the native obs_type of the engine is 'xy_discrete' so low-level operations use it
        self.compute_observation_space()

        self.change_goal = change_goal
        self._goal_state = None

        self.verbose = verbose
        self._reward_type = reward_type
        Gridworld.num_env += 1
        self.this_fig_num = 2 * Gridworld.num_env
        if self.verbose == True:
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
        self.seed(seed)

        if self.obs_type == 'xy_discrete':
            # transitions are not goal dependent so we can compute them only once
            self._compute_Pmat()

    def compute_observation_space(self):
        if self.obs_type == 'xy_discrete':
            # x,y coordinate in terms of j,i of the position matrix
            self.observation_space = spaces.MultiDiscrete(self._grid_size)
            # for value itération
            self.nA = self.action_space.n
            self.nS = self._x_lim * self._y_lim

        elif self.obs_type == 'xy_continuous':
            # (x/(x_lim -1), y/(y_lim-1)) continuous coordinates
            self.observation_space = spaces.Box(low=np.array([0., 0.]), high=np.array([1., 1.]))

        else:
            raise NotImplementedError

    def seed(self, seed):
        self.np_random, _ = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _engine_sample_random_state(self):
        return tuple(self.np_random.randint((0, 0), self._grid_size))

    def sample_random_state(self):
        discrete_state = tuple(self.np_random.randint((0, 0), self._grid_size))
        return self.encode_obs(discrete_state, to_obs_type=self.obs_type)

    def _sample_random_start_goal_states(self):
        start_state = self._engine_sample_random_state()
        goal_state = self._engine_sample_random_state()
        return start_state, goal_state

    def compute_manhattan_distance(self, obs):
        state = self.decode_obs(obs, self.obs_type)
        return self._engine_compute_manhattan_distance(state)

    def _engine_compute_manhattan_distance(self, state):
        dist = abs(self._goal_state[0] - state[0]) + abs(self._goal_state[1] - state[1])
        return dist

    def _engine_on_goal(self, state):
        state = tuple(state)
        return state == self._goal_state

    def on_goal(self, obs):
        state = self.decode_obs(obs, from_obs_type=self.obs_type)
        return self._engine_on_goal(state)

    def done(self, obs):
        return self.on_goal(obs)

    def _engine_reset(self):
        if self.change_goal:
            self._current_state, self._goal_state = self._sample_random_start_goal_states()
        else:
            if self._goal_state is None:
                self._current_state, self._goal_state = self._sample_random_start_goal_states()
            else:
                self._current_state = self._engine_sample_random_state()

        self._goal_image = self._goal_to_image()
        if self.obs_type == 'xy_discrete':
            self._compute_Rmat()
        return self._current_state, self._goal_state

    def reset(self):
        current_state, goal_state = self._engine_reset()
        return self.encode_obs(current_state, to_obs_type=self.obs_type), \
               self.encode_obs(goal_state, to_obs_type=self.obs_type)

    def get_init_state(self):
        init_state = self.sample_random_state()
        return self.encode_obs(init_state, to_obs_type=self.obs_type)

    @property
    def _internal_state(self):
        return {'current_state': tuple(self._current_state),
                'goal_state': tuple(self._goal_state)}

    def reset_to_internal_state(self, internal_state):
        self._goal_state = tuple(internal_state['goal_state'])
        self._current_state = tuple(internal_state['current_state'])
        self._goal_image = self._goal_to_image()
        if self.obs_type == 'xy_discrete':
            self._compute_Rmat()
        return self._current_state, self._goal_state

    def _clamp_to_grid(self, state):
        return (max(0, min(state[0], self._x_lim - 1)), max(0, min(state[1], self._y_lim - 1)))

    def _engine_reward_fct(self, state, action, next_state):
        if self._reward_type == 'manhattan':
            distance = self.compute_manhattan_distance(next_state)
            worst_distance = sum(self._grid_size)
            return (worst_distance - distance) / worst_distance
        elif self._reward_type == 'sparse':
            return self._engine_on_goal(next_state)
        elif self._reward_type == 'progress':
            return self.compute_manhattan_distance(state) - self.compute_manhattan_distance(next_state)
        else:
            raise ValueError

    def reward_fct(self, obs, action, next_obs):
        state = self.decode_obs(obs, from_obs_type=self.obs_type)
        next_state = self.decode_obs(next_obs, from_obs_type=self.obs_type)
        return self._engine_reward_fct(state, action, next_state)

    def two_dim_state_to_one_dim_state(self, two_dim_state):
        # two-dim states are (x,y) and one-dim start at 0 on lower-left corner and increase from left
        # to right and down to up, max value is at upper-right corner v = x + y * max_x.
        return two_dim_state[0] + two_dim_state[1] * self._x_lim

    def one_dim_state_to_two_dim_state(self, one_dim_state):
        # two-dim states are (x,y) and one-dim start at 0 on lower-left corner and increase from left
        # to right and down to up, max value is at upper-right corner (x,y) = (v % x_max, v // x_max) .
        return (one_dim_state % self._x_lim, one_dim_state // self._x_lim)

    def _compute_Rmat(self):
        self.Rmat = np.zeros((self.nS, self.nA, self.nS))

        # NOTE THAT THIS IS ONLY OK FOR DETERMINISTIC REWARDS
        for one_dim_state in range(self.nS):
            for action in range(self.nA):
                for one_dim_next_state in range(self.nS):
                    two_dim_state = self.one_dim_state_to_two_dim_state(one_dim_state)
                    two_dim_next_state = self.one_dim_state_to_two_dim_state(one_dim_next_state)
                    self.Rmat[one_dim_state, action, one_dim_next_state] = \
                        self.reward_fct(two_dim_state, action, two_dim_next_state)

    def _compute_Pmat(self):
        self.Pmat = np.zeros((self.nS, self.nA, self.nS))

        # NOTE THAT THIS IS ONLY OK FOR DETERMINISTIC TRANSITIONS
        for one_dim_state in range(self.nS):
            for action in range(self.nA):
                two_dim_state = self.one_dim_state_to_two_dim_state(one_dim_state)
                two_dim_next_state = self._engine_transition_fct(two_dim_state, action)
                one_dim_next_state = self.two_dim_state_to_one_dim_state(two_dim_next_state)
                self.Pmat[one_dim_state, action, one_dim_next_state] = 1.

    def _engine_transition_fct(self, state, action):
        next_state = self._clamp_to_grid((state[0] + self._action_direction_dict[action][0],
                                          state[1] + self._action_direction_dict[action][1]))
        return next_state

    def transition_fct(self, state, action):
        state = self.decode_obs(state, from_obs_type=self.obs_type)
        next_state = self._engine_transition_fct(state, action)
        return self.encode_obs(next_state, to_obs_type=self.obs_type)

    def step(self, action):
        info = {}
        action = int(action)
        nxt_agent_state = self._engine_transition_fct(state=self._current_state, action=action)

        if action == 0:  # deliberately stay in place
            info['legal_action'] = True

        elif nxt_agent_state == self._current_state:  # stuck in a wall
            info['legal_action'] = False

        else:
            info['legal_action'] = True

        reward = self._engine_reward_fct(self._current_state, action, nxt_agent_state)
        done = False  # we are in the infinite setting (with interaction time-limit) there is no final/absorbing state
        # and therefore done is always False
        nxt_obs = self.encode_obs(nxt_agent_state, self.obs_type)
        info.update({'on_goal': done, 'manhattan_distance': self.compute_manhattan_distance(nxt_agent_state),
                     'goal_image': self._goal_image,
                     'xy_continuous': self.encode_obs(nxt_agent_state, 'xy_continuous')})

        self._current_state = nxt_agent_state

        return (nxt_obs, reward, done, info)

    def decode_obs(self, state, from_obs_type):
        if from_obs_type == 'xy_continuous':
            return [state[0] * (self._x_lim - 1), state[1] * (self._y_lim - 1)]
        elif from_obs_type == 'xy_discrete':
            return state
        else:
            raise NotImplementedError

    def encode_obs(self, state, to_obs_type):
        if to_obs_type == 'xy_discrete':
            return list(state)

        elif to_obs_type == 'xy_continuous':
            return [state[0] / (self._x_lim - 1), state[1] / (self._y_lim - 1)]

        else:
            raise NotImplementedError

    def heuristic_value(self, obs, discount, return_steps=False):
        steps = self.compute_manhattan_distance(obs)
        value = discount ** steps / (1 - discount)

        if return_steps:
            return value, steps
        else:
            return value

    def _compute_n_steps_optim(self, obs):
        return self.compute_manhattan_distance(obs)

    def _position_to_image(self):
        image = np.zeros(list(self._grid_size) + [3])
        image[self._current_state] = [1, 0, 0]
        return image

    def _goal_to_image(self):
        image = np.zeros(list(self._grid_size) + [3])
        image[self._goal_state] = [0, 1, 0]
        return image

    def render(self, mode='human', everything=False):
        if not self.verbose:
            return
        img = self._position_to_image()
        render_image(img, self.this_fig_num)
        if everything:
            self.render_goal()
        return

    def render_goal(self):
        if not self.verbose:
            return
        img = self._goal_to_image()
        render_image(img, self.this_fig_num + 1)
        return
