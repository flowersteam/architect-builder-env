import numpy as np
import readchar
import pickle

from env_comem.gym_gridworld.buildworld import Buildworld
from main_comem.mcts.mcts import MCTS
from main_comem.mcts.utils import EnvTransitionFct, EnvRewardFct, PossibleActions
from main_comem.mcts.tree_policy import UCT
from main_comem.mcts.default_policy import HeuristicDefaultPolicy, MonteCarloReturnPolicy
import main_comem.mcts.utils as mctsutils
from tqdm import tqdm

import time

# # MACROS
DO_NOTHING = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
TOGGLE = 5

# Key mapping
arrow_keys = {
    '\x1b[D': LEFT,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[A': UP,
    ' ': DO_NOTHING,
    't': TOGGLE}

if __name__ == '__main__':

    discount_factor = 0.95

    bw = Buildworld(grid_size=(6, 6),
                    change_goal=False,
                    obs_type='xy_continuous',
                    verbose=True,
                    seed=12346,
                    goal='make_shape')
    policy_type = 'mcts'

    if policy_type == 'mcts':

        # default_policy = MonteCarloReturnPolicy(
        #     env_transition_fct=bw.transition_fct,
        #     env_reward_fct=bw.reward_fct,
        #     env_is_terminal_fct=lambda x: False,
        #     possible_action_fct=PossibleActions(bw.action_space, 1234),
        #     discount_factor=discount_factor,
        #     horizon=5)

        default_policy= HeuristicDefaultPolicy(bw.heuristic_value, discount_factor, scaling=0.5)
        policy = MCTS(state_cast_fct=lambda x: x,
                      transition_fct=EnvTransitionFct(bw.transition_fct),
                      reward_fct=EnvRewardFct(bw.reward_fct),
                      is_terminal_fct=lambda x: False,
                      possible_actions_fct=PossibleActions(bw.action_space, 1234),
                      budget=25,
                      tree_policy=UCT(2 ** 0.5, 12324),
                      default_policy=default_policy,
                      discount_factor=discount_factor,
                      keep_tree=False,
                      get_new_root=mctsutils.get_new_root,
                      max_depth=1000)

    n_episodes = 200
    steps_per_episode = 100
    bw.render_legend()
    images = []
    successes = []

    # ## Just to check random states that are generated
    # from env_comem.gym_gridworld.buildworld import render_image_on_figure
    # import matplotlib.pyplot as plt
    # time_1 = time.time()
    # for _ in range(4*600*40):
    #     random_tile = bw.get_random_state()
    #     #render_image_on_figure(random_tile, plt.figure(bw.this_fig_num))
    #
    # print("--- %s seconds ---" % (time.time() - time_1))
    #
    # import sys
    # sys.exit()

    pbar = tqdm(total=n_episodes)

    for episode in range(n_episodes):
        obs, _ = bw.reset()
        bw.render()
        bw.render()
        heuristic_value, heuristic_steps = bw.heuristic_value(obs, 0.95, True)
        for step in range(steps_per_episode):
            bw.heuristic_value(obs, 0.95, True)
            im = bw.render()
            images.append(im)
            if policy_type == 'random':
                action = bw.action_space.sample()
            elif policy_type == 'interactive':
                key = readchar.readkey()
                if key not in arrow_keys.keys():
                    break

                action = arrow_keys[key]
            elif policy_type == 'heuristic':
                action = bw.heuristic_policy(obs)

            elif policy_type == 'mcts':
                action = policy.act(obs)

            obs, reward, _, _ = bw.step(action)
            if reward == 0.9 or reward == 1.:
                succ = 1.
                successes.append(succ)
                print(step + 1 == heuristic_steps)
                break
            print(reward)
        pbar.update()

    print(np.mean(successes))

    with open('images.pkl', 'wb') as fh:
        pickle.dump(images, fh)
        fh.close()
