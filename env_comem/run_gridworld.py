import argparse
import readchar
from alfred.utils.config import parse_bool
from env_comem.gym_gridworld.gridworld import Gridworld


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=3)
    parser.add_argument('--n_objects', type=int, default=3)
    parser.add_argument('--grid_size', type=int, nargs='+', default=(5, 10))
    parser.add_argument('--reward_type', type=str, default='progress',
                        choices=['manhattan', 'sparse', 'progress'])
    parser.add_argument('--obs_type', type=str, default='xy_discrete')
    parser.add_argument('--change_goal', type=parse_bool, default=True)
    parser.add_argument('--seed', type=int, default=131214)
    parser.add_argument('--verbose', type=parse_bool, default=True)
    return parser.parse_args()


# # MACROS
DO_NOTHING = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
KILL = 5

actions_dict = {DO_NOTHING: 'do_noting',
                LEFT: 'left',
                RIGHT: 'right',
                UP: 'up',
                DOWN: 'down'}

# Key mapping
arrow_keys = {
    '\x1b[D': LEFT,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[A': UP,
    ' ': DO_NOTHING}


def _run_env(args):
    env = Gridworld(n_objects=args.n_objects,
                    grid_size=args.grid_size,
                    reward_type=args.reward_type,
                    change_goal=args.change_goal,
                    obs_type=args.obs_type,
                    verbose=args.verbose,
                    seed=args.seed)
    env.seed(args.seed)

    for ep in range(args.n_episodes):
        _ = env.reset()

        env.render_goal()
        env.render()

        while True:
            key = readchar.readkey()
            if key not in arrow_keys.keys():
                break

            action = arrow_keys[key]
            next_state, reward, done, info = env.step(action)
            print([next_state, reward, done])
            info.update({'action': actions_dict[action]})
            info.pop('goal_image')
            print(info)

            env.render()


if __name__ == '__main__':
    args = get_args()
    _run_env(args)
