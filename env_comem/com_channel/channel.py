import sys

sys.path.append('..')
import numpy as np
from env_comem.utils import render_image, OneHotEncoding
import gym
import matplotlib.pyplot as plt


class Channel(object):
    def __init__(self, dict_size, type, seed, save_fig=False):
        type = type.lower()
        assert type in {'identity', 'one-hot'}
        assert isinstance(dict_size, int)
        assert dict_size > 0

        self.dict_size = dict_size
        self._type = type
        self._state = None

        self.send_space = gym.spaces.Discrete(n=self.dict_size)

        if self._type == "identity":
            self.read_space = gym.spaces.Discrete(n=self.dict_size)
        elif self._type == "one-hot":
            self.read_space = OneHotEncoding(size=self.dict_size)
        else:
            raise NotImplementedError

        self.seed(seed)
        self.save_fig = save_fig
        self.i = 0

    def _transform(self, message, mode):
        if mode == "identity":
            return message
        elif mode == "one-hot":
            vec = np.zeros(self.dict_size)
            vec[message] = 1
            return vec
        else:
            raise ValueError

    def send(self, message):
        assert 0 <= message < self.dict_size, "input message should be an int from 0 to dict_size -1"
        self._state = message

    def read(self):
        return self._transform(self._state, self._type)

    def seed(self, seed):
        self.send_space.seed(seed)
        self.read_space.seed(seed)

    def render(self, verbose):
        if verbose:
            img = self._transform(self._state, "one-hot").reshape((self.dict_size, 1))
            render_image(img, 2)
            if self.save_fig:
                plt.savefig(f'./figures/{self.i}')
                self.i += 1