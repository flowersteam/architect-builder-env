import gym
import numpy as np

from env_comem.utils import OneHotEncoding
from main_comem.utils.obs_dict import ObsDict


class ObsBlender(object):
    def __init__(self, gw_obs_space, channel_obs_space, type=None):
        if type == 'obs_dict':
            assert (isinstance(gw_obs_space, gym.spaces.Box)
                    and
                    isinstance(channel_obs_space, gym.spaces.Discrete))
            self._type = 'obs_dict'
            self.gw_obs_space = gw_obs_space
            self.channel_obs_space = channel_obs_space
            self.obs_space = {'tile': gw_obs_space, 'message': channel_obs_space}

        elif type == 'obs_dict_flat_gw':
            assert (isinstance(gw_obs_space, gym.spaces.Box)
                    and
                    isinstance(channel_obs_space, gym.spaces.Discrete))

            if len(gw_obs_space.shape) > 1:
                gw_low = gw_obs_space.low.flatten()
                gw_high = gw_obs_space.high.flatten()
            else:
                gw_low = gw_obs_space.low
                gw_high = gw_obs_space.high

            self._type = 'obs_dict_flat_gw'
            self.obs_space = gym.spaces.Box(low=gw_low, high=gw_high)
            self.channel_obs_space = channel_obs_space
            self.obs_space = {'tile': gw_obs_space, 'message': channel_obs_space}


        elif (isinstance(gw_obs_space, gym.spaces.MultiDiscrete)
              and
              isinstance(channel_obs_space, gym.spaces.Discrete)) and type == None:

            self.obs_space = gym.spaces.MultiDiscrete(list(gw_obs_space.nvec)
                                                      + [channel_obs_space.n])

            self.gw_obs_space = gym.spaces.MultiDiscrete(list(gw_obs_space.nvec))
            self.channel_obs_space = gym.spaces.Discrete(channel_obs_space.n)

            self._type = 'tabular'

        elif (isinstance(gw_obs_space, gym.spaces.Box)
              and
              isinstance(channel_obs_space, OneHotEncoding)) and type == None:

            if len(gw_obs_space.shape) > 1:
                gw_low = gw_obs_space.low.flatten()
                gw_high = gw_obs_space.high.flatten()
            else:
                gw_low = gw_obs_space.low
                gw_high = gw_obs_space.high

            concat_low = np.concatenate((gw_low, [0.] * channel_obs_space.size))
            concat_high = np.concatenate((gw_high, [1.] * channel_obs_space.size))

            self.obs_space = gym.spaces.Box(low=concat_low, high=concat_high)

            self.gw_obs_space = gym.spaces.Box(low=gw_low, high=gw_high)

            self.channel_obs_space = OneHotEncoding(channel_obs_space.size)

            self._type = 'concat'

        elif (isinstance(gw_obs_space, gym.spaces.Box)
              and
              isinstance(channel_obs_space, gym.spaces.Discrete)) and type == None:

            gw_low = gw_obs_space.low
            gw_high = gw_obs_space.high

            chnl_obs_low = np.expand_dims(np.zeros_like(gw_low[:, :, 0]), -1)
            chnl_obs_high = np.expand_dims(np.ones_like(gw_low[:, :, 0]) * float(channel_obs_space.n), -1)

            add_channel_low = np.concatenate([gw_low, chnl_obs_low], -1)
            add_channel_high = np.concatenate([gw_high, chnl_obs_high], -1)

            self.obs_space = gym.spaces.Box(low=add_channel_low, high=add_channel_high)

            self.gw_obs_space = gw_obs_space

            self.channel_obs_space = channel_obs_space

            self._type = 'additional_channel'
        else:
            raise NotImplementedError

    def blend(self, gw_obs, chnl_obs):
        if self._type == 'tabular':
            return TabularBlendedObs(gw_obs, chnl_obs)
        elif self._type == 'concat':
            gw_obs = np.asarray(gw_obs, dtype=np.float)
            if len(np.shape(gw_obs)) > 1:
                gw_obs = gw_obs.flatten()
            return np.concatenate((gw_obs, chnl_obs))
        elif self._type == 'additional_channel':
            gw_obs = np.asarray(gw_obs, dtype=np.float)
            chnl_obs = np.expand_dims(np.ones_like(gw_obs[:, :, 0]), -1) * float(chnl_obs)
            to_return = np.concatenate((gw_obs, chnl_obs), -1)
            return to_return
        elif self._type == 'obs_dict':
            return ObsDict({'tile': np.array(gw_obs), 'message': np.array(chnl_obs)})
        elif self._type == 'obs_dict_flat_gw':
            gw_obs = np.asarray(gw_obs, dtype=np.float)
            if len(np.shape(gw_obs)) > 1:
                gw_obs = gw_obs.flatten()
            return ObsDict({'tile': gw_obs, 'message': np.array(chnl_obs)})
        else:
            raise NotImplementedError

    def np_batch_blend_raw(self, batch_gw_obs, batch_chnl_obs):
        assert len(batch_gw_obs) == len(batch_chnl_obs)
        if self._type == 'tabular':
            batch_gw_obs = np.asarray(batch_gw_obs)
            batch_chnl_obs = np.asarray(batch_chnl_obs)

            assert len(batch_gw_obs.shape) == 2

            if len(batch_chnl_obs.shape) == 1:
                batch_chnl_obs = np.expand_dims(batch_chnl_obs, axis=1)

            raw_blended = np.concatenate((batch_gw_obs, batch_chnl_obs), axis=1)

            return raw_blended

        elif self._type == 'concat':
            batch_gw_obs = np.asarray(batch_gw_obs)
            batch_chnl_obs = np.asarray(batch_chnl_obs)

            if len(batch_gw_obs.shape) > 2:
                batch_gw_obs = np.reshape(batch_gw_obs, (batch_gw_obs.shape[0],
                                                         batch_gw_obs.shape[1] * batch_gw_obs.shape[2] *
                                                         batch_gw_obs.shape[3]))

            assert len(batch_chnl_obs.shape) == 2

            raw_blended = np.concatenate((batch_gw_obs, batch_chnl_obs), axis=1)

            return raw_blended

        elif self._type == 'additional_channel':
            batch_gw_obs = np.asarray(batch_gw_obs)
            batch_chnl_obs = np.asarray(batch_chnl_obs)

            assert len(batch_gw_obs.shape) == 4
            if len(batch_chnl_obs.shape) == 1:
                batch_chnl_obs = np.expand_dims(batch_chnl_obs, -1)

            add_channel = np.einsum('bxyc,bc -> bxyc', np.expand_dims(np.ones_like(batch_gw_obs[:, :, :, 0]), -1),
                                    batch_chnl_obs)

            raw_blended = np.concatenate((batch_gw_obs, add_channel), -1)

            return raw_blended

        elif self._type == 'obs_dict':
            return ObsDict({'tile': np.array(batch_gw_obs), 'message': np.array(batch_chnl_obs)})

        elif self._type == 'obs_dict_flat_gw':
            batch_gw_obs = np.asarray(batch_gw_obs)
            if len(batch_gw_obs.shape) > 2:
                batch_gw_obs = np.reshape(batch_gw_obs, (batch_gw_obs.shape[0],
                                                         batch_gw_obs.shape[1] * batch_gw_obs.shape[2] *
                                                         batch_gw_obs.shape[3]))

            return ObsDict({'tile': batch_gw_obs, 'message': np.array(batch_chnl_obs)})
        else:
            raise NotImplementedError

    def seed(self, seed):
        self.obs_space.seed(seed)


class BlendedObs(object):
    def __init__(self):
        pass


class TabularBlendedObs(BlendedObs):
    def __init__(self, gw_obs, channel_obs):
        assert all([isinstance(coord, (int, np.integer)) for coord in gw_obs])
        assert isinstance(channel_obs, (int, np.integer))

        self._obs = {'blended_obs': tuple(list(gw_obs) + [channel_obs]),
                     'gw_obs': tuple(gw_obs),
                     'channel_obs': int(channel_obs)}

    @property
    def state(self):
        return self._obs['blended_obs']

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return self.state.__hash__()
