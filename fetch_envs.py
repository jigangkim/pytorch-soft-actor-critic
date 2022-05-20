import gym
import itertools
import numpy as np

from gym.core import ObservationWrapper
from gym.envs.registration import register
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv as FetchPickAndPlaceEnv_gym
from gym.envs.robotics.fetch.push import FetchPushEnv as FetchPushEnv_gym
from gym.envs.robotics.fetch.reach import FetchReachEnv as FetchReachEnv_gym
from gym.envs.robotics.fetch.slide import FetchSlideEnv as FetchSlideEnv_gym


class FetchPickAndPlaceEnv(FetchPickAndPlaceEnv_gym):
    def __init__(self, *args, goal_dist='uniform', **kwargs):
        self.goal_dist = goal_dist
        super(FetchPickAndPlaceEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, achieved_goal, goal, info):
        return super(FetchPickAndPlaceEnv, self).compute_reward(achieved_goal, goal, info) + 1

    def _sample_goal(self):
        if self.goal_dist == 'uniform':
            return super(FetchPickAndPlaceEnv, self)._sample_goal()
        elif self.goal_dist == 'compact':
            return self._sample_compact_goal()
        elif self.goal_dist == 'single':
            return self._sample_single_goal()
        else:
            raise NotImplementedError

    def _sample_compact_goal(self):
        # self.initial_gripper_xpos[:3] = [1.3419322596591514, 0.7491003711085993, 0.5347228411650838]
        # self.target_range = 0.15
        # self.target_offset = 0.0
        # self.height_offset = 0.42469974945955385
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            0.6*self.target_range, 0.8*self.target_range, size=3
        )
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)

        return goal.copy()

    def _sample_single_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.target_range
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)

        return goal.copy()


class FetchPushEnv(FetchPushEnv_gym):
    def __init__(self, *args, goal_dist='uniform', **kwargs):
        self.goal_dist = goal_dist
        super(FetchPushEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, achieved_goal, goal, info):
        return super(FetchPushEnv, self).compute_reward(achieved_goal, goal, info) + 1

    def _sample_goal(self):
        if self.goal_dist == 'uniform':
            return super(FetchPushEnv, self)._sample_goal()
        elif self.goal_dist == 'compact':
            return self._sample_compact_goal()
        elif self.goal_dist == 'single':
            return self._sample_single_goal()
        else:
            raise NotImplementedError

    def _sample_compact_goal(self):
        # self.initial_gripper_xpos[:3] = [1.347869484341299, 0.748949476594253, 0.4136377287383395]
        # self.target_range = 0.15
        # self.target_offset = 0.0
        # self.height_offset = 0.42469974945955385
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            0.6*self.target_range, 0.8*self.target_range, size=3
        )
        goal += self.target_offset
        goal[2] = self.height_offset

        return goal.copy()

    def _sample_single_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.target_range
        goal += self.target_offset
        goal[2] = self.height_offset

        return goal.copy()


class FetchReachEnv(FetchReachEnv_gym):
    def __init__(self, *args, goal_dist='uniform', **kwargs):
        self.goal_dist = goal_dist
        super(FetchReachEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, achieved_goal, goal, info):
        return super(FetchReachEnv, self).compute_reward(achieved_goal, goal, info) + 1

    def _sample_goal(self):
        if self.goal_dist == 'uniform':
            return super(FetchReachEnv, self)._sample_goal()
        elif self.goal_dist == 'compact':
            return self._sample_compact_goal()
        elif self.goal_dist == 'single':
            return self._sample_single_goal()
        else:
            raise NotImplementedError

    def _sample_compact_goal(self):
        # self.initial_gripper_xpos[:3] = [1.3418322596791512, 0.7491003840395923, 0.5347228411518367]
        # self.target_range = 0.15
        # self.target_offset = 0.0
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            0.6*self.target_range, 0.8*self.target_range, size=3
        )

        return goal.copy()

    def _sample_single_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.target_range

        return goal.copy()


class FetchSlideEnv(FetchSlideEnv_gym):
    def __init__(self, *args, goal_dist='uniform', **kwargs):
        self.goal_dist = goal_dist
        super(FetchSlideEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, achieved_goal, goal, info):
        return super(FetchSlideEnv, self).compute_reward(achieved_goal, goal, info) + 1

    def _sample_goal(self):
        if self.goal_dist == 'uniform':
            return super(FetchSlideEnv, self)._sample_goal()
        elif self.goal_dist == 'compact':
            return self._sample_compact_goal()
        elif self.goal_dist == 'single':
            return self._sample_single_goal()
        else:
            raise NotImplementedError

    def _sample_compact_goal(self): # TODO:
        # self.initial_gripper_xpos[:3] = [0.9957062748555665, 0.7489068402663717, 0.412685995916744]
        # self.target_range = 0.3
        # self.target_offset = [0.4, 0.0, 0.0]
        # self.height_offset = 0.414018943520536
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            0.6*self.target_range, 0.8*self.target_range, size=3
        )
        goal += self.target_offset
        goal[2] = self.height_offset

        return goal.copy()

    def _sample_single_goal(self):
        goal = self.initial_gripper_xpos[:3] + self.target_range
        goal += self.target_offset
        goal[2] = self.height_offset

        return goal.copy()


class NonGoalEnvWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        obs_space, goal_space = self.env.observation_space['observation'], self.env.observation_space['desired_goal']
        low = np.concatenate([obs_space.low, goal_space.low, goal_space.low])
        high = np.concatenate([obs_space.high, goal_space.high, goal_space.high])
        self.observation_space = gym.spaces.Box(low, high)

        self._obs_dim = obs_space.shape[0]
        self._goal_dim = goal_space.shape[0]

        self._max_episode_steps = self.env._max_episode_steps

    def observation(self, observation):
        # achieved goal is redundant but it is needed for HER (compute_reward)
        return np.concatenate([observation['observation'], observation['achieved_goal'], observation['desired_goal']])

    def get_obs_dict(self, obs):
        return {
            'observation': obs[:self._obs_dim],
            'achieved_goal': obs[self._obs_dim:self._obs_dim+self._goal_dim],
            'desired_goal': obs[self._obs_dim+self._goal_dim:],
        }


for reward_type, goal_dist in list(itertools.product(["sparse", "dense"], ["uniform", "compact", "single"])):
    suffix1 = "Dense" if reward_type == "dense" else ""
    if goal_dist == "compact":
        suffix2 = "Compact"
    elif goal_dist == "single":
        suffix2 = "Single"
    else:
        suffix2 = ""
    kwargs = {
        "reward_type": reward_type,
        "goal_dist": goal_dist,
    }
    
    register(
        id="FetchPickAndPlace{}{}-v2".format(suffix1, suffix2),
        entry_point=FetchPickAndPlaceEnv,
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="FetchPush{}{}-v2".format(suffix1, suffix2),
        entry_point=FetchPushEnv,
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="FetchReach{}{}-v2".format(suffix1, suffix2),
        entry_point=FetchReachEnv,
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id="FetchSlide{}{}-v2".format(suffix1, suffix2),
        entry_point=FetchSlideEnv,
        kwargs=kwargs,
        max_episode_steps=50,
    )


if __name__ == '__main__':
    env = gym.make('FetchPickAndPlaceCompact-v2')
    env = gym.make('FetchPickAndPlaceSingle-v2')
    env = gym.make('FetchPushCompact-v2')
    env = gym.make('FetchPushSingle-v2')
    env = gym.make('FetchReachCompact-v2')
    env = gym.make('FetchReachSingle-v2')
    env = gym.make('FetchSlideCompact-v2')
    env = gym.make('FetchSlideSingle-v2')