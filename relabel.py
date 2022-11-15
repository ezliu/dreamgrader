import abc


class TrajectoryExperience(object):
    """An experience that holds a reference to the trajectory it came from.
    This should be substitutable wherever Experience is used.
    
    In particular, it holds:
        - trajectory (list[Experience]): the in-order trajectory that this
                experience is part of
        - index (int): the index inside of this trajectory that this experience
                is.
    """
    def __init__(self, experience, trajectory, index):
        self._experience = experience
        self._trajectory = trajectory
        self._index = index

    def __getattr__(self, attr):
        if attr[0] == "_" and attr != "_replace":
            raise AttributeError("accessing private attribute '{}'".format(attr))
        return getattr(self._experience, attr)

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def index(self):
        return self._index

    @property
    def experience(self):
        return self._experience

    def cpu(self):
        return TrajectoryExperience(self.experience.cpu(), self.trajectory, self.index)

    def cuda(self):
        return TrajectoryExperience(self.experience.cuda(), self.trajectory, self.index)

    @classmethod
    def episode_to_device(cls, episode, cpu=True):
        """Creates trajectory experiences & updates

        Makes sure experiences are on correct device

        Args:
            trajectory (List[Experience]): List of experiences to update on.
        """
        new_episode = []
        trajectory = []
        for idx, exp in enumerate(episode):
            if cpu:
                exp_on_device = exp.cpu()
            else:
                exp_on_device = exp.cuda()
            new_episode.append(exp_on_device)
            trajectory.append(TrajectoryExperience(exp_on_device, new_episode, idx))
        return trajectory


class RewardLabeler(abc.ABC):
    """Computes rewards for trajectories on the fly."""

    @abc.abstractmethod
    def label_rewards(self, trajectories):
        """Computes rewards for each experience in the trajectory.

        Args:
            trajectories (list[list[TrajectoryExperience]]): batch of
                    trajectories.

        Returns:
            rewards (torch.FloatTensor): of shape (batch_size, max_seq_len) where
                rewards[i][j] is the rewards for the experience trajectories[i][j].
                This is padded with zeros and is detached from the graph.
            distances (torch.FloatTensor): of shape (batch_size, max_seq_len + 1)
                equal to ||f(e) - g(\tau^e_{:t})|| for each t.
        """
