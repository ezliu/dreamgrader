import collections

import torch


# TODO: Switch from namedtuple to dataclass
class _Experience(collections.namedtuple(
        "Experience", ("state", "action", "reward", "next_state", "done", "info",
                                     "agent_state", "next_agent_state"))):
    """Defines a single (s, a, r, s')-tuple.

    Includes the agent state, as well for any agents with hidden state.
    """

class Experience:
    """Handles logic of storing cpu and cuda experiences"""
    def __init__(self, *args, **kwargs) -> None:
        experience = _Experience(*args, **kwargs)
        if not experience.state.observation.is_cuda:
            self._cpu = experience
            self._cuda = None
            self._primary_cpu = True
        else:
            self._cuda = experience
            self._cpu = None
            self._primary_cpu = False

    def __getattr__(self, attr):
        if self._primary_cpu:
            return getattr(self._cpu, attr)
        return getattr(self._cuda, attr)

    def cpu(self):
        """Returns a copy of the experience on cpu
        
        Will buffer the cpu copy in the original experience
        to accelerate future calls to this method.
        
        Returns:
            - experience: Experience on cpu
        """
        def agent_state_cpu(agent_state):
            return (None if agent_state is None else tuple( # figure out if we need to move this
                [t.cpu().pin_memory() if isinstance(t, torch.Tensor) else t for t in agent_state]))

        if self._cpu is None:
            self._cpu = self._cuda._replace(
                state=self.state.cpu(),
                next_state=self.next_state.cpu(),
                agent_state=agent_state_cpu(self.agent_state),
                next_agent_state=agent_state_cpu(self.next_agent_state)
            )
        return Experience(*self._cpu)

    def cuda(self):
        """Returns a copy of the experience on cuda
        
        Will not alter the original experience.
        
        Returns:
            - experience: Experience on cuda
        """
        def agent_state_cuda(agent_state):
            return (None if agent_state is None else tuple( # figure out if we need to move this
                [t.cuda(non_blocking=True) if isinstance(t, torch.Tensor) else t for t in agent_state]))

        experience = self._cuda
        if experience is None:
            experience = self._cpu._replace(
                state=self.state.cuda(),
                next_state=self.next_state.cuda(),
                agent_state=agent_state_cuda(self.agent_state), # figure out if we need to move this
                next_agent_state=agent_state_cuda(self.next_agent_state)
            )
        return Experience(*experience)
