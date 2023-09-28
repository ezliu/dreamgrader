import abc
import math
import collections
import numpy as np
import torch
from torch import nn, Tensor
from torch import distributions as td
from torch.nn import functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from envs import bounce
from envs import grid
from envs import miniwob
from envs.miniwob.constants import QUESTIONS, PEOPLE_NAMES, LOREM_WORDS, HTML_TOKENS
import relabel
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Embedder(abc.ABC, nn.Module):
    """Defines the embedding of an object in the forward method.

    Subclasses should register to the from_config method.
    """

    def __init__(self, embed_dim):
        """Sets the embed dim.

        Args:
            embed_dim (int): the dimension of the outputted embedding.
        """
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        """Returns the dimension of the output (int)."""
        return self._embed_dim

    @classmethod
    def from_config(cls, config):
        """Constructs and returns Embedder from config.

        Args:
            config (Config): parameters for constructing the Embedder.

        Returns:
            Embedder
        """
        config_type = config.get("type")
        if config_type == "simple_grid_state":
            return SimpleGridStateEmbedder.from_config(config)
        elif config_type == "fixed_vocab":
            return FixedVocabEmbedder.from_config(config)
        elif config_type == "linear":
            return LinearEmbedder.from_config(config)
        else:
            raise ValueError("Config type {} not supported".format(config_type))


def get_state_embedder(env):
    """Returns the appropriate type of embedder given the environment type."""
    env = env.unwrapped
    if isinstance(env.unwrapped, grid.GridEnv):
        return SimpleGridStateEmbedder
    elif isinstance(env.unwrapped, bounce.BounceBinaryMetaEnv) and env.IMG_STATE:
        return BounceImageEmbedder
    elif isinstance(env.unwrapped, bounce.BounceMetaEnv):
        return BounceEmbedder
    elif isinstance(env.unwrapped, miniwob.fake_inbox_scroll_vectorized.FakeInboxScrollVectorizedMetaEnv):
        return MiniWobVectorizedEmbedder
    elif isinstance(env.unwrapped, miniwob.inbox.InboxMetaEnv) or isinstance(env.unwrapped, miniwob.fake_inbox.FakeInboxMetaEnv) or isinstance(env.unwrapped, miniwob.fake_inbox_scroll.FakeInboxScrollMetaEnv):
        return MiniWobEmbedder
    # Dependencies on OpenGL, so only load if absolutely necessary
    from envs.miniworld import sign
    if isinstance(env, sign.MiniWorldSign):
        return MiniWorldEmbedder

    raise ValueError()


class TransitionEmbedder(Embedder):
    def __init__(self, state_embedder, action_embedder, reward_embedder, embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._action_embedder = action_embedder
        self._reward_embedder = reward_embedder
        reward_embed_dim = (
                0 if reward_embedder is None else reward_embedder.embed_dim)

        self._transition_embedder = nn.Sequential(
                nn.Linear(
                    self._state_embedder.embed_dim * 2 +
                    self._action_embedder.embed_dim + reward_embed_dim,
                    128),
                nn.ReLU(),
                nn.Linear(128, embed_dim)
        )

    def forward(self, experiences):
        state_embeds = self._state_embedder(
                [exp.state.observation for exp in experiences])
        next_state_embeds = self._state_embedder(
                [exp.next_state.observation for exp in experiences])
        action_embeds = self._action_embedder([exp.action for exp in experiences])
        embeddings = [state_embeds, next_state_embeds, action_embeds]
        if self._reward_embedder is not None:
            embeddings.append(self._reward_embedder(
                    [exp.next_state.prev_reward for exp in experiences]))
        transition_embeds = self._transition_embedder(torch.cat(embeddings, -1))
        return transition_embeds

    @classmethod
    def from_config(cls, config, env):
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                config.get("experience_embedder").get("state_embed_dim"),
                config.get("use_dom"))
        action_embedder = FixedVocabEmbedder(
                env.action_space.n,
                config.get("experience_embedder").get("action_embedder").get("embed_dim"))
        return cls(state_embedder, action_embedder, config.get("embed_dim"))


class TrajectoryEmbedder(Embedder, relabel.RewardLabeler):
    def __init__(self, transition_embedder, id_embedder, penalty, embed_dim):
        super().__init__(embed_dim)

        self._transition_embedder = transition_embedder
        self._id_embedder = id_embedder
        self._transition_lstm = nn.LSTM(transition_embedder.embed_dim, 128)
        self._transition_fc_layer = nn.Linear(128, 128)
        self._transition_output_layer = nn.Linear(128, embed_dim)
        # Outputs binary prediction
        self._decoder_head = nn.Linear(embed_dim, 2)
        self._penalty = penalty
        self._use_ids = True

    def use_ids(self, use):
        self._use_ids = use

    def _compute_contexts(self, trajectories):
        """Returns contexts and masks.

        Args:
            trajectories (list[list[Experience]]): see forward().

        Returns:
            id_contexts (torch.FloatTensor): tensor of shape (batch_size)
                embedding the id's in the trajectories.
            all_transition_contexts (torch.FloatTensor): tensor of shape
                (batch_size, max_len + 1, embed_dim) embedding the sequences of states
                and actions in the trajectories.
            transition_contexts (torch.FloatTensor): tensor of shape
                (batch_size, embed_dim) equal to the last unpadded value in
                all_transition_contexts.
            mask (torch.BoolTensor): tensor of shape (batch_size, max_len + 1).
                The value is False if the trajectory_contexts value should be masked.
        """
        # trajectories: (batch_size, max_len)
        # mask: (batch_size, max_len)
        padded_trajectories, mask = utils.pad(trajectories)
        sequence_lengths = torch.tensor([len(traj) for traj in trajectories], device="cpu").long()

        # (batch_size * max_len, embed_dim)
        transition_embed = self._transition_embedder(
                [exp for traj in padded_trajectories for exp in traj])

        # Sorted only required for ONNX
        padded_transitions = nn.utils.rnn.pack_padded_sequence(
                transition_embed.reshape(mask.shape[0], mask.shape[1], -1),
                sequence_lengths, batch_first=True, enforce_sorted=False)

        transition_hidden_states = self._transition_lstm(padded_transitions)[0]
        # (batch_size, max_len, hidden_dim)
        transition_hidden_states, hidden_lengths = nn.utils.rnn.pad_packed_sequence(
                transition_hidden_states, batch_first=True)
        initial_hidden_states = torch.zeros(
                transition_hidden_states.shape[0], 1,
                transition_hidden_states.shape[-1])
        # (batch_size, max_len + 1, hidden_dim)
        transition_hidden_states = torch.cat(
                (initial_hidden_states, transition_hidden_states), 1)
        transition_hidden_states = F.relu(
                self._transition_fc_layer(transition_hidden_states))
        # (batch_size, max_len + 1, embed_dim)
        all_transition_contexts = self._transition_output_layer(
                transition_hidden_states)

        # (batch_size, 1, embed_dim)
        # Don't need to subtract 1 off of hidden_lengths as transition_contexts is
        # padded with init hidden state at the beginning.
        indices = hidden_lengths.unsqueeze(-1).unsqueeze(-1).expand(
                hidden_lengths.shape[0], 1, all_transition_contexts.shape[2]).to(
                        all_transition_contexts.device)
        transition_contexts = all_transition_contexts.gather(1, indices).squeeze(1)

        # (batch_size,)
        # HACK: This is just the env_ids now
        id_contexts = torch.tensor([traj[0].state.env_id for traj in trajectories])
        #id_contexts = self._id_embedder(
        #        torch.tensor([traj[0].state.env_id for traj in trajectories]))

        # don't mask the initial hidden states (batch_size, max_len + 1)
        mask = torch.cat(
                (torch.ones(transition_contexts.shape[0], 1).bool(), mask), -1)
        return id_contexts, all_transition_contexts, transition_contexts, mask

    def _compute_losses(
            self, trajectories, id_contexts, all_transition_contexts,
            transition_contexts, mask):
        """Computes losses based on the return values of _compute_contexts.

        Args:
            See return values of _compute_contexts.

        Returns:
            losses (dict(str: torch.FloatTensor)): see forward().
        """
        del trajectories

        # (batch_size, seq_len, 2)
        decoder_logits = self._decoder_head(all_transition_contexts)
        decoder_distribution = td.Categorical(logits=decoder_logits)
        # We want to maximize \sum_t E[log q(z | tau^exp_{:t})]
        # Repeat to be (batch_size, seq_len)
        id_contexts = id_contexts.unsqueeze(-1).expand(
                -1, all_transition_contexts.shape[1])
        # (batch_size, seq_len)
        decoder_loss = -decoder_distribution.log_prob(id_contexts)
        decoder_loss = (decoder_loss * mask).sum() / mask.sum()
        losses = {
            "decoder_loss": decoder_loss,
            # No need for info bottleneck
            #"id_context_loss": torch.max((id_contexts ** 2).sum(-1), cutoff).mean()
        }
        return losses

    def forward(self, trajectories):
        """Embeds a batch of trajectories.

        Args:
            trajectories (list[list[Experience]]): batch of trajectories, where each
                trajectory comes from the same episode.

        Returns:
            embedding (torch.FloatTensor): tensor of shape (batch_size, embed_dim)
                embedding the trajectories. This embedding is based on the ids if
                use_ids is True, otherwise based on the transitions.
            losses (dict(str: torch.FloatTensor)): maps auxiliary loss names to their
                values.
        """
        # HACK: This now just returns the decoder label predictions
        id_contexts, all_transition_contexts, transition_contexts, mask = (
                self._compute_contexts(trajectories))
        # (batch_size, 2)
        decoder_logits = self._decoder_head(transition_contexts)
        #decoder_distribution = td.Categorical(logits=decoder_logits)
        ## (batch_size)
        #decoder_predictions = decoder_distribution.sample()

        losses = self._compute_losses(
                trajectories, id_contexts, all_transition_contexts,
                transition_contexts, mask)
        #return decoder_predictions, losses
        return decoder_logits, losses

    def label_rewards(self, trajectories):
        """Computes rewards for each experience in the trajectory.

        Args:
            trajectories (list[list[Experience]]): batch of trajectories.

        Returns:
            rewards (torch.FloatTensor): of shape (batch_size, max_seq_len) where
                rewards[i][j] is the rewards for the experience trajectories[i][j].
                This is padded with zeros and is detached from the graph.
            distances (torch.FloatTensor): of shape (batch_size, max_seq_len + 1)
                equal to ||f(e) - g(\tau^e_{:t})|| for each t.
        """
        id_contexts, all_transition_contexts, _, mask = self._compute_contexts(
                trajectories)

        # Compute rewards as E[log q(y | tau_{:t + 1}) - log q(y | tau_{:t})]
        # (batch_size, seq_len, 2)
        decoder_logits = self._decoder_head(all_transition_contexts)
        decoder_distribution = td.Categorical(logits=decoder_logits)
        # (batch_size, seq_len)
        decoder_log_probs = decoder_distribution.log_prob(
                id_contexts.unsqueeze(-1).expand(-1, decoder_logits.shape[1]))
        # Add penalty
        rewards = (decoder_log_probs[:, 1:] -
                   decoder_log_probs[:, :-1] - self._penalty)

        #distances = (
        #        (all_transition_contexts - id_contexts.unsqueeze(1).expand_as(
        #         all_transition_contexts).detach()) ** 2).sum(-1)
        #rewards = distances[:, :-1] - distances[:, 1:] - self._penalty
        #return (rewards * mask[:, 1:]).detach(), distances
        return (rewards * mask[:, 1:]).detach(), decoder_log_probs.detach()


class InstructionPolicyEmbedder(Embedder):
    """Embeds (s, i, \tau^e) where:

        - s is the current state
        - i is the current instruction
        - \tau^e is an exploration trajectory (s_0, a_0, s_1, ..., s_T)
    """

    def __init__(self, trajectory_embedder, obs_embedder, instruction_embedder,
                             embed_dim):
        """Constructs around embedders for each component.

        Args:
            trajectory_embedder (TrajectoryEmbedder): embeds batches of \tau^e
                (list[list[rl.Experience]]).
            obs_embedder (Embedder): embeds batches of states s.
            instruction_embedder (Embedder): embeds batches of instructions i.
            embed_dim (int): see Embedder.
        """
        super().__init__(embed_dim)

        self._obs_embedder = obs_embedder
        self._instruction_embedder = instruction_embedder
        self._trajectory_embedder = trajectory_embedder
        self._fc_layer = nn.Linear(
                obs_embedder.embed_dim + self._trajectory_embedder.embed_dim, 256)
        self._final_layer = nn.Linear(256, embed_dim)

    def forward(self, states, hidden_state):
        # obs_embed, hidden_state = self._obs_embedder(states, hidden_state)
        trajectory_embed, _ = self._trajectory_embedder(
                [state[0].trajectory for state in states])
        # This is just the decoder prediction
        #return trajectory_embed, hidden_state
        return trajectory_embed, None

        if len(obs_embed.shape) > 2:
            trajectory_embed = trajectory_embed.unsqueeze(1).expand(
                    -1, obs_embed.shape[1], -1)

        hidden = F.relu(self._fc_layer(
                torch.cat((obs_embed, trajectory_embed), -1)))
        return self._final_layer(hidden), hidden_state

    def aux_loss(self, experiences):
        _, aux_losses = self._trajectory_embedder(
                [exp[0].state.trajectory for exp in experiences])
        return aux_losses

    @classmethod
    def from_config(cls, config, env):
        """Returns a configured InstructionPolicyEmbedder.

        Args:
            config (Config): see Embedder.from_config.
            env (gym.Wrapper): the environment to run on. Expects this to be wrapped
                with an InstructionWrapper.

        Returns:
            InstructionPolicyEmbedder: configured according to config.
        """
        obs_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                config.get("obs_embedder").get("embed_dim"),
                config.get("use_dom"))
        # Use SimpleGridEmbeder since these are just discrete vars
        instruction_embedder = SimpleGridStateEmbedder(
                env.observation_space["instructions"],
                config.get("instruction_embedder").get("embed_dim"))
        # Exploitation recurrence is not observing the rewards
        exp_embedder = ExperienceEmbedder(
                obs_embedder, instruction_embedder, None, None, None,
                obs_embedder.embed_dim)
        obs_embedder = RecurrentStateEmbedder(exp_embedder, obs_embedder.embed_dim)

        transition_config = config.get("transition_embedder")
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                transition_config.get("state_embed_dim"),
                config.get("use_dom"))
        # This needs to cover embedding of the exploration time env...
        action_embedder = FixedVocabEmbedder(
                env.unwrapped.action_space.n, transition_config.get("action_embed_dim"))
        reward_embedder = None
        if transition_config.get("reward_embed_dim") is not None:
            reward_embedder = LinearEmbedder(
                    1, transition_config.get("reward_embed_dim"))
        transition_embedder = TransitionEmbedder(
                state_embedder, action_embedder, reward_embedder,
                transition_config.get("embed_dim"))
        id_embedder = IDEmbedder(
                env.observation_space["env_id"].high,
                config.get("transition_embedder").get("embed_dim"))
        if config.get("trajectory_embedder").get("type") == "ours":
            trajectory_embedder = TrajectoryEmbedder(
                    transition_embedder, id_embedder,
                    config.get("trajectory_embedder").get("penalty"),
                    transition_embedder.embed_dim)
        else:
            raise ValueError("Unsupported trajectory embedder {}".format(
                config.get("trajectory_embedder")))
        return cls(trajectory_embedder, obs_embedder, instruction_embedder,
                             config.get("embed_dim"))


class RecurrentAndTaskIDEmbedder(Embedder):
    """Embedding used by IMPORT.

    Compute both:
        - g(\tau_{:t}) recurrently
        - f(e)

    Full embedding is:
        \phi(s_t, z), where z is randomly chosen from g(\tau_{:t}) and f(e).
    """

    def __init__(
            self, recurrent_state_embedder, id_embedder, state_embedder, embed_dim):
        super().__init__(embed_dim)
        assert id_embedder.embed_dim == recurrent_state_embedder.embed_dim
        self._recurrent_state_embedder = recurrent_state_embedder
        self._id_embedder = id_embedder
        self._state_embedder = state_embedder
        self._final_layer = nn.Linear(
                id_embedder.embed_dim + state_embedder.embed_dim, embed_dim)
        self._use_id = False

    def use_ids(self, use):
        self._use_id = use

    def _compute_embeddings(self, states, hidden_state=None):
        # (batch_size, seq_len, embed_dim)
        recurrent_embedding, hidden_state = self._recurrent_state_embedder(
                states, hidden_state)
        # (batch_size, embed_dim)
        id_embedding = self._id_embedder(
                torch.tensor([seq[0].env_id for seq in states]))

        if len(recurrent_embedding.shape) > 2:
            id_embedding = id_embedding.unsqueeze(1).expand_as(recurrent_embedding)
        return recurrent_embedding, id_embedding, hidden_state

    def forward(self, states, hidden_state=None):
        recurrent_embedding, id_embedding, hidden_state = self._compute_embeddings(
                states, hidden_state)

        history_embed = recurrent_embedding
        if self._use_id:
            history_embed = id_embedding

        # (batch_size, seq_len, state_embed_dim) or (batch_size, state_embed_dim)
        state_embeds = self._state_embedder(
                [state for seq in states for state in seq])
        if len(history_embed.shape) > 2:
            state_embeds = state_embeds.reshape(
                    history_embed.shape[0], history_embed.shape[1], -1)
        return self._final_layer(
                F.relu(torch.cat((history_embed, state_embeds), -1))), hidden_state

    def aux_loss(self, trajectories):
        # (batch_size, max_seq_len)
        trajectories, mask = utils.pad(trajectories)

        # (batch_size, max_seq_len, embed_dim)
        recurrent_embeddings, id_embeddings, hidden_state = self._compute_embeddings(
                [[exp.state for exp in traj] for traj in trajectories],
                [traj[0].agent_state for traj in trajectories])

        return {
            "embedding_distance": (
                    ((recurrent_embeddings - id_embeddings.detach()) ** 2)
                    .mean(0).sum())
        }

    @classmethod
    def from_config(cls, config, env):
        recurrent_state_embedder = RecurrentStateEmbedder.from_config(
                config.get("recurrent_embedder"), env)
        state_embed_config = config.get("state_embedder")
        state_embedder = get_state_embedder(env)(
            env.observation_space["observation"],
            state_embed_config.get("embed_dim"),
            config.get("use_dom"))
        instruction_embedder = SimpleGridStateEmbedder(
            env.observation_space["instructions"],
            state_embed_config.get("embed_dim"))
        state_embedder = StateInstructionEmbedder(
                state_embedder, instruction_embedder,
                state_embed_config.get("embed_dim"))

        id_embed_config = config.get("id_embedder")
        id_embedder = IDEmbedder(
                env.observation_space["env_id"].high,
                id_embed_config.get("embed_dim"))
        return cls(
                recurrent_state_embedder, id_embedder, state_embedder,
                config.get("embed_dim"))


class VariBADEmbedder(Embedder):
    """Embedding used by VariBAD.

    Computes:
        - g(\tau_{:t}) recurrently and applies fully connected heads on top to
            produce q(z_t | \tau_{:t}) = N(head1(g(\tau_{:t})), head2(g(\tau_{:t})))
        - embedding = \phi(z_t.detach(), embed(s_t))

    Decoding auxiliary loss:
        - \sum_t \sum_i ||decoder(z_i, e(s_t), e(a_t)) - r_t||_2^2
        - \sum_t \sum_i ||decoder(z_i, e(s_t), e(a_t)) - s_{t + 1}||_2^2
    """

    def __init__(
            self, recurrent_state_embedder, z_dim, state_embedder, action_embedder,
            state_dim, embed_dim, predict_state=True):
        super().__init__(embed_dim)
        self._recurrent_state_embedder = recurrent_state_embedder
        self._fc_mu = nn.Linear(recurrent_state_embedder.embed_dim, z_dim)
        self._fc_logvar = nn.Linear(recurrent_state_embedder.embed_dim, z_dim)
        self._state_embedder = state_embedder
        self._phi = nn.Linear(
                z_dim + state_embedder.embed_dim, embed_dim)
        self._action_embedder = action_embedder
        self._decoder = nn.Sequential(
            nn.Linear(z_dim + state_embedder.embed_dim + action_embedder.embed_dim,
                                128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Predicts reward / state
        self._reward_head = nn.Linear(128, 1)
        self._state_head = nn.Linear(128, state_dim)

        # If False, does not do state prediction
        self._predict_state = predict_state
        self._z_dim = z_dim

    def _compute_z_distr(self, states, hidden_state=None):
        embeddings, hidden_state = self._recurrent_state_embedder(
                states, hidden_state=hidden_state)

        # (batch_size, sequence_length, embed_dim)
        mu = embeddings
        std = torch.ones_like(mu) * 1e-6

        q = td.Independent(td.Normal(mu, std), 1)
        return q, hidden_state

    def forward(self, states, hidden_state=None):
        q, hidden_state = self._compute_z_distr(states, hidden_state)
        # Don't backprop through encoder
        z = q.rsample()

        # (batch_size, seq_len, state_embed_dim) or (batch_size, state_embed_dim)
        state_embeds = self._state_embedder(
                [state for seq in states for state in seq])
        if len(z.shape) > 2:
            state_embeds = state_embeds.reshape(z.shape[0], z.shape[1], -1)
        return self._phi(F.relu(torch.cat((z, state_embeds), -1))), hidden_state

    def aux_loss(self, trajectories):
        # The trajectories that we will try to decode
        # (batch_size, max_trajectory_len)
        trajectories_to_predict, predict_mask = utils.pad(
                [traj[0].trajectory for traj in trajectories])

        # The trajectories we're using to encode z
        # They differ when we sample not the full trajectory
        # (batch_size, max_sequence_len)
        padded_trajectories, mask = utils.pad(trajectories)

        q = self._compute_z_distr(
                [[exp.state for exp in traj] for traj in padded_trajectories],
                [traj[0].agent_state for traj in padded_trajectories])[0]
        # (batch_size, max_sequence_len, z_dim)
        z = q.rsample()
        # (batch_size, max_trajectory_len, max_sequence_len, z_dim)
        z = z.unsqueeze(1).expand(-1, predict_mask.shape[1], -1, -1)

        # (batch_size, max_trajectory_len, embed_dim)
        # e(s)
        state_embeds = self._state_embedder(
                [exp.state for trajectory in trajectories_to_predict
                 for exp in trajectory]).reshape(z.shape[0], z.shape[1], -1)
        # e(a)
        action_embeds = self._action_embedder(
                [exp.action for trajectory in trajectories_to_predict
                 for exp in trajectory]).reshape(z.shape[0], z.shape[1], -1)

        # (batch_size, max_trajectory_len, max_sequence_len, embed_dim)
        state_embeds = state_embeds.unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        action_embeds = action_embeds.unsqueeze(2).expand(-1, -1, z.shape[2], -1)

        decoder_input = torch.cat((z, state_embeds, action_embeds), -1)
        decoder_embed = self._decoder(decoder_input)

        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        predicted_rewards = self._reward_head(F.relu(decoder_embed))

        # (batch_size, max_trajectory_len)
        true_rewards = torch.tensor(
                [[exp.next_state.prev_reward for exp in trajectory]
                 for trajectory in trajectories_to_predict])

        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        true_rewards = true_rewards.unsqueeze(-1).unsqueeze(-1).expand_as(
                predicted_rewards)

        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        reward_decoding_loss = ((predicted_rewards - true_rewards) ** 2)

        predict_mask = predict_mask.unsqueeze(2).expand(-1, -1, mask.shape[-1])
        mask = mask.unsqueeze(1).expand_as(predict_mask)
        # (batch_size, max_trajectory_len, max_sequence_len, 1)
        aggregate_mask = (predict_mask * mask).unsqueeze(-1)
        reward_decoding_loss = ((reward_decoding_loss * aggregate_mask).sum() /
                                                        reward_decoding_loss.shape[0])

        state_decoding_loss = torch.tensor(0).float()
        if self._predict_state:
            # (batch_size, max_trajectory_len, max_sequence_len, state_dim)
            predicted_states = self._state_head(F.relu(decoder_embed))

            # (batch_size, max_trajectory_len, state_dim)
            next_states_to_predict = torch.stack(
                    [torch.stack([exp.next_state.observation for exp in trajectory])
                     for trajectory in trajectories_to_predict])

            # (batch_size, max_trajectory_len, max_sequence_len, state_dim)
            next_states_to_predict = next_states_to_predict.unsqueeze(2).expand_as(
                    predicted_states)

            # (batch_size, max_trajectory_len, max_sequence_len, state_dim)
            state_decoding_loss = ((predicted_states - next_states_to_predict) ** 2)
            state_decoding_loss = ((state_decoding_loss * aggregate_mask).sum() /
                                                         state_decoding_loss.shape[0])

        #kl_loss = td.kl_divergence(q, self._prior(mask.shape[0], mask.shape[1]))
        return {
            "reward_decoding_loss": reward_decoding_loss,
            "state_decoding_loss": state_decoding_loss * 0.01,
            #"kl_loss": kl_loss * 0.1,
        }

    def _prior(self, batch_size, sequence_len):
        mu = torch.zeros(batch_size, sequence_len, self._z_dim)
        std = torch.ones_like(mu)
        return td.Independent(td.Normal(mu, std), 1)

    @classmethod
    def from_config(cls, config, env):
        recurrent_state_embedder = RecurrentStateEmbedder.from_config(
                config.get("recurrent_embedder"), env)
        state_embed_config = config.get("state_embedder")
        state_embedder = get_state_embedder(env)(
            env.observation_space["observation"],
            state_embed_config.get("embed_dim"),
            config.get("use_dom"))
        instruction_embedder = SimpleGridStateEmbedder(
            env.observation_space["instructions"],
            state_embed_config.get("embed_dim"))
        state_embedder = StateInstructionEmbedder(
                state_embedder, instruction_embedder,
                state_embed_config.get("embed_dim"))

        action_embed_config = config.get("action_embedder")
        action_embedder = FixedVocabEmbedder(
                env.action_space.n, action_embed_config.get("embed_dim"))
        state_dim = len(env.observation_space["observation"].high)
        return cls(
                recurrent_state_embedder, config.get("z_dim"), state_embedder,
                action_embedder, state_dim, config.get("embed_dim"),
                config.get("predict_states"))


class RecurrentStateEmbedder(Embedder):
    """Applies an LSTM on top of a state embedding."""

    def __init__(self, state_embedder, embed_dim):
        super().__init__(embed_dim)

        self._state_embedder = state_embedder
        self._lstm_cell = nn.LSTMCell(state_embedder.embed_dim, embed_dim)

    def forward(self, states, hidden_state=None):
        """Embeds a batch of sequences of contiguous states.

        Args:
            states (list[list[np.array]]): of shape
                (batch_size, sequence_length, state_dim).
            hidden_state (list[object] | None): batch of initial hidden states
                to use with the LSTM. During inference, this should just be the
                previously returned hidden state.

        Returns:
            embedding (torch.tensor): shape (batch_size, sequence_length, embed_dim)
            hidden_state (object): hidden state after embedding every element in the
                sequence.
        """
        batch_size = len(states)
        sequence_len = len(states[0])

        # Stack batched hidden state
        if batch_size > 1 and hidden_state is not None:
            hs = []
            cs = []
            for hidden in hidden_state:
                if hidden is None:
                    hs.append(torch.zeros(1, self.embed_dim))
                    cs.append(torch.zeros(1, self.embed_dim))
                else:
                    hs.append(hidden[0])
                    cs.append(hidden[1])
            hidden_state = (torch.cat(hs, 0), torch.cat(cs, 0))

        flattened = [state for seq in states for state in seq]

        # (batch_size * sequence_len, embed_dim)
        state_embeds = self._state_embedder(flattened)
        state_embeds = state_embeds.reshape(batch_size, sequence_len, -1)
        embeddings = []
        for seq_index in range(sequence_len):
            hidden_state = self._lstm_cell(
                    state_embeds[:, seq_index, :], hidden_state)

            # (batch_size, 1, embed_dim)
            embeddings.append(hidden_state[0].unsqueeze(1))

        # (batch_size, sequence_len, embed_dim)
        # squeezed to (batch_size, embed_dim) if sequence_len == 1
        embeddings = torch.cat(embeddings, 1).squeeze(1)

        # Detach to save GPU memory.
        detached_hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        return embeddings, detached_hidden_state

    @classmethod
    def from_config(cls, config, env):
        experience_embed_config = config.get("experience_embedder")
        state_embedder = get_state_embedder(env)(
                env.observation_space["observation"],
                experience_embed_config.get("state_embed_dim"),
                config.get("use_dom"))
        action_embedder = FixedVocabEmbedder(
                env.action_space.n + 1, experience_embed_config.get("action_embed_dim"))
        instruction_embedder = None
        if experience_embed_config.get("instruction_embed_dim") is not None:
            # Use SimpleGridEmbedder since these are just discrete vars
            instruction_embedder = SimpleGridStateEmbedder(
                    env.observation_space["instructions"],
                    experience_embed_config.get("instruction_embed_dim"))

        reward_embedder = None
        if experience_embed_config.get("reward_embed_dim") is not None:
            reward_embedder = LinearEmbedder(
                    1, experience_embed_config.get("reward_embed_dim"))

        done_embedder = None
        if experience_embed_config.get("done_embed_dim") is not None:
            done_embedder = FixedVocabEmbedder(
                    2, experience_embed_config.get("done_embed_dim"))

        experience_embedder = ExperienceEmbedder(
                state_embedder, instruction_embedder, action_embedder,
                reward_embedder, done_embedder,
                experience_embed_config.get("embed_dim"))
        return cls(experience_embedder, config.get("embed_dim"))


class StateInstructionEmbedder(Embedder):
    """Embeds instructions and states and applies a linear layer on top."""

    def __init__(self, state_embedder, instruction_embedder, embed_dim):
        super().__init__(embed_dim)
        self._state_embedder = state_embedder
        self._instruction_embedder = instruction_embedder
        if instruction_embedder is not None:
            self._final_layer = nn.Linear(
                    state_embedder.embed_dim + instruction_embedder.embed_dim, embed_dim)
            assert self._state_embedder.embed_dim == embed_dim

    def forward(self, states):
        state_embeds = self._state_embedder([state.observation for state in states])
        if self._instruction_embedder is not None:
            instruction_embeds = self._instruction_embedder(
                    [torch.tensor(state.instructions) for state in states])
            return self._final_layer(
                    F.relu(torch.cat((state_embeds, instruction_embeds), -1)))
        return state_embeds


def init(module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


class Flatten(nn.Module):
        def forward(self, x):
                return x.view(x.size(0), -1)


class MiniWorldEmbedder(Embedder):
    """Embeds 80x60 MiniWorld inputs.

    Network taken from gym-miniworld/.
    """
    def __init__(self, observation_space, embed_dim):
        super().__init__(embed_dim)

        # Architecture from gym-miniworld
        # For 80x60 input
        num_inputs = observation_space.shape[0]

        self._network = nn.Sequential(
                nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2),
                nn.ReLU(),

                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.ReLU(),

                nn.Conv2d(32, 32, kernel_size=4, stride=2),
                nn.ReLU(),

                Flatten(),

                nn.Linear(32 * 7 * 5, embed_dim),
        )

    def forward(self, obs):
        # (batch_size, 80, 60, 3)
        tensor = torch.stack(obs) / 255.
        return self._network(tensor)


class BounceEmbedder(Embedder):
    def __init__(self, observation_space, embed_dim):
        super().__init__(embed_dim)

        hidden_size = 128
        self._network = nn.Sequential(
                nn.Linear(observation_space.shape[0], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, embed_dim),
        )

    def forward(self, obs):
        obs = torch.stack(obs).float()
        return self._network(obs)


class BounceImageEmbedder(Embedder):
    """Embeds 84x84x1 bounce images."""
    def __init__(self, observation_space, embed_dim):
        super().__init__(embed_dim)

        # Architecture from DQN
        # For 84x84 input

        self._network = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, padding=(1, 1), kernel_size=3, stride=1),
                nn.Flatten(),
                nn.Linear(10 * 10 * 64, embed_dim))

    def forward(self, obs):
        # (batch_size, 80, 60, 3)
        tensor = torch.stack(obs) / 255.
        # This is hacking in the channel dim
        tensor = tensor.permute(0, 3, 2, 1)
        return self._network(tensor)


class MiniWobScreenshotEmbedder(Embedder):
    "Embeds screenshots using architecture from https://proceedings.mlr.press/v162/humphreys22a/humphreys22a.pdf"

    def __init__(self, observation_space, embed_dim=512):
        super().__init__(embed_dim)

        self._network = nn.Sequential(
                Residual(1, 32, use_1x1conv=True, strides=2),
                Residual(32, 128, use_1x1conv=True, strides=2),
                Residual(128, 256, use_1x1conv=True, strides=2),
                Residual(256, embed_dim, use_1x1conv=True, strides=2)
        )
        # self.load_state_dict(torch.load("screenshot_encoder_2023-08-04T02:46:37.666328.pth"))

    def forward(self, obs):
        # (batch_size, 80, 60, 3)
        tensor = obs / 255.
        # This is hacking in the channel dim
        tensor = tensor.permute(0, 3, 2, 1)
        result = self._network(tensor)
        return result.reshape(result.shape[0], self._embed_dim, -1).permute(0, 2, 1)


class Residual(nn.Module):  #@save
    """The Residual block of ResNet models. From https://d2l.ai/chapter_convolutional-modern/resnet.html"""
    def __init__(self, num_in, num_out, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                                   stride=strides)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(num_out, num_out, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if use_1x1conv:
            self.conv3 = nn.Conv2d(num_in, num_out, kernel_size=1,
                                       stride=strides)
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class MiniWobLanguageEmbedder(Embedder):
    d_vocab = 128
    
    def __init__(self, observation_space, embed_dim=256):
        super().__init__(embed_dim)

        self.tokenizer = get_tokenizer('basic_english')
        phrases = QUESTIONS + [" ".join(LOREM_WORDS), " ".join(PEOPLE_NAMES), "."]
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, phrases), specials=["<unk>", "<pad>", "<bos>"])
        for t in HTML_TOKENS:
            if t not in self.vocab:
                self.vocab.append_token(t)
               
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.embed = nn.Embedding(len(self.vocab), self.d_vocab).cuda()
        print(self.vocab["<unk>"])
        

    def forward(self, obs):
        """Expects shape (batch_size, 1)"""
        obs = [torch.tensor([self.vocab["<bos>"]] + self.vocab(self.tokenizer(item))) for item in obs]
        # Pad to max length
        obs = nn.utils.rnn.pad_sequence(obs, batch_first=True, padding_value=self.vocab["<pad>"]).to(device)
        # Generate padding mask
        src_pad_mask = (obs == self.vocab["<pad>"]).to(device)
        obs = obs.permute(1, 0)
        embeddings = self.embed(obs)
        return embeddings, src_pad_mask


class MiniWobLanguageTransformer(Embedder):
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout_prob = 0.1  # dropout probability
    d_vocab = 128

    def __init__(self, observation_space, embed_dim=256):
        super().__init__(embed_dim)

        self.query = nn.Parameter(torch.randn((1, embed_dim)))
        self.pos_encoding = PositionalEncoding(self.d_vocab)
        self.encoder = nn.MultiheadAttention(
            embed_dim, 
            self.nhead,
            dropout=self.dropout_prob,
            kdim=self.d_vocab, 
            vdim=self.d_vocab,
            batch_first=False)
        self.norm1 = nn.LayerNorm(self.d_vocab, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        
    def forward(self, obs, query=None, pad_mask=None):
        if query is None:
            query = torch.repeat_interleave(self.query.unsqueeze(0), obs.shape[1], dim=1)
        else:
            # expected input to be (batch, 1, embed)
            query = query.permute(1, 0, 2)
        # obs = self.pos_encoding(obs)
        obs = self.norm1(obs)
        query = self.norm3(query)
        embeddings = query + self.encoder(query, obs, obs, key_padding_mask=pad_mask)[0]
        embeddings = embeddings.reshape(obs.shape[1], -1)
        embeddings = embeddings + self.dropout(self.fc1(self.norm2(embeddings)))
        return embeddings


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

"""class MiniWobQuestionEmbedder(Embedder):
    d_hid = 128  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    transf_embed_dim = 64
    
    def __init__(self, observation_space, embed_dim=256, use_dom=False):
        super().__init__(embed_dim)

        self.tokenizer = get_tokenizer('basic_english')
        phrases = QUESTIONS + [" ".join(LOREM_WORDS), " ".join(PEOPLE_NAMES), "."]
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, phrases), specials=["<unk>", "<pad>", "<bos>"])
        if use_dom:
            for t in HTML_TOKENS:
                if t not in self.vocab:
                    self.vocab.append_token(t)
               
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.model = TransformerEmbedder(len(self.vocab),self.transf_embed_dim, self.nhead, self.d_hid, self.nlayers, self.dropout)
        self.output_proj = nn.Linear(self.transf_embed_dim, embed_dim)

    def forward(self, obs):
        # Expects shape (batch_size, 1)
        obs = [torch.tensor([self.vocab["<bos>"]] + self.vocab(self.tokenizer(item))) for item in obs]
        # Pad to max length
        obs = nn.utils.rnn.pad_sequence(obs, batch_first=True, padding_value=self.vocab["<pad>"]).to(device)
        # Generate padding mask
        src_pad_mask = (obs == self.vocab["<pad>"]).to(device)
        obs = obs.permute(1, 0)
        src_mask = self.model.generate_square_subsequent_mask(len(obs)).to(device)
        embeddings = self.model(obs, src_mask, src_pad_mask)

        # Mean pool while taking into account mask
        attn_mask = ~src_pad_mask.bool()
        num_tokens = attn_mask.sum(axis=-1).unsqueeze(-1)
        sum = (embeddings.permute(1, 0, 2) * attn_mask.unsqueeze(-1)).sum(axis=1)
        pooled_embedding = sum / num_tokens
        return F.relu(self.output_proj(pooled_embedding))"""


"""class TransformerEmbedder(nn.Module):
    # Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor, src_pad_mask: Tensor = None) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)"""


class MiniWobVectorizedEmbedder(Embedder):
    def __init__(self, observation_space, embed_dim, use_dom=False):
        super().__init__(embed_dim)

        hidden_size = 128
        self.state_embedder = nn.Sequential(
                nn.Linear(observation_space.feature_space["screenshot"].shape[0], hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, embed_dim),
        )
        self.instruction_embedder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, obs):

        if isinstance(obs, list):
            question = [o.question for o in obs]
            dom = [o.dom for o in obs]
            screenshot = torch.stack([o.screenshot for o in obs])
        else:
            question = [obs.question]
            dom = [obs.dom]
            screenshot = obs.screenshot.unsqueeze(0)
        
        # Check batch size
        assert len(question) == screenshot.shape[0], "Batch size mismatch"
        B = len(question)

        tensor_questions = []
        for q in question:
            format_number = lambda num: '1st' if num == 0 else '2nd' if num == 1 else '3rd' if num == 2 else f'{num+1}th'
            tensor_inputs = [1 if format_number(i) in q else 0 for i in range(7)] + [1 if i in q else 0 for i in ['small', 'medium', 'large']]
            tensor_questions.append(tensor_inputs)

        tensor_questions = torch.FloatTensor(tensor_questions).to(device)
        tensor_screenshot = screenshot.float().to(device)
        question_embedding = self.instruction_embedder(tensor_questions)
        state_embedding = self.state_embedder(tensor_screenshot)
        res = torch.cat([question_embedding, state_embedding], dim=1).reshape(B, -1)
        res = self.linear(res)
        return res


class MiniWobEmbedder(Embedder):
    # nlayers = 8
    # nhead = 8
    nlayers = 2 #6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # number of heads in nn.MultiheadAttention
    dropout = 0.1  # dropout probability
    
    def __init__(self, observation_space, embed_dim=256, use_dom=False):
        super().__init__(embed_dim)

        # self.language_embedder = MiniWobLanguageEmbedder(None, embed_dim=embed_dim)
        # self.question_embedder = MiniWobLanguageTransformer(None, embed_dim=embed_dim)
        # self.dom_embedder = MiniWobLanguageTransformer(None, embed_dim=embed_dim)
        self.instruction_embedder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.screenshot_embedder = MiniWobScreenshotEmbedder(None, embed_dim=embed_dim)
        self.extra_embedding1 = nn.Parameter(torch.randn((1, embed_dim)))
        self.extra_embedding2 = nn.Parameter(torch.randn((1, embed_dim)))
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, self.nhead, embed_dim, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.nlayers)
        self.linear = nn.Linear((3 if use_dom else 2) * embed_dim, embed_dim)
        self.use_dom = use_dom

        # self.load_state_dict(torch.load("font_size_embedder.pth"))

    def forward(self, obs):
        if isinstance(obs, list):
            question = [o.question for o in obs]
            dom = [o.dom for o in obs]
            screenshot = torch.stack([o.screenshot for o in obs])
        else:
            question = [obs.question]
            dom = [obs.dom]
            screenshot = obs.screenshot.unsqueeze(0)
        
        # Check batch size
        assert len(question) == screenshot.shape[0], "Batch size mismatch"
        B = len(question)

        """question_embedding, question_pad_mask = self.language_embedder(question)
        question_embedding = self.question_embedder(question_embedding, pad_mask=question_pad_mask).unsqueeze(1)
        dom_embedding = None
        if self.use_dom:
            dom_token_embedding, dom_pad_mask = self.language_embedder(dom)
            dom_embedding = self.dom_embedder(dom_token_embedding, query=question_embedding, pad_mask=dom_pad_mask).unsqueeze(1)"""

        # Process vectorize question
        tensor_questions = []
        for q in question:
            format_number = lambda num: '1st' if num == 0 else '2nd' if num == 1 else '3rd' if num == 2 else f'{num+1}th'
            tensor_inputs = [1 if format_number(i) in q else 0 for i in range(10)] + [1 if i in q else 0 for i in ['small', 'medium', 'large']]
            tensor_questions.append(tensor_inputs)

        tensor_questions = torch.FloatTensor(tensor_questions).to(device)
        question_embedding = self.instruction_embedder(tensor_questions).unsqueeze(1)
        screenshot_embedding = self.screenshot_embedder(screenshot)
        extra_emb1 = torch.repeat_interleave(self.extra_embedding1, B, dim=0).unsqueeze(1)
        extra_emb2 = torch.repeat_interleave(self.extra_embedding2, B, dim=0).unsqueeze(1)
        multi_embedding = torch.cat([
            question_embedding,
            screenshot_embedding,
        ] + ([dom_embedding] if self.use_dom else [])
        + [extra_emb1, extra_emb2], dim=1)
        multi_embedding = multi_embedding.permute(1, 0, 2)
        multi_embedding = self.transformer_encoder(multi_embedding)
        multi_embedding = multi_embedding.permute(1, 0, 2)
        res = torch.concat(([dom_embedding.squeeze(1)] if self.use_dom else []) + [
            multi_embedding[:,-2,:],
            multi_embedding[:,-1,:],
        ], dim=1).reshape(B, -1)
        res = self.linear(res)
        return res


class SimpleGridStateEmbedder(Embedder):
    """Embedder for SimpleGridEnv states.

    Concretely, embeds (x, y) separately with different embeddings for each cell.
    """

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (spaces.Box): limits for the observations to embed.
        """
        super().__init__(embed_dim)

        assert all(dim == 0 for dim in observation_space.low)
        assert observation_space.dtype == np.int

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim, hidden_size) for dim in observation_space.high])
        self._fc_layer = nn.Linear(hidden_size * len(observation_space.high), 256)
        self._final_fc_layer = nn.Linear(256, embed_dim)

    def forward(self, obs):
        tensor = torch.stack(obs)
        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        return self._final_fc_layer(F.relu(self._fc_layer(torch.cat(embeds, -1))))


class IDEmbedder(Embedder):
    """Embeds N-dim IDs by embedding each component and applying a linear
    layer."""

    def __init__(self, observation_space, embed_dim):
        """Constructs for SimpleGridEnv.

        Args:
            observation_space (np.array): discrete max limits for each dimension of the
                state (expects min is 0).
        """
        super().__init__(embed_dim)

        hidden_size = 32
        self._embedders = nn.ModuleList(
                [nn.Embedding(dim, hidden_size) for dim in observation_space])
        self._fc_layer = nn.Linear(hidden_size * len(observation_space), embed_dim)

    @classmethod
    def from_config(cls, config, observation_space):
        return cls(observation_space, config.get("embed_dim"))

    def forward(self, obs):
        tensor = obs
        if len(tensor.shape) == 1:  # 1-d IDs
            tensor = tensor.unsqueeze(-1)

        embeds = []
        for i in range(tensor.shape[1]):
            embeds.append(self._embedders[i](tensor[:, i]))
        return self._fc_layer(torch.cat(embeds, -1))


class FixedVocabEmbedder(Embedder):
    """Wrapper around nn.Embedding obeying the Embedder interface."""

    def __init__(self, vocab_size, embed_dim):
        """Constructs.

        Args:
            vocab_size (int): number of unique embeddings.
            embed_dim (int): dimension of output embedding.
        """
        super().__init__(embed_dim)

        self._embedder = nn.Embedding(vocab_size, embed_dim)

    @classmethod
    def from_config(cls, config):
        return cls(config.get("vocab_size"), config.get("embed_dim"))

    def forward(self, inputs):
        """Embeds inputs according to the underlying nn.Embedding.

        Args:
            inputs (list[int]): list of inputs of length batch.

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        tensor_inputs = torch.tensor(np.stack(inputs)).long()
        return self._embedder(tensor_inputs)


class LinearEmbedder(Embedder):
    """Wrapper around nn.Linear obeying the Embedder interface."""

    def __init__(self, input_dim, embed_dim):
        """Wraps a nn.Linear(input_dim, embed_dim).

        Args:
            input_dim (int): dimension of inputs to embed.
            embed_dim (int): dimension of output embedding.
        """
        super().__init__(embed_dim)

        self._embedder = nn.Linear(input_dim, embed_dim)

    @classmethod
    def from_config(cls, config):
        return cls(config.get("input_dim"), config.get("embed_dim"))

    def forward(self, inputs):
        """Embeds inputs according to the underlying nn.Linear.

        Args:
            inputs (list[np.array]): list of inputs of length batch.
                Each input is an array of shape (input_dim).

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        inputs = np.stack(inputs)
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, 1)
        tensor_inputs = torch.tensor(inputs).float()
        return self._embedder(tensor_inputs)


class ExperienceEmbedder(Embedder):
    """Optionally embeds each of:

        - state s
        - instructions i
        - actions a
        - rewards r
        - done d

    Then passes a single linear layer over their concatenation.
    """

    def __init__(self, state_embedder, instruction_embedder, action_embedder,
                             reward_embedder, done_embedder, embed_dim):
        """Constructs.

        Args:
            state_embedder (Embedder | None)
            instruction_embedder (Embedder | None)
            action_embedder (Embedder | None)
            reward_embedder (Embedder | None)
            done_embedder (Embedder | None)
            embed_dim (int): dimension of the output
        """
        super().__init__(embed_dim)

        self._embedders = collections.OrderedDict()
        if state_embedder is not None:
            self._embedders["state"] = state_embedder
        if instruction_embedder is not None:
            self._embedders["instruction"] = instruction_embedder
        if action_embedder is not None:
            self._embedders["action"] = action_embedder
        if reward_embedder is not None:
            self._embedders["reward"] = reward_embedder
        if done_embedder is not None:
            self._embedders["done"] = done_embedder

        # Register the embedders so they get gradients
        self._register_embedders = nn.ModuleList(self._embedders.values())
        self._final_layer = nn.Linear(
                sum(embedder.embed_dim for embedder in self._embedders.values()),
                embed_dim)

    def forward(self, instruction_states):
        """Embeds the components for which this has embedders.

        Args:
            instruction_states (list[InstructionState]): batch of states.

        Returns:
            embedding (torch.Tensor): of shape (batch, embed_dim)
        """
        def get_inputs(key, states):
            if key == "state":
                return [state.observation for state in states]
            elif key == "instruction":
                return [torch.tensor(state.instructions) for state in states]
            elif key == "action":
                actions = np.array(
                        [state.prev_action if state.prev_action is not None else -1
                         for state in states])
                return actions + 1
            elif key == "reward":
                return [state.prev_reward for state in states]
            elif key == "done":
                return [state.done for state in states]
            else:
                raise ValueError("Unsupported key: {}".format(key))

        embeddings = []
        for key, embedder in self._embedders.items():
            inputs = get_inputs(key, instruction_states)
            embeddings.append(embedder(inputs))
        return self._final_layer(F.relu(torch.cat(embeddings, -1)))
