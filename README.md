# Giving Feedback on Interactive Student Programs with Meta-Exploration
## Introduction

Providing near-human level accuracy on grading interactive student programs with meta-reinforcmeent learning.

[Evan Zheran Liu\*](https://cs.stanford.edu/~evanliu/), [Moritz Stephan\*](https://www.linkedin.com/in/moritz-stephan/), [Allen Nie](https://anie.me/about), [Chris Piech](https://stanford.edu/~cpiech/bio/index.html), [Emma Brunskill](https://cs.stanford.edu/people/ebrun/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/)\
Advances in Neural Information Processing Systems (NeurIPS), 2022. **Selected as an Oral**.

Also see our [project web page](https://ezliu.github.io/dreamgrader/) and [paper](https://arxiv.org/abs/2211.08802).

## Requirements

This code requires Python3.6+.
The Python3 requirements are specified in `requirements.txt`.
We recommend creating a `virtualenv`, e.g.:

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

To train a DreamGrader policy for a particular error type, invoke the following
command:

```
python3 main.py {exp_name} -b instruction_agent.policy.type=\"classifier\" -c configs/default.json -c configs/bounce.json -c configs/bounce_bigger_train.json -b environment.error_type=\"{error_type}\" -s {seed}
```

This will create a directory `experiments/exp_name`, which will contain:

- A tensorboard subdirectory at `experiments/exp_name/tensorboard`, which logs
  statistics, such as accumulated returns vs. number of training episodes, and
  also vs. number of training steps.
- A visualization subdirectory at `experiments/exp_name/visualize`, which will
  contain videos of the learned agent. The videos prefixed with `exploration`
  show the learned exploration behavior, while the `instruction` videos show
  the predicted bug label, but are not particularly informative.
- A checkpoints subdirectory at `experiments/exp_name/checkpoints`, which will
  periodically save model checkpoints.
- Metadata about the run, such as the configs used.

The `error_type` argument specifies which error type to use. The values used in
the main experiments from the paper are:
- `whenGoal-noBallLaunch`: When the ball enters the goal, no ball is created.
- `whenMiss-noOpponentScore`: When the ball hits the ground, the opponent's
  score does not increment.
- `whenGoal-illegal-incrementOpponentScore`: When the ball enters the goal, the
  opponent's score increments.
- `whenMove-error`: Either the left or right action do not appropriately move
  the paddle.
- `whenPaddle-illegal-incrementPlayerScore`: When the paddle moves, the
  player's score increments.
- `whenWall-illegal-incrementOpponentScore`: When the ball hits the wall, the
  opponent's score increments.
- `whenGoal-illegal-bounceBall`: The ball bounces off the goal.
- `whenRun-noBallLaunch`: No ball is launced at the beginning of the episode.

The `seed` argument takes an integer value. The results in the paper used seeds
0, 1, and 2.

### Example Command

This example command will train a DreamGrader agent on the bug type where no
ball is created after the ball enters the goal.

```
python3 main.py dream_grader_exp -b instruction_agent.policy.type=\"classifier\" -c configs/default.json -c configs/bounce.json -c configs/bounce_bigger_train.json -b environment.error_type=\"whenGoal-noBallLaunch\" -s 0
```

## Bounce Meta-RL Wrapper

This code also releases Bounce as a new meta-RL task.
We recommend using the `BounceMetaEnv` class from `envs/bounce.py`.
This class creates Gym environments, which correspond to different (student)
programs. The class can either be configured to directly read from a csv file
of real student programs, or can create its own programs programmatically.
This is achieved by passing the appropriate config file to the `load_config`
method.
By default, the example command provided above configures reading and using
real student programs with the train / test split used in the paper.
After configuring the `BounceMetaEnv` class, Gym environments corresponding to
different programs can be created with the `create_env` method.
Here is an example usage:

```
import config as cfg
from envs import bounce

# Uses the default config of real student programs
config_filenames = ["configs/default.json", "configs/bounce.json",
                    "configs/bounce_bigger_train.json"]
config = cfg.Config.from_files_and_bindings(config_filenames, [])

# Configure the class
bounce.BounceMetaEnv.load_config(config.get("environment"))

# Creates Gym environments:
# - The provided random seed and test flag randomly select a student program
#   from either the train or test split
seed = 0
test = False
env = bounce.BounceMetaEnv.create_env(seed, test=test)
```

## Citation

If you use this code, please cite our paper.

```
@article{liu2022giving,
  title={Giving Feedback on Interactive Student Programs with Meta-Exploration},
  author={Liu, Evan Zheran and Stephan, Moritz and Nie, Allen and Piech, Chris and Brunskill, Emma and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
