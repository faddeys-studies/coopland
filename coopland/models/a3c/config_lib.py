from dataclasses import dataclass
from typing import List, Callable, Tuple


@dataclass
class AgentModelHParams:
    rnn_units: List[int]


@dataclass
class TrainingContext:
    model: "AgentModelHParams"
    problem: "ProblemParams"
    training: "TrainingParams"
    infrastructure: "TrainingInfrastructure"
    performance: "PerformanceParams"


@dataclass
class ProblemParams:
    reward_function: Callable
    """(maze, replay, exit_pos) -> [N], where N - number of game steps"""
    maze_size: Tuple[int, int]
    n_agents: int = 1


@dataclass
class TrainingParams:
    learning_rate: float
    actor_loss_weight: float
    critic_loss_weight: float
    discount_rate: float
    entropy_strength: float
    sync_each_n_games: int = 1
    use_data_augmentation: bool = False
    regularization_strength: float = -1


@dataclass
class TrainingInfrastructure:
    summaries_dir: str
    model_dir: str
    do_visualize: bool
    per_game_callback: Callable = None
    """(replays, immediate_reward, reward, critic_values, advantage) -> None"""


@dataclass
class PerformanceParams:
    system_supports_omp: bool
    omp_thread_limit: int
    multithreaded_training: bool
    session_config: object  # tf.ConfigProto (we just don't import tf here)
    n_workers: int = None  # if None, then set automatically

