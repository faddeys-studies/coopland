from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional, Union


@dataclass
class ModelHParamsNoCoop:
    rnn_units: List[int]


@dataclass
class ModelHParamsVisOnly:
    rnn_units: List[int]
    max_agents: int


@dataclass
class ModelHParamsNestRNN:
    rnn_units: List[int]
    comm_units: Optional[List[int]]
    max_agents: int


@dataclass
class ModelHParamsSignalRL:
    rnn_units: List[int]
    max_agents: int


ModelHParams = Union[
    ModelHParamsNoCoop, ModelHParamsVisOnly, ModelHParamsNestRNN, ModelHParamsSignalRL
]


@dataclass
class TrainingContext:
    model_type: str
    model_hparams: ModelHParams
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
    entropy_strength: Optional[float] = None
    sync_each_n_games: int = 1
    use_data_augmentation: bool = False
    regularization_strength: Optional[float] = None
    logit_regularization_strength: Optional[float] = None
    actor_label_smoothing: Optional[float] = None


@dataclass
class TrainingInfrastructure:
    summaries_dir: str
    model_dir: str
    do_visualize: bool
    train_until_n_games: int = None
    per_game_callback: Callable = None
    """
    (
        worker_id, game_index, replays, 
        immediate_reward, reward, critic_values, advantage
    ) -> None
    """


@dataclass
class PerformanceParams:
    system_supports_omp: bool
    omp_thread_limit: int
    multithreaded_training: bool
    session_config: object  # tf.ConfigProto (we just don't import tf here)
    n_workers: int = None  # if None, then set automatically


@dataclass
class RewardParams:
    step_reward: float
    exit_reward: float
    exit_reward_diff_decay: float
    discount_rate: float = 0.9


@dataclass
class ModelConfig:
    model_type: str
    model_hparams: ModelHParams
    training: TrainingParams
    maze_size: int
    reward: RewardParams
