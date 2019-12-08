from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional, Dict


@dataclass
class CommunicationParams:
    type: str
    units: Optional[List[int]]
    signal_dropout_rate: Optional[float] = None
    use_gru: bool = False
    use_bidir: bool = False
    can_see_others: bool = False


@dataclass
class AgentModelHParams:
    rnn_units: List[int]
    max_agents: int
    comm: Optional[CommunicationParams]


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
    entropy_strength: Optional[float] = None
    sync_each_n_games: int = 1
    use_data_augmentation: bool = False
    regularization_strength: Optional[float] = None
    logit_regularization_strength: Optional[float] = None
    actor_label_smoothing: Optional[float] = None
    optimizer: str = "RMSProp"
    max_game_steps: Optional[int] = None
    optimizer_kwargs: Optional[Dict[str, object]] = None


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
    exploration_reward: float = 0.0
    discount_rate: float = 0.9
    average_over_team: bool = True
    one_agent_exit_reward: float = 0.0
    failure_reward: float = None
    overall_shift: float = None


@dataclass
class ModelConfig:
    model: AgentModelHParams
    training: TrainingParams
    maze_size: int
    reward: RewardParams
