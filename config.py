from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

@dataclass
class TrainConfig:
    # env params
    R_earth: float = 6378.137 # [km], radius of the Earth
    altitude: float = 550 # [km]
    umax: float = 5e2  # [km/s^2], umax
    alpha_deg: float = 20  # [deg], half-cone angle
    h1_margin_deg: float = 2  # [deg], margin for half-cone angle
    h3_margin: float = 1e-4
    h3_gamma2: float = 5  # soft-docking parameters
    activate_distance: float = 1  # [km], distance to activate soft-docking constraint
    pos_terminal_stability: float = 1e-2 # [km], (<10 m) terminal stability for position
    vel_terminal_stability: float = 1e-5 # [km/sec], (<1 cm/sec)terminal stability for velocity
    # wandb params
    project: str = "PDCA"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "PDCA"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = False
    enable_wandb: bool = True
    # ppo params
    update_timestep: int = 2000
    action_std_init: float = 2.0
    action_std_decay_freq: int = 20000
    action_std_decay_rate: float = 0.005
    min_action_std: float = 0.05
    gamma: float = 0.995
    learning_rate_actor: float = 0.0001
    learning_rate_critic: float = 0.0003
    ppo_epoch: int = 20
    # training params
    num_episodes: int = 20000
    num_time_steps: int = 10000
    simulation_time: float = 50
    # reward params
    reward_xv: float = 0.5
    reward_terminate: float = 200
    penalty_h1: float = -20
    penalty_h2: float = -100
    penalty_h3: float = -20
    penalty_time: float = -0.1
    # evaluation params
    # simulation params
    render: bool = False
