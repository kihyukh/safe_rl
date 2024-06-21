from ppo import PPO
from cwh_env_ki import CWHEnv
from dataclasses import asdict, dataclass
import numpy as np
import pyrallis
import time
import matplotlib.pyplot as plt
from collections import defaultdict


from logger import WandbLogger, DummyLogger
from config import TrainConfig


@pyrallis.wrap()
def train(args: TrainConfig):
    ppo_agent = PPO(
        state_dim=6,
        action_dim=3,
        lr_actor=args.learning_rate_actor,
        lr_critic=args.learning_rate_critic,
        gamma=args.gamma,
        K_epochs=args.ppo_epoch,
        eps_clip=0.2,
        has_continuous_action_space=1,
        action_std_init=args.action_std_init,
    )

    update_timestep = args.update_timestep
    action_std_decay_freq = args.action_std_decay_freq
    action_std_decay_rate = args.action_std_decay_rate
    min_action_std = args.min_action_std

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    print_freq = 100

    time_step = 0

    R_earth  = args.R_earth
    altitude = args.altitude
    R = altitude + R_earth
    w_pos, w_vel, w_control = 1e3, 1, 1
    s0 = np.array([10, 20, 20, 0.5, 0.3, 0.1]); # initial state
    tf = args.simulation_time # [sec], final simulation time

    env = CWHEnv(R,w_pos,w_vel,w_control, tf, args.num_time_steps + 1, asdict(args))

    if args.enable_wandb:
        logger = WandbLogger(asdict(args), 'rl', 'group', 'experiment', None)
    else:
        logger = DummyLogger()
    if args.render:
        ax = plt.figure(figsize=(6, 6)).add_subplot(projection='3d')
        ax.set_xlabel('$X$', fontsize=18)
        ax.set_ylabel('$Y$', fontsize=18)
        ax.set_zlabel('$Z$', fontsize=18)
        plt.ion()
        plt.show()

    buffer = []
    plot_every = 1

    # training loop
    for episode_id in range(args.num_episodes):
        state, info = env.reset(s0)
        cumulative_reward = 0
        episode_length = 0
        num_violations = 0

        if args.render:
            ax.cla()
            ax.scatter(0, 0, 0, c="tab:red")
            ax.scatter(s0[0], s0[1], s0[2], c="tab:blue")
        reward_dict = defaultdict(float)
        violation_dict = defaultdict(float)
        for t in range(1, args.num_time_steps - 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            action = np.clip(action / 10 + 0.5, 0, 1)
            state, reward, done, truncated, info = env.step(action)
            for key, value in info.get('reward', {}).items():
                reward_dict[key] += value
            for key, value in info.get('violation', {}).items():
                violation_dict[key] += value

            if t % plot_every == 0:
                buffer.append(state)
            if args.render and len(buffer) == 10:
                x = [data[0] for data in buffer]
                y = [data[1] for data in buffer]
                z = [data[2] for data in buffer]
                ax.scatter(x, y, z, c="tab:blue")
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.0000001)
                buffer = []

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            episode_length += 1
            cumulative_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)


            # break; if the episode is over
            if done:
                break
            if args.verbose:
                logger.store(tab="detail", Reward=reward)
                logger.write(t, display=False)

        num_violations = violation_dict.get('violation_cone', 0)
        print("Episode : {}, Timestep : {}, Reward : {:.2f}, Violation : {}, Success : {}".format(
            episode_id + 1, episode_length, cumulative_reward, num_violations, 'True' if done else 'False'))
        logger.store(tab="eval", Timestep=episode_length, ConeViolation=num_violations, Reward=cumulative_reward)
        logger.store(tab="reward", **reward_dict)
        logger.store(tab="reward", **violation_dict)
        logger.store(tab="ppo", ActionSTD=ppo_agent.action_std)
        logger.write(episode_id, display=False)


    env.close()

if __name__ == "__main__":
    train()
