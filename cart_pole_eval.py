import argparse
import os
import pickle
from importlib import metadata
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import time

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl-rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from cart_pole_env import CartPoleEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="cartpole-training")
    parser.add_argument("--ckpt", type=int, default=499)
    parser.add_argument("--num_episodes", type=int, default=1)
    args = parser.parse_args()
    
    gs.init()

    eval_seed = 10
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    random.seed(eval_seed)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))

    env = CartPoleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
        eval_mode=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    eval_log_dir = os.path.join(log_dir, "eval", f"ckpt_{args.ckpt}")
    os.makedirs(eval_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=eval_log_dir, flush_secs=10)
    print(f"Evaluation logs → {eval_log_dir}")
    print(f"TensorBoard: tensorboard --logdir={eval_log_dir}")

    obs, _ = env.reset()
    
    episode_rewards = []
    current_reward = 0.0
    global_step = 0

    obs, _ = env.reset()

    with torch.no_grad():
        while len(episode_rewards) < args.num_episodes:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            global_step += 1

            current_reward += rews.item()

            writer.add_scalar("Eval/Raw_Reward", rews.item() * 1000, global_step)

            if dones.item():
                episode_rewards.append(current_reward)
                obs, _ = env.reset()

    writer.close()


if __name__ == "__main__":
    main()