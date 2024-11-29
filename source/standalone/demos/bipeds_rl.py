#!/usr/bin/env python3

"""Train the Isaac Humanoid environment using Stable-Baselines3 PPO."""

import argparse
from omni.isaac.lab.app import AppLauncher
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Isaac Humanoid SB3 PPO example script")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Launch the simulator
    launcher = AppLauncher(args)

    # Import environment-related modules after simulator is launched
    from omni.isaac.lab_tasks.direct.humanoid.humanoid_env import HumanoidEnv, HumanoidEnvCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg

    # Configure environment
    cfg = HumanoidEnvCfg(
        # Environment settings
        episode_length_s=30.0,
        decimation=2,
        action_scale=1.0,
        action_space=21,
        observation_space=75,
        
        # Scene settings
        scene=InteractiveSceneCfg(
            num_envs=1,
            env_spacing=8.0,
            replicate_physics=True
        ),
    )   
    
    # Create environment
    env = HumanoidEnv(cfg)
    print("Created Humanoid environment with:")
    print(f"\tObservation space: {env.observation_space}")
    print(f"\tAction space: {env.action_space}")

    # Verify environment compatibility with SB3
    check_env(env)

    # Wrap environment with DummyVecEnv for SB3 compatibility
    vec_env = DummyVecEnv([lambda: env])

    # Define PPO model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=30,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_humanoid_tensorboard/"
    )

    # Set up evaluation callback
    eval_callback = EvalCallback(
        vec_env, best_model_save_path="./logs/best_model/",
        log_path="./logs/results/", eval_freq=5000,
        deterministic=True, render=False
    )

    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=1000000, callback=eval_callback)

    # Save the trained model
    model.save("ppo_humanoid")
    print("Model saved as 'ppo_humanoid'.")

    # Test the trained model
    print("Testing the trained model...")
    obs = vec_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()

    # Cleanup
    env.close()
    launcher.app.close()

if __name__ == "__main__":
    main()
