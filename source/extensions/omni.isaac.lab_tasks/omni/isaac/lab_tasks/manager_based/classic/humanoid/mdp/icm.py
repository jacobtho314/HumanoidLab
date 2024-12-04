import torch
import torch.nn as nn
import torch.nn.functional as F
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv


class ICMFeatureEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ICMForwardModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
    
    def forward(self, state_features, actions):
        x = torch.cat([state_features, actions], dim=-1)
        return self.net(x)

class ICMInverseModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, current_features, next_features):
        x = torch.cat([current_features, next_features], dim=-1)
        return self.net(x)

class ICM(nn.Module):
    def __init__(self, input_dim, action_dim, feature_dim=256, beta=0.2, device="cuda:0"):
        super().__init__()
        self.feature_encoder = ICMFeatureEncoder(input_dim, feature_dim).to(device)
        self.forward_model = ICMForwardModel(feature_dim, action_dim).to(device)
        self.inverse_model = ICMInverseModel(feature_dim, action_dim).to(device)
        self.feature_encoder.train()
        self.forward_model.train()
        self.inverse_model.train()
        self.beta = beta
        self.device = device

        print(input())
        
    def reset(self):
        """Reset accumulated gradients"""
        pass
            
    def compute_intrinsic_reward(self, obs, next_obs, actions):
        # Encode states into features
        current_features = self.feature_encoder(obs)
        print("current_features.requires_grad:", current_features.requires_grad)

        next_features = self.feature_encoder(next_obs)
        print("next_features.requires_grad:", next_features.requires_grad)

        # Forward model prediction
        predicted_next_features = self.forward_model(current_features, actions)
        print("predicted_next_features.requires_grad:", predicted_next_features.requires_grad)

        # Compute forward model loss (intrinsic reward)
        forward_loss = F.mse_loss(predicted_next_features, next_features, reduction='none')
        print("forward_loss.requires_grad:", forward_loss.requires_grad)

        intrinsic_reward = torch.mean(forward_loss, dim=-1)
        print("intrinsic_reward.requires_grad:", intrinsic_reward.requires_grad)

        # Inverse model prediction
        predicted_actions = self.inverse_model(current_features, next_features)
        print("predicted_actions.requires_grad:", predicted_actions.requires_grad)

        inverse_loss = F.mse_loss(predicted_actions, actions, reduction='none')
        print("inverse_loss.requires_grad:", inverse_loss.requires_grad)

        inverse_loss = torch.mean(inverse_loss, dim=-1)
        print("inverse_loss_mean.requires_grad:", inverse_loss.requires_grad)

        # Total loss for training
        total_loss = (1 - self.beta) * inverse_loss + self.beta * torch.mean(forward_loss, dim=-1)
        print("total_loss.requires_grad:", total_loss.requires_grad)

        # Test gradient flow explicitly
        try:
            grads = torch.autograd.grad(outputs=total_loss.mean(), inputs=list(self.parameters()), retain_graph=True)
            print("Gradient computation successful for total_loss!")
        except RuntimeError as e:
            print("Gradient computation failed:", e)

        return intrinsic_reward, total_loss
