import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LLMStrategicPrior:
    """A lightweight LLM-style planner: state -> textual intent -> action prior."""

    def __init__(self, action_dim: int, alpha_start: float = 0.35, alpha_end: float = 0.05, decay_steps: int = 200000):
        self.action_dim = action_dim
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.decay_steps = max(decay_steps, 1)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def build_prompt(self, state: np.ndarray) -> str:
        # obs: [yaw, vel, mean_other_goal_dist, rel_goal_x, rel_goal_y, rel_radar_x, rel_radar_y, rel_a1_x, rel_a1_y, rel_a2_x, rel_a2_y]
        yaw, vel = float(state[0]), float(state[1])
        rel_goal = state[3:5]
        rel_radar = state[5:7]
        return (
            f"UAV状态: yaw={yaw:.2f}, vel={vel:.2f}; "
            f"目标相对位置=({rel_goal[0]:.2f}, {rel_goal[1]:.2f}); "
            f"雷达中心相对位置=({rel_radar[0]:.2f}, {rel_radar[1]:.2f}). "
            "请输出策略: [转向强度, 加速度强度]。"
        )

    def suggest_action(self, state: np.ndarray) -> np.ndarray:
        rel_goal = state[3:5]
        rel_radar = state[5:7]
        yaw = float(state[0])
        vel = float(state[1])

        goal_dist = np.linalg.norm(rel_goal) + 1e-6
        radar_dist = np.linalg.norm(rel_radar) + 1e-6
        desired_yaw = np.arctan2(rel_goal[1], rel_goal[0])
        yaw_err = self._wrap_pi(desired_yaw - yaw)

        turn = np.clip(yaw_err / np.pi, -1.0, 1.0)
        accel = np.clip((goal_dist - 1.2) / 1.8, -1.0, 1.0)

        # risk-aware shaping: entering radar area should prioritize turning and slowing down
        if radar_dist < 0.9:
            turn = np.clip(turn * 1.3, -1.0, 1.0)
            accel = min(accel, -0.3)

        if vel > 2.5:
            accel = min(accel, -0.2)

        action = np.zeros(self.action_dim, dtype=np.float32)
        action[0] = turn
        action[1] = accel
        return action

    def alpha(self, total_steps: int) -> float:
        ratio = min(total_steps / self.decay_steps, 1.0)
        return self.alpha_start + ratio * (self.alpha_end - self.alpha_start)


class RNDModule(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False

    def novelty(self, s: torch.Tensor):
        with torch.no_grad():
            t = self.target(s)
        p = self.predictor(s)
        mse = (p - t).pow(2).mean(dim=1, keepdim=True)
        return mse, p, t


class IntrinsicRewarder:
    def __init__(self, state_dim: int, device: torch.device, lr: float = 1e-3, coef: float = 0.05):
        self.device = device
        self.coef = coef
        self.module = RNDModule(state_dim).to(device)
        self.opt = optim.Adam(self.module.predictor.parameters(), lr=lr)

    def compute_and_update(self, state_next: np.ndarray) -> float:
        s = torch.tensor(state_next, dtype=torch.float32, device=self.device).unsqueeze(0)
        nov, _, _ = self.module.novelty(s)
        loss = nov.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(self.coef * nov.detach().cpu().item())


class AdaptiveCurriculum:
    """Adjust radar radius by recent performance: easier->harder progression."""

    def __init__(self, initial_radius: float, min_radius: float = 0.35, max_radius: float = 0.8, step: float = 0.02):
        self.radius = initial_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.step = step

    def update(self, env, episode_reward_sum: float):
        # reward high => harder (larger radar region), reward low => easier
        if episode_reward_sum > 90:
            self.radius = min(self.max_radius, self.radius + self.step)
        elif episode_reward_sum < 20:
            self.radius = max(self.min_radius, self.radius - self.step)
        env.world.target_radius = self.radius
        return self.radius
