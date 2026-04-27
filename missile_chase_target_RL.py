import gymnasium as gym
import numpy as np


class MissileEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.tmax = 90
        self.dt = 0.1

        self.max_accel = 500.0
        self.miss_vel = 1200
        self.targ_vel = 600.0   # easier learning

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

    # ===============================
    # OBSERVATION
    # ===============================
    def _get_obs(self):
        los = self.aircraft_pos - self.missile_pos
        distance = np.linalg.norm(los) + 1e-6

        ulos = los / distance
        rel_vel = self.missile_vel - self.aircraft_vel

        Vc = np.dot(rel_vel, ulos)
        omega_los = np.cross(ulos, rel_vel) / distance

        obs = np.concatenate([
            ulos,
            self.missile_vel / self.miss_vel,
            [Vc / self.miss_vel],
            omega_los,
            [distance / 50000.0]
        ])

        return np.nan_to_num(obs).astype(np.float32)

    # ===============================
    # AIRCRAFT MOTION (FIXED)
    # ===============================
    def _update_aircraft(self):

        # simple forward motion
        new_pos = self.aircraft_pos + self.aircraft_vel * self.dt

        # small random maneuver
        if np.random.random() < 0.02:
            noise = np.random.uniform(-100, 100, 3)
            noise[2] *= 0.2

            self.aircraft_vel += noise

            norm = np.linalg.norm(self.aircraft_vel) + 1e-6
            self.aircraft_vel = self.aircraft_vel / norm * self.targ_vel

        return new_pos

    # ===============================
    # RESET
    # ===============================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = np.random.default_rng(seed)

        self.aircraft_pos = np.array([
            rng.uniform(-20000, 20000),
            rng.uniform(-20000, 20000),
            rng.uniform(8000, 12000)
        ])

        self.missile_pos = np.array([
            rng.uniform(-30000, 30000),
            rng.uniform(-30000, 30000),
            0.0
        ])

        # initial missile velocity toward target
        direction = self.aircraft_pos - self.missile_pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        self.missile_vel = direction * self.miss_vel

        # aircraft random velocity
        self.aircraft_vel = np.array([
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
            0.0
        ])
        self.aircraft_vel = self.aircraft_vel / (np.linalg.norm(self.aircraft_vel)+1e-6) * self.targ_vel

        self.prev_distance = np.linalg.norm(self.aircraft_pos - self.missile_pos)
        self.t = 0.0

        return self._get_obs(), {}

    # ===============================
    # STEP
    # ===============================
    def step(self, action):

        # missile dynamics
        accel = action * self.max_accel
        self.missile_vel += accel * self.dt

        speed = np.linalg.norm(self.missile_vel) + 1e-6
        self.missile_vel = self.missile_vel / speed * min(speed, self.miss_vel)

        self.missile_pos += self.missile_vel * self.dt

        # prevent underground
        if self.missile_pos[2] < 0:
            self.missile_pos[2] = 0
            self.missile_vel[2] = 0

        # update aircraft
        self.aircraft_pos = self._update_aircraft()

        self.t += self.dt

        # ===============================
        # REWARD (STRONG & STABLE)
        # ===============================
        distance = np.linalg.norm(self.aircraft_pos - self.missile_pos)

        # base penalty (stay close)
        reward = -0.01 * distance

        # reward for reducing distance
        reward += (self.prev_distance - distance) * 5

        # small control penalty
        reward -= 0.001 * np.linalg.norm(accel)

        done = False

        # interception reward
        if distance < 50:
            reward += 1000
            done = True

        # timeout penalty
        if self.t > self.tmax:
            reward -= 200
            done = True

        self.prev_distance = distance

        return self._get_obs(), reward, done, False, {}

    def render(self):
        pass