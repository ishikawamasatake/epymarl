import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium import register

# 行動: 0:上(-1, 0), 1:右(0, 1), 2:下(1, 0), 3:左(0, -1), 4:待機(0, 0)
MOVES = np.array([(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)])

class PredatorPreyEnv(gym.Env):
    """Predator-Prey
    引数：
        dim      : グリッドのサイズ（dim × dim）
        vision   : 視野範囲．観測は（2*vision+1) ×（2*vision+1）
        n_agents : Predator(agent)の数
    
    観測（エージェントごと）：
        視野範囲内の各グリッドについて，長さ dim*dim+4のベクトルを並べて1次元に平坦化したもの
        0~dim*dim-1 : マスID(ont-hot)
        dim*dim+1   : 盤面外
        dim*dim+2   : Preyの数
        dim*dim+3   : predatorの数（自身含む）
    """

    def __init__(self, dim=5, vision=0, n_agents=3):
        super().__init__()
        self.dim = dim
        self.vision = vision
        self.n_agents = n_agents
        self.naction = 5

        self.TIMESTEP_PRNALTY = -0.01
        self.POS_PREY_REWARD = 0.1

        self.BASE = dim * dim
        self.OUTSIDE_CLASS = self.BASE + 1
        self.PREY_CLASS = self.BASE + 2
        self.PREDATOR_CLASS = self.BASE + 3
        self.vocab_size = self.BASE + 4

        self.window = 2 * vision + 1
        self.obs_dim = self.window * self.window * self.vocab_size
        # 環境の真の状態（MAPPOのCritic用）
        self.state_size = (n_agents + 1) * self.BASE
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(0.0, float(n_agents), shape=(self.obs_dim,), dtype=np.float32)
                for _ in range(n_agents)
            )
        )
        self.action_space = spaces.Tuple(
            tuple(spaces.Discrete(self.naction) for _ in range(n_agents))
        )

        # visionに合わせて盤面外をOUTSIDE_CLASSでパディング
        self.grid = np.pad(
            np.arange(self.BASE).reshape(dim, dim),
            vision,
            "constant",
            constant_values=self.OUTSIDE_CLASS,
        )

        self.predator_loc = None #(n_agents, 2)
        self.prey_loc = None #(1, 2)
        self.episode_over = False
        self.success = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_over = False
        self.success = 0

        # predatorとpreyは互いに異なるマスから開始
        idx = self.np_random.choice(self.BASE, self.n_agents + 1, replace=False)
        locs = np.vstack(np.unravel_index(idx, (self.dim, self.dim))).T
        self.predator_loc, self.prey_loc = locs[:self.n_agents], locs[self.n_agents:]

        return self._get_obs(), {}

    def step(self, actions):
        if self.episode_over:
            raise RuntimeError("Episode is done. Call reset().")

        actions = np.asarray(actions).reshape(-1)
        assert actions.shape[0] == self.n_agents, "one action per agent is required"
        assert np.all((actions >= 0) & (actions < self.naction)), "action out of range"

        for i, a in enumerate(actions):
            self.predator_loc[i] = np.clip(self.predator_loc[i] + MOVES[a], 0, self.dim - 1)

        reward = self._get_reward()
        obs = self._get_obs()
        info = {"success": self.success}
        return obs, reward.tolist(), self.episode_over, False, info

    def get_state(self):
        """中央集権criticに渡す環境の真の状態
        各エージェントのone-hot + prey位置のone-hotベクトル
        """
        B = self.BASE
        s = np.zeros(self.state_size, dtype=np.float32)
        for i, (y, x) in enumerate(self.predator_loc):
            s[i * B + y * self.dim + x] = 1.0
        y, x = self.prey_loc[0]
        s[self.n_agents * B + y * self.dim + x] = 1.0
        return s
        
    def _get_obs(self):
        v, k = self.vision, self.window
        pred_count = np.zeros(self.grid.shape, dtype=np.float32)
        prey_count = np.zeros(self.grid.shape, dtype=np.float32)
        for y, x in self.predator_loc:
            pred_count[y + v, x + v] += 1
        for y, x in self.prey_loc:
            prey_count[y + v, x + v] += 1

        rows = np.arange(k)[:, None]
        cols = np.arange(k)[None, :]
        obs = []
        for y, x in self.predator_loc:
            ys, xs = slice(y, y + k), slice(x, x + k)
            o = np.zeros((k, k, self.vocab_size), dtype=np.float32)
            o[rows, cols, self.grid[ys, xs]] = 1.0
            o[:, :, self.PREDATOR_CLASS] = pred_count[ys, xs]
            o[:, :, self.PREY_CLASS] = prey_count[ys, xs]
            obs.append(o.reshape(-1))
        return tuple(obs)

    def _get_reward(self):
        reward = np.full(self.n_agents, self.TIMESTEP_PRNALTY)
        on_prey = np.where(np.all(self.predator_loc == self.prey_loc, axis=1))[0]
        n_on = on_prey.size

        reward[on_prey] = 0

        if n_on == self.n_agents:
            reward[on_prey] = self.POS_PREY_REWARD * self.n_agents
            self.episode_over = True
            self.success = 1

        return reward

    def render(self):
        g = [["." for _ in range(self.dim)] for _ in range(self.dim)]
        for y, x in self.predator_loc:
            g[y][x] = "X"
        for y, x in self.prey_loc:
            g[y][x] = "*" if g[y][x] == "X" else "P"  # * は同じマスに重なっている状態
        print("\n".join(" ".join(row) for row in g))

register(
    id="PredatorPrey",
    entry_point="envs.predator_prey:PredatorPreyEnv",
    disable_env_checker=True,
)