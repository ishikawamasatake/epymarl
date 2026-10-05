import torch.nn as nn
import torch.nn.functional as F

class CommNetAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CommNetAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim
        if self.n_agents < 2:
            raise ValueError("CommNetAgent requires n_agents >= 2")

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        self.comm = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs: (batch * n_agents, input_dim)
        batch = inputs.shape[0] // self.n_agents
        e = F.relu(self.fc1(inputs))

        h_prev = hidden_state.reshape(batch, self.n_agents, self.hidden_dim)
        # 受信者自身を除いた平均：（全員の和　―　自分）/ (n - 1)
        c = (h_prev.sum(dim=1, keepdim=True) - h_prev) / (self.n_agents - 1)
        m = self.comm(c)

        x = e + m.reshape(batch * self.n_agents, self.hidden_dim)
        h = self.rnn(x, h_prev.reshape(-1, self.hidden_dim))
        out = self.fc2(h)
        return out, h