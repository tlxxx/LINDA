# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.num_calculation import cal_gauss_kl
#
#
# class LINDAAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(LINDAAgent, self).__init__()
#         self.args = args
#         self.rnn_hidden_dim = args.rnn_hidden_dim
#         self.hidden_dim = 64
#         self.encoder_dim = 3
#         self.n_agents = args.n_agents
#         self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
#         self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
#
#         self.awareness_encoder = nn.Sequential(
#             nn.Linear(self.args.rnn_hidden_dim, self.hidden_dim),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(self.hidden_dim, 2 * self.n_agents * self.encoder_dim),
#         )
#         self.poster = nn.Sequential(
#             nn.Linear(2 * self.args.rnn_hidden_dim, self.hidden_dim),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(self.hidden_dim, 2 * self.encoder_dim),
#         )
#
#         self.fc2 = nn.Linear(self.rnn_hidden_dim + self.encoder_dim * self.n_agents, args.n_actions)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
#
#     def forward(self, inputs, hidden_state, test_mode=False):
#         batch_size = inputs.size(0) // self.n_agents
#         x = F.relu(self.fc1(inputs))
#         h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
#         # print(x.shape, h_in.shape, batch_size, self.n_agents, self.rnn_hidden_dim)
#         h = self.rnn(x, h_in)
#
#         awareness = self.awareness_encoder(h)
#         mu = awareness[:, :self.encoder_dim * self.n_agents].reshape(batch_size, self.n_agents, self.n_agents * self.encoder_dim)
#         sigma = torch.exp(awareness[:, self.encoder_dim * self.n_agents:]).reshape(batch_size, self.n_agents, self.n_agents * self.encoder_dim)
#         awareness_dict = torch.distributions.Normal(mu, sigma ** 0.5)
#         c = awareness_dict.sample().view(-1, self.n_agents * self.encoder_dim)
#
#         KL = torch.zeros(batch_size, 1).to(self.args.device)
#
#         # if not test_mode:
#         #     h_detach = h.view(batch_size, self.n_agents, self.rnn_hidden_dim).detach()
#         #     H = torch.zeros(batch_size, self.n_agents, self.n_agents * self.rnn_hidden_dim * 2).to(self.args.device)
#         #     for id in range(self.n_agents):
#         #         h_detach_i = h_detach[:, id: id + 1, :].repeat(1, self.n_agents, 1)
#         #         H[:, id, :] = torch.cat([h_detach, h_detach_i], dim=-1).view(-1, 2 * self.rnn_hidden_dim)
#         #     H = H.reshape(-1, self.rnn_hidden_dim * 2)
#         #     # print(H.shape)
#         #     out = self.poster(H).reshape(batch_size, self.n_agents, 2 * self.n_agents * self.encoder_dim)
#         #     mu1 = out[:, :, :self.encoder_dim * self.n_agents]
#         #     sigma1 = torch.exp(out[:, :, self.encoder_dim * self.n_agents:])
#
#         if not test_mode:
#             self.awareness_dim = self.encoder_dim
#             latent_size = self.n_agents * self.awareness_dim
#             h_detach = h.view(batch_size, self.n_agents, self.rnn_hidden_dim).detach()
#             infer_input = torch.zeros(self.n_agents, batch_size * self.n_agents, 2 * self.rnn_hidden_dim).to(self.args.device)
#             for agent_i in range(self.n_agents):
#                 h_detach_i = h_detach[:, agent_i:agent_i + 1].repeat(1, self.n_agents, 1)
#                 infer_input[agent_i, :, :] = torch.cat([h_detach_i, h_detach], dim=-1).view(-1, 2 * self.rnn_hidden_dim)
#             infer_input = infer_input.view(self.n_agents * batch_size * self.n_agents,
#                                            2 * self.rnn_hidden_dim)  # (N * B * N, 2R)
#
#             infer_params = self.poster(infer_input)  # (N * B * N, 2A)
#             infer_means = infer_params[:, :self.awareness_dim].reshape(self.n_agents, batch_size, latent_size)  # (N * B, N * A)
#             infer_vars = torch.exp(infer_params[:, self.awareness_dim:]).reshape(self.n_agents, batch_size, latent_size)  # (N, B, N * A)
#             mu1 = torch.transpose(infer_means, 0, 1)  # (B, N, N * A)
#             sigma1 = torch.transpose(infer_vars, 0, 1)  # (B, N, N * A)
#
#             KL = cal_gauss_kl(mu, sigma, mu1, sigma1).mean(dim=-1).mean(dim=-1, keepdim=True).to(self.args.device)
#
#         h = h.reshape(batch_size * self.n_agents, -1)
#         x = self.fc2(torch.cat([h, c], dim=-1))
#         return x, h, KL


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.num_calculation import cal_gauss_kl


class LINDAAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LINDAAgent, self).__init__()
        self.args = args
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.hidden_dim = 64
        self.encoder_dim = 3
        self.n_agents = args.n_agents
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.awareness_encoder = nn.Sequential(
            nn.Linear(self.args.rnn_hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.n_agents * self.encoder_dim),
        )
        self.poster = nn.Sequential(
            nn.Linear(2 * self.args.rnn_hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2 * self.encoder_dim),
        )

        self.fc2 = nn.Linear(self.rnn_hidden_dim + self.encoder_dim * self.n_agents, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, test_mode=False):
        batch_size = inputs.size(0) // self.n_agents
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        # size: (batch_size * n_agents) * rnn_dim
        # print(x.shape, h_in.shape, batch_size, self.n_agents, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # size: (batch_size * n_agents) * rnn_dim

        awareness = self.awareness_encoder(h).view(batch_size, self.n_agents, self.encoder_dim * self.n_agents * 2)
        # size: batch_size * n_agents * (self.encoder_dim * self.n_agents * 2)
        mu = awareness[:, :, :self.encoder_dim * self.n_agents]
        sigma = torch.exp(awareness[:, :, self.encoder_dim * self.n_agents:])
        awareness_dict = torch.distributions.Normal(mu, sigma)
        c = awareness_dict.rsample().view(-1, self.n_agents * self.encoder_dim)
        x = self.fc2(torch.cat([h, c], dim=-1))

        KL = torch.zeros(batch_size, 1).to(self.args.device)

        if not test_mode:
            h_detach = h.view(batch_size, self.n_agents, self.rnn_hidden_dim).detach()
            H = torch.zeros(batch_size, self.n_agents, self.n_agents * self.rnn_hidden_dim * 2).to(self.args.device)
            for id in range(self.n_agents):
                h_detach_i = h_detach[:, id: id +1, :].repeat(1, self.n_agents, 1)
                t = torch.cat([h_detach, h_detach_i], dim=-1)
                # print(t.shape)
                H[:, id, :] = t.view(-1, 2 * self.rnn_hidden_dim * self.n_agents)
            H = H.reshape(-1, self.rnn_hidden_dim * 2)
            # print(H.shape)
            out = self.poster(H).reshape(batch_size, self.n_agents, 2 * self.n_agents * self.encoder_dim)
            mu1 = out[:, :, :self.encoder_dim * self.n_agents]
            sigma1 = torch.exp(out[:, :, self.encoder_dim * self.n_agents:])

            KL = cal_gauss_kl(mu, sigma, mu1, sigma1).mean(dim=-1).mean(dim=-1, keepdim=True).to(self.args.device)
        return x, h, KL
