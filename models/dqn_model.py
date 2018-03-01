import torch.nn as nn
from copy import deepcopy


class DQN(nn.Module):
    """
    Base model architecture for Nature DQN
    """
    def __init__(self, in_channels=4, n_actions=18):
        super(DQN, self).__init__()

        # define convolutional + fc layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, n_actions)

        # cutting down on size for downsampled Atari images
        # self.fc4 = nn.Linear(64, 32)
        # self.fc5 = nn.Linear(32, num_actions)

    def variational(self):
        # todo: come back and fix this
        return False

    def forward(self, x):
        """
        runs forward propagation of vanilla DQN
        :param x:
        :return:
        """
        # (32, 84, 84, 4) --> (32, 4, 84, 84)
        x = x.permute(0, 3, 1, 2)
        out = self.conv1(x)  # (32, 32, 9, 9)
        out = self.relu(out)

        out = self.conv2(out)  # (32, 64, 3, 3)
        out = self.relu(out)

        out = self.conv3(out)  # (32, 64, 1, 1)
        out = self.relu(out)

        out = self.fc4(out.view(out.size(0), -1))
        out = self.relu(out)
        out = self.fc5(out)

        return out

    def save_target(self):
        # self.target = deepcopy(self.head)
        pass

    def target_value(self, rewards, gamma, states, reset_volatile=True):
        # assert self.target is not None, "Must call save_target at least once before calculating target_value"
        # q_s = self.target(states.view(states.size(0), -1))
        # q_sa = q_s.max(1)[0]
        # return rewards + gamma * q_sa
        pass