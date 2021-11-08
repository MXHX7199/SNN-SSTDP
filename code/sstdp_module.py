import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import TypeVar, Union, Tuple

torch.backends.cudnn.benchmark = True
_seed_ = 2021
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SpikeOnceNeuron(nn.Module):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
        r"""
            :param tau: time constant for decay
            :param v_threshold: firing threshold membrane voltage
            :param v_reset: reset membrane voltage after firing
        """
        super(SpikeOnceNeuron, self).__init__()
        self.initial_threshold = v_threshold
        self.v_threshold = None
        self.v_reset = v_reset
        self.tau = tau
        self.spiked = None
        self.monitor = {'v': [], 's': [], 'spike_v': 0}

        self.v = self.v_reset
        self.firing_frequency = 0

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}'

    def spiking(self):
        spike = (1 - self.spiked) * (self.v > self.v_threshold)
        self.spiked += spike

        self.monitor['spike_v'] += self.v * spike
        self.monitor['v'] = self.v

        return spike

    def forward(self, dv: torch.Tensor):
        if self.spiked is None:
            self.spiked = torch.zeros_like(dv).to(dv.device)
        if self.v_threshold is None:
            self.v_threshold = nn.Parameter(torch.full_like(dv[0], fill_value=self.initial_threshold)).to(dv.device)

        if self.tau is None:
            self.v += dv
        else:
            self.v += (dv - (self.v - self.v_reset)) / self.tau
        return self.spiking()

    def reset(self):
        self.v = self.v_reset
        self.monitor = {'v': [], 's': [], 'spike_v': 0}
        self.firing_frequency += self.spiked.sum(dim=0)
        self.spiked = None


def stdp_update(t_i=None, t_j=None, a=0.8, b=0.2, tau=None, t_max=256) -> (torch.Tensor, torch.Tensor):
    r"""
        :param t_max: total simulation time
        :param t_i: pre-synapse firing time
        :param t_j: post-synapse firing time
        :param a: positive update constant
        :param b: negative update constant
        :param tau: time constant for update decay
        :return: weight update, pre-synaptic update
    """
    t_i = t_i.unsqueeze(1)
    t_j = t_j.unsqueeze(2)
    delta_t = t_i - t_j
    if tau is None:
        update = ((delta_t <= 0) * (t_i < t_max)).float() * (a + b) - b  # [-b, a]
        time_update = ((delta_t <= 0) * (t_i < t_max)).float()
    else:
        update = torch.zeros_like(delta_t)
        update = torch.where(delta_t > 0, - b * torch.exp(-delta_t / tau),
                             torch.where(delta_t < 0, a * torch.exp(delta_t / tau), update))
        pre = 255
        post = 0
        time_update = (delta_t < 0).float()

    return update, time_update


def stdp_linear_container(neuron_layer, t_max, a=0.8, b=0.2, tau=None, thermal=False, lr_ratio=1.0):
    class StdpLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_time, weight) -> torch.Tensor:
            r"""
            Shape:
                - input_time: (batch, feature_in)
                - weight: (feature_out, feature_in)
                - output: (batch, feature_out)
            """
            input_time_original = input_time.detach().clone()
            device = input_time.device
            placeholder = torch.arange(start=0, end=t_max, step=1).to(device)
            if thermal is False:
                input: torch.Tensor = input_time.unsqueeze(-1).repeat(*([1] * len(input_time.shape) + [t_max]))\
                                      == placeholder
            else:
                input: torch.Tensor = input_time.unsqueeze(-1).repeat(*([1] * len(input_time.shape) + [t_max]))\
                                      >= placeholder

            input = input.float().permute(2, 0, 1)

            valid = t_max - 3
            output = neuron_layer(F.linear(input[0], weight)).repeat(valid+1, 1, 1)
            for t in range(1, valid+1):
                output[t] = neuron_layer(F.linear(input[t], weight))

            output_time = torch.where(torch.sum(output, dim=0) == 0, t_max, (torch.argmax(output, dim=0))).float()
            ctx.save_for_backward(input_time, output_time, weight)
            return output_time

        @staticmethod
        def backward(ctx, output_grad):  # batch, out_feature
            input_time, output_time, weight = ctx.saved_tensors

            normalization = torch.sum(output_grad ** 2, dim=1) ** 0.5 + 1e-10
            output_grad /= normalization.unsqueeze(1)

            stdp_result, fire_ahead = stdp_update(t_i=input_time, t_j=output_time, a=a, b=b, tau=tau, t_max=t_max)

            weight_masked = (weight.repeat(output_time.shape[0], 1, 1) * fire_ahead)
            weight_grad = (- stdp_result * output_grad.unsqueeze(2)).mean(dim=0) * lr_ratio
            input_grad = output_grad.unsqueeze(1).matmul(weight_masked).squeeze(1)

            return input_grad, weight_grad

    return StdpLinearFunction.apply


class StdpLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, tau: float = None, t_max: int = 10,
                 threshold: float = 1.0, initial_range: float = 1.0, a=0.8, b=0.2, retrieve=False, thermal=False, lr_ratio=1.0):
        super().__init__(in_features, out_features, bias)
        self.neuron_layer = SpikeOnceNeuron(tau=tau, v_threshold=threshold)
        self.container = stdp_linear_container(self.neuron_layer, t_max=t_max, a=a, b=b, tau=tau, thermal=thermal, lr_ratio=lr_ratio)
        self.initial_range = initial_range
        self.retrieve = retrieve
        np.random.seed(0)
        self.weight = nn.Parameter(
            torch.tensor(np.random.random_sample(self.weight.shape)).float() * initial_range)

    def forward(self, input: torch.Tensor):
        return self.container(input, self.weight)

    def retrieve_dead_neurons(self, total_length):
        if self.retrieve:
            dead = self.neuron_layer.firing_frequency < 0.001 * total_length
            for idx, is_dead in enumerate(dead):
                if is_dead:
                    self.weight[idx].data = torch.tensor(
                        np.random.random_sample(self.weight[idx].shape)).float() * self.initial_range
            self.neuron_layer.firing_frequency = torch.zeros_like(self.neuron_layer.firing_frequency)


def stdp_conv2d_container(neuron_layer, t_max, a=0.8, b=0.2, tau=None):
    class StdpConv2dFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_time, weight, bias, stride) -> torch.Tensor:
            r"""
            Shape:
                - input_time: (batch, feature_in, h, w)
                - weight: (feature_out, feature_in, k, k)
                - output_time: (batch, feature_out, h_out, w_out)
            """

            device = input_time.device
            placeholder = torch.arange(start=0, end=t_max, step=1).to(device)
            input: torch.Tensor = input_time.unsqueeze(-1).repeat(*([1] * len(input_time.shape) + [t_max])) == placeholder
            input = input.float().permute(4, 0, 1, 2, 3)

            valid = t_max - 3
            input_v = F.conv2d(input[0], weight, bias, stride, (0, 0))
            output = neuron_layer(input_v).repeat(valid+1, 1, 1, 1, 1)  # t_max * batch * feature_out * w_out * h_out
            for t in range(1, valid+1):
                output[t] = neuron_layer(F.conv2d(input[t], weight, bias, stride, (0, 0)))

            output_time = torch.where(torch.sum(output, dim=0) == 0, t_max, torch.argmax(output, dim=0)).float()
            output_time = output_time.clamp(max=t_max)
            ctx.save_for_backward(input_time, output_time, weight, bias, torch.tensor(stride))
            return output_time

        @staticmethod
        def backward(ctx, output_grad):  # batch, out_feature, h_out, w_out
            # todo: `bias == True` not implemented
            input_time, output_time, weight, bias, stride = ctx.saved_tensors

            normalization = torch.sum(output_grad ** 2, dim=(1, 2, 3)) ** 0.5 + 1e-10
            output_grad /= normalization.reshape(-1, 1, 1, 1)  # batch-wise grad normalizaion

            batch, feature_in, h, w = input_time.shape
            h_k, w_k = weight.shape[2:4]
            feature_out = output_time.shape[1]

            weight_grad = torch.zeros_like(weight)
            input_grad = torch.zeros_like(input_time).float()
            for i in range(weight.shape[2]):
                for j in range(weight.shape[3]):
                    input_time_part = input_time[:, :, i: i + h - h_k + 1: stride[0], j: j + w - w_k + 1: stride[1]]
                    # input_time_part.shape = batch * feature_in * h_out * w_out
                    stdp_result, fire_ahead = stdp_update(input_time_part, output_time, a=a, b=b, tau=tau)
                    # both shape of batch * feature_out * feature_in * h_out * w_out
                    weight_grad[:, :, i, j] = torch.mean(torch.sum(-stdp_result * output_grad.unsqueeze(2),
                                                                   dim=(3, 4)), dim=0)
                    weight_part_masked = (weight[:, :, i, j].reshape(1, feature_out, feature_in, 1, 1)) * fire_ahead
                    input_grad[:, :, i: i + h - h_k + 1: stride[0], j: j + w - w_k + 1: stride[1]] += \
                        output_grad.unsqueeze(1).permute(0, 3, 4, 1, 2).matmul(
                            weight_part_masked.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2).squeeze(1)

            return input_grad, weight_grad, None, None
    return StdpConv2dFunction.apply


class StdpConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = False,
                 tau: Union[None, float] = None,
                 t_max: int = 10,
                 threshold: float = 1.0,
                 initial_range: float = 1.0,
                 a: float = 0.8,
                 b: float = 0.2):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.neuron_layer = SpikeOnceNeuron(tau=tau, v_threshold=threshold)
        self.container = stdp_conv2d_container(self.neuron_layer, t_max=t_max, tau=tau, a=a, b=b)
        self.initial_range = initial_range
        self.weight = nn.Parameter(torch.rand(self.weight.shape).float() * initial_range)
        self.t_max = t_max

    def forward(self, input: torch.Tensor):
        # input.shape = batch * input_feature * w * h
        if self.padding[0] > 0 or self.padding[1] > 0:
            padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            input_pad = F.pad(input, padding, mode='constant', value=self.t_max)
        else:
            input_pad = input
        return self.container(input_pad, self.weight, self.bias, self.stride)
