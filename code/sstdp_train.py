from torch.utils.tensorboard import SummaryWriter   

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np


import readline
import argparse
from tqdm import tqdm
import stdp_module
import os
import lr_scheduler as ls

def encoder(x, T_max):
    encoded = ((1-x) * (T_max-1)).round().int()
    return encoded


def thermal_encoder(x, T_max):
    x_max = torch.max(x)
    x_min = torch.min(x)
    x = ((x-x_min) / (x_max - x_min) * (T_max - 1)).round().int()       # [0, T_max-1]
    temp = torch.arange(start=T_max-1, end=-1, step=-1).to(x.device)
    ret = x.unsqueeze(-1).repeat(*([1] * len(x.shape) + [T_max])) >= temp
    return ret


def target_output(output: torch.Tensor, label: torch.Tensor, T_max: int, delta: int):
    first = torch.min(output, dim=1)[0]
    target = torch.maximum(output, (first + delta).view(-1,1).repeat((1, output.shape[1]))).clamp(max=T_max)
    target[torch.arange(output.shape[0]), label] = first

    return target


def main():
    parser = argparse.ArgumentParser(description='SNN_mnist_test')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64) # init 64
    parser.add_argument('--T', type=int, default=255)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=100.)
    parser.add_argument('--learning_rate', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--dataset_dir', type=str, default="../mnist")
    parser.add_argument('--log_dir', type=str, default="./logpath/")
    parser.add_argument('--result_dir', type=str, default="./linear_termal_test_3")
    parser.add_argument('--net_type', type=str, default='linear')
    parser.add_argument('--A', type=float, default=1.0, help='positive stdp coefficient')
    parser.add_argument('--B', type=float, default=0.0, help='negative stdp coefficient')
    parser.add_argument('--delta', type=int, default=5, help='target difference between correct label and others')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--temp_a', type=float, default=5)
    parser.add_argument('--lr_scheduler', type=str, default='fixed')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    writer = SummaryWriter(args.log_dir)

    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    with open(os.path.join(args.result_dir, "args.txt"), 'w') as f:
        print(args, file=f)

    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)
        test_dataset = torchvision.datasets.MNIST(
            root=args.dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True)

        if args.net_type == 'linear':
            net = stdp_module.StdpLinearNetMnist(t_max=args.T, tau=args.tau, threshold=args.threshold, a=args.A, b=args.B, first_initial=args.temp_a)
        elif args.net_type == 'conv':
            net = stdp_module.StdpConvNetMnist(t_max=args.T, tau=args.tau, threshold=args.threshold, a=args.A, b=args.B)
        else:
            raise ValueError(f"Undefined network type {args.net_type}")
    else:
        raise ValueError(f"Undefined dataset {args.dataset}")

    print(len(list(train_dataset)))

    net = net.to(device)
    print(net.state_dict().keys())

    print(list(train_data_loader)[0][0].shape)
    input_sample = list(train_data_loader)[0][0].to(device)
    net(input_sample)       # initialize threshold

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location=device)
        net.load_state_dict(state_dict)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = ls.lr_scheduler(optimizer,
                                     batch_size=train_data_loader.batch_size,
                                     num_samples=len(train_data_loader.sampler),
                                     mode = args.lr_scheduler)

    train_times = 0
    max_test_accuracy = 0

    test_accs = []
    train_accs = []
    deuce_count = []

    used_encoder = encoder

    for epoch in range(args.train_epoch):
        net.train()
        train_total = 0
        train_correct = 0
        train_deuce = 0
        for img, label in tqdm(train_data_loader, ncols=100):
            img = img.to(device)
            label = label.to(device).float()

            optimizer.zero_grad()
            output = net(used_encoder(img, args.T))

            minFiringTime = torch.min(output, dim=1)[0]
            minFiringTime[minFiringTime == args.T] = args.T - args.delta
            target = torch.maximum(output, (minFiringTime + args.delta).view(-1, 1)
                                   .repeat((1, output.shape[1]))).clamp(max=args.T)
            target[torch.arange(output.shape[0]), label.long()] = minFiringTime


            target = target.detach()
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            # todo: whether to add the learning rate scheduler?
            # lr_scheduler.step()

            correct = 0
            for idx in range(args.batch_size):
                correct += output[idx, label[idx].long().item()].item() == minFiringTime[idx].item()
                if torch.sum(output[idx] == minFiringTime[idx]) > 1:
                    train_deuce += 1

            train_correct += correct
            train_total += label.numel()

            for m in net.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            train_times += 1

        net.eval()
        with torch.no_grad():
            test_total = 0
            test_correct = 0
            test_deuce = 0
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                output = net(used_encoder(img, args.T))

                minFiringTime = torch.min(output, dim=1)[0]
                correct = 0

                for idx in range(args.batch_size):
                    if output[idx, label[idx].long().item()].item() == minFiringTime[idx].item():
                        correct += 1
                        if torch.sum(output[idx] == minFiringTime[idx]) > 1:
                            print(output[idx].detach().cpu().numpy(), label[idx].item())
                            test_deuce += 1

                test_correct += correct
                test_total += label.numel()
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
            test_accuracy = test_correct / test_total
            test_accs.append(test_accuracy)
            deuce_count.append(test_deuce)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)

        print(
            f'Epoch {epoch}: current_acc={test_accuracy}, max_acc={max_test_accuracy}, train_times={train_times}, '
            f'deuce_times={test_deuce}')
        
        # tensorboard
        writer.add_scalar('Accuracy (%)', test_accuracy, global_step=epoch, walltime=None)

        if max_test_accuracy == test_accuracy:
            torch.save(net.state_dict(),
                       os.path.join(args.result_dir, f'epoch_{epoch}_acc_{max_test_accuracy:.3f}_deuce_{test_deuce}_best.pt'))
        else:
            torch.save(net.state_dict(),
                       os.path.join(args.result_dir, f'epoch_{epoch}_acc_{test_accuracy:.3f}_deuce_{test_deuce}.pt'))

        # retrieve dead neurons
        length = len(list(train_dataset))
        for m in net.modules():
            if hasattr(m, 'retrieve_dead_neurons'):
                m.retrieve_dead_neurons(length)


    train_accs = np.array(train_accs)
    np.save(os.path.join(args.result_dir, 'train_accs.npy'), train_accs)
    test_accs = np.array(test_accs)
    np.save(os.path.join(args.result_dir, 'test_accs.npy'), test_accs)
    deuce_counts = np.array(deuce_count)
    np.save(os.path.join(args.result_dir, 'deuce_count.npy'), deuce_counts)


if __name__ == '__main__':
    main()

