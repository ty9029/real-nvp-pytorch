import argparse
import os
import glob
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from distributions import Normal, DoubleMoons
from models import RealNVP
from loss import Loss


def train(model, epoch, opt):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = Loss(2, opt)

    train_dataset = DoubleMoons(10000)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(opt.device)

        optimizer.zero_grad()
        sample_z, log_det_jacobian = model(data)
        loss = criterion(sample_z, log_det_jacobian)
        loss.backward()
        optimizer.step()

    x, z = eval_inference(model, opt)
    np.savez_compressed("outputs/inference/{:04}.npz".format(epoch), x=x, z=z)

    x, z = eval_generate(model, opt)
    np.savez_compressed("outputs/generate/{:04}.npz".format(epoch), x=x, z=z)

    print(loss)


def eval_inference(model, opt):
    eval_dataset = DoubleMoons(1000)
    eval_loader = DataLoader(eval_dataset, batch_size=opt.batch_size)

    xs, zs = [], []
    model.eval()
    for data in eval_loader:
        data = data.to(opt.device)
        z, _ = model(data)

        x = data.cpu().tolist()
        xs.extend(x)

        z = z.cpu().tolist()
        zs.extend(z)

    return np.array(xs), np.array(zs)


def eval_generate(model, opt):
    eval_dataset = Normal(1000)
    eval_loader = DataLoader(eval_dataset, batch_size=opt.batch_size)

    xs, zs = [], []
    model.eval()
    for data in eval_loader:
        data = data.to(opt.device)
        x = model.inverse(data)

        x = x.cpu().tolist()
        xs.extend(x)

        z = data.cpu().tolist()
        zs.extend(z)

    return np.array(xs), np.array(zs)


def main():
    parser = argparse.ArgumentParser(description="RealNVP")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    opt = parser.parse_args()

    os.makedirs("outputs/generate", exist_ok=True)
    os.makedirs("outputs/inference", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    model = RealNVP(
        n_layers=8,
        in_dim=2,
        hidden_dim=256
    ).to(opt.device)

    for i in range(opt.num_epoch):
        train(model, i, opt)

    torch.save(model.state_dict(), "weights/model.pth")


if __name__ == "__main__":
    main()
