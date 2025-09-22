import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

# 固定随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ===== 数据读取函数 =====
def read_data(dataset, idx, is_train=True):
    if is_train:
        path = os.path.join('./dataset', dataset, 'train', f"{idx}.npz")
    else:
        path = os.path.join('./dataset', dataset, 'test', f"{idx}.npz")
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data

def load_all_clients(dataset, num_clients):
    trainloaders, testloaders = [], []
    for cid in range(num_clients):
        # 读 train
        train_data = read_data(dataset, cid, is_train=True)
        X_train = torch.tensor(train_data['x'], dtype=torch.float32)
        y_train = torch.tensor(train_data['y'], dtype=torch.int64)
        train_dataset = list(zip(X_train, y_train))
        trainloaders.append(DataLoader(train_dataset, batch_size=64, shuffle=True))

        # 读 test
        test_data = read_data(dataset, cid, is_train=False)
        X_test = torch.tensor(test_data['x'], dtype=torch.float32)
        y_test = torch.tensor(test_data['y'], dtype=torch.int64)
        test_dataset = list(zip(X_test, y_test))
        testloaders.append(DataLoader(test_dataset, batch_size=128, shuffle=False))
    return trainloaders, testloaders

# ===== 模型定义 =====
def get_model():
    return models.resnet18(num_classes=10)

# ===== 客户端本地训练 =====
def local_train(model, trainloader, epochs=5, lr=0.01, device='cpu'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# ===== FedAVG 聚合（等权）=====
def fedavg(updates):
    avg_state = {}
    for k in updates[0].keys():
        avg_state[k] = sum([u[k] for u in updates]) / len(updates)
    return avg_state

# ===== 测试函数 =====
@torch.no_grad()
def test(model, testloader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0

# ===== 主程序 =====
def main():
    dataset = "Cifar10"
    num_clients = 20
    rounds = 30
    local_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 一次性加载所有 clients 的数据
    trainloaders, testloaders = load_all_clients(dataset, num_clients)

    # —— 构建“Server 全局测试集”：拼接所有 client 的测试集 —— #
    all_test_examples = []
    for tl in testloaders:
        # tl.dataset 是一个 list[(x,y), ...]
        all_test_examples += list(tl.dataset)
    server_testloader = DataLoader(all_test_examples, batch_size=256, shuffle=False)

    # 初始化全局模型
    global_model = get_model().to(device)

    round_post_avg_accs = []  # 每轮“聚合后”的平均 client acc
    server_global_accs   = []  # 每轮“Server（全测试集）”的全局 acc

    for r in tqdm(range(rounds), desc="Global Rounds"):
        updates = []
        pre_accs = []   # 本轮“聚合前”各 client 的本地模型 acc
        post_accs = []  # 本轮“聚合后”全局模型在各 client 上的 acc

        # 本地训练 + 聚合前评估
        for cid in tqdm(range(num_clients), desc=f"Round {r+1} Clients", leave=False):
            local_model = get_model().to(device)
            local_model.load_state_dict(global_model.state_dict())

            new_state = local_train(local_model, trainloaders[cid],
                                    epochs=local_epochs, lr=0.01, device=device)
            updates.append(new_state)

            # 聚合前：本地模型在本 client 测试集上的 acc
            pre_acc = test(local_model, testloaders[cid], device=device)
            pre_accs.append(pre_acc)

        # 打印聚合前各 client acc 与平均值
        pre_line = ", ".join([f"c{cid}={a:.4f}" for cid, a in enumerate(pre_accs)])
        tqdm.write(f"[Pre-Agg ] Round {r+1} client accs: {pre_line}")
        tqdm.write(f"[Pre-Agg ] Round {r+1} Avg = {np.mean(pre_accs):.4f}")

        # FedAVG 聚合
        new_global_state = fedavg(updates)
        global_model.load_state_dict(new_global_state)

        # 聚合后：全局模型在每个 client 的测试集上
        for cid in range(num_clients):
            acc = test(global_model, testloaders[cid], device=device)
            post_accs.append(acc)
        post_line = ", ".join([f"c{cid}={a:.4f}" for cid, a in enumerate(post_accs)])
        tqdm.write(f"[Post-Agg] Round {r+1} client accs: {post_line}")

        post_avg = float(np.mean(post_accs))
        round_post_avg_accs.append(post_avg)
        tqdm.write(f"[Post-Agg] Round {r+1} Avg = {post_avg:.4f}")

        # —— Server 全局测试：用所有 client 的测试集的全集 —— #
        server_acc = test(global_model, server_testloader, device=device)
        server_global_accs.append(server_acc)
        tqdm.write(f"[Server  ] Round {r+1} Global Test (all clients' test union) = {server_acc:.4f}")

    print("\n=== All Rounds Post-Agg Average Client Acc ===")
    for rr, acc in enumerate(round_post_avg_accs, 1):
        print(f"Round {rr}: {acc:.4f}")
    print("post_agg_avg_list =", round_post_avg_accs)

    print("\n=== All Rounds Server Global Acc (union test) ===")
    for rr, acc in enumerate(server_global_accs, 1):
        print(f"Round {rr}: {acc:.4f}")
    print("server_global_list =", server_global_accs)

if __name__ == "__main__":
    main()