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
        trainloaders.append(DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2))

        # 读 test
        test_data = read_data(dataset, cid, is_train=False)
        X_test = torch.tensor(test_data['x'], dtype=torch.float32)
        y_test = torch.tensor(test_data['y'], dtype=torch.int64)
        test_dataset = list(zip(X_test, y_test))
        testloaders.append(DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2))
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

# ===== FedAVG 聚合 =====
def fedavg(updates):
    avg_state = {}
    for k in updates[0].keys():
        avg_state[k] = sum([u[k] for u in updates]) / len(updates)
    return avg_state

# ===== 测试函数 =====
def test(model, testloader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# ===== 主程序 =====
def main():
    dataset = "Cifar10"
    num_clients = 20
    rounds = 30
    local_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # group 分配
    groups = [[i, i+5, i+10, i+15] for i in range(5)]

    # 一次性加载所有 clients 的数据
    trainloaders, testloaders = load_all_clients(dataset, num_clients)

    # 每个 group 有自己的模型
    group_models = [get_model().to(device) for _ in range(5)]

    round_avg_accs = []

    for r in tqdm(range(rounds), desc="Global Rounds"):
        # 存放每个 group 的更新
        new_group_states = []

        for g, clients in enumerate(groups):
            updates = []
            # 每个 client 从当前 group model 出发
            for cid in tqdm(clients, desc=f"Round {r+1} Group {g}", leave=False):
                local_model = get_model().to(device)
                local_model.load_state_dict(group_models[g].state_dict())
                new_state = local_train(local_model, trainloaders[cid],
                                        epochs=local_epochs, lr=0.01, device=device)
                updates.append(new_state)

            # group 内 FedAVG
            new_group_state = fedavg(updates)
            new_group_states.append(new_group_state)

        # 更新 group 模型
        for g in range(5):
            group_models[g].load_state_dict(new_group_states[g])

        # 所有 client 测试
        client_accs = []
        for g, clients in enumerate(groups):
            for cid in clients:
                acc = test(group_models[g], testloaders[cid], device=device)
                client_accs.append(acc)

        avg_acc = sum(client_accs) / len(client_accs)
        round_avg_accs.append(avg_acc)
        tqdm.write(f"Round {r+1}: Avg Client Test Acc = {avg_acc:.4f}")

    print("\n=== All Rounds Average Acc ===")
    for r, acc in enumerate(round_avg_accs, 1):
        print(f"Round {r}: {acc:.4f}")

if __name__ == "__main__":
    main()