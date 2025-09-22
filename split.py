import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ======= 固定随机种子 =======
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ======= 数据读取 =======
def read_data(dataset, idx, is_train=True):
    path = os.path.join('./dataset', dataset, 'train' if is_train else 'test', f"{idx}.npz")
    with open(path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data

def load_all_clients(dataset, num_clients):
    trainloaders, testloaders, train_sizes = [], [], []
    for cid in range(num_clients):
        tr = read_data(dataset, cid, True)
        te = read_data(dataset, cid, False)
        Xtr = torch.tensor(tr['x'], dtype=torch.float32)
        ytr = torch.tensor(tr['y'], dtype=torch.int64)
        Xte = torch.tensor(te['x'], dtype=torch.float32)
        yte = torch.tensor(te['y'], dtype=torch.int64)
        train_dataset = list(zip(Xtr, ytr))
        test_dataset  = list(zip(Xte, yte))
        trainloaders.append(DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2))
        testloaders.append(DataLoader(test_dataset,  batch_size=128, shuffle=False, num_workers=2))
        train_sizes.append(len(train_dataset))
    return trainloaders, testloaders, train_sizes

# ======= ResNet18 拆分（encoder / classifier） =======
class ResNet18Split(nn.Module):
    def __init__(self, k_layers_encoder=3, num_classes=10):
        super().__init__()
        assert 1 <= k_layers_encoder <= 4
        base = models.resnet18(num_classes=num_classes)
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            *[base.layer1, base.layer2, base.layer3, base.layer4][:k_layers_encoder]
        )
        self.classifier_features = nn.Sequential(
            *[base.layer1, base.layer2, base.layer3, base.layer4][k_layers_encoder:],
            base.avgpool
        )
        self.fc = base.fc

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier_features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def encoder_state(self):
        return {k: v.clone() for k, v in self.state_dict().items() if k.startswith("encoder.")}

    def load_encoder_state(self, enc_state):
        sd = self.state_dict()
        sd.update(enc_state)
        self.load_state_dict(sd)

    def classifier_state(self):
        return {k: v.clone() for k, v in self.state_dict().items() if not k.startswith("encoder.")}

    def load_classifier_state(self, cls_state):
        sd = self.state_dict()
        sd.update(cls_state)
        self.load_state_dict(sd)

# ratio -> k
def ratio_to_k(ratio: float) -> int:
    k = int(4 * ratio)
    return max(1, min(4, k))

# encoder 输出通道数（ResNet18）
def enc_out_channels(k_layers_encoder: int) -> int:
    return [64, 128, 256, 512][k_layers_encoder - 1]

# ======= 本地训练 / 测试 =======
def local_train(model, trainloader, epochs=5, lr=0.01, device='cpu'):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model.encoder_state(), model.classifier_state()

@torch.no_grad()
def test_full_model(model, loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0

# 仅对 encoder 做等权 FedAvg
def fedavg_encoder(enc_state_list):
    keys = enc_state_list[0].keys()
    avg = {}
    for k in keys:
        avg[k] = sum([sd[k] for sd in enc_state_list]) / len(enc_state_list)
    return avg

# ======= 用最终 server encoder 抽“GAP向量”特征（保持原逻辑） =======
@torch.no_grad()
def extract_features_with_encoder(encoder_module, loader, device):
    encoder_module.eval()
    feats, labels = [], []
    gap = nn.AdaptiveAvgPool2d((1, 1))
    for x, y in loader:
        x = x.to(device)
        f = encoder_module(x)
        f = gap(f)               # [N, C, 1, 1]
        f = torch.flatten(f, 1)  # [N, C]
        feats.append(f.cpu())
        labels.append(y.clone())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

# ======= 新增：抽“空间特征” N×C×H×W（不给 GAP） =======
@torch.no_grad()
def extract_spatial_features_with_encoder(encoder_module, loader, device):
    encoder_module.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        f = encoder_module(x)    # [N, C, H, W]
        feats.append(f.cpu())
        labels.append(y.clone())
    feats = torch.cat(feats, dim=0)   # [N, C, H, W]
    labels = torch.cat(labels, dim=0) # [N]
    return feats, labels

# ======= 服务器端 MLP 分类器（可替换为更大模型） =======
class ServerMLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.net(x)

# ======= Server 端分类器训练：每 epoch 结束在 test 上评估 =======
def train_classifier_epochs(model, train_loader, val_loader, epochs, device, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    train_acc_hist, val_acc_hist = [], []

    for ep in tqdm(range(1, epochs + 1), desc="Server classifier epochs"):
        # ---- 训练一个 epoch ----
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        train_acc = correct / total if total > 0 else 0.0
        train_acc_hist.append(train_acc)

        # ---- 该 epoch 结束后在测试集上评估 ----
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        val_acc = v_correct / v_total if v_total > 0 else 0.0
        val_acc_hist.append(val_acc)

        tqdm.write(f"[Server-Head] Epoch {ep}: train_acc={train_acc:.4f} | test_acc={val_acc:.4f}")

    return train_acc_hist, val_acc_hist

# ======= CLI =======
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="按 4 个 layer 的比例切分 encoder（0<ratio<=1）。例如 0.5->前2层，0.75->前3层。")
    return parser.parse_args()

# ======= 主流程 =======
def main():
    args = parse_args()
    dataset = "Cifar10"
    num_clients = 20
    rounds = 30
    local_epochs = 5
    ratio = args.ratio
    k_enc = ratio_to_k(ratio)
    enc_dim = enc_out_channels(k_enc)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 分组（5 组）
    groups = [[i, i+5, i+10, i+15] for i in range(5)]

    # 数据一次性加载
    trainloaders, testloaders, _ = load_all_clients(dataset, num_clients)

    # 初始化：每组的 encoder、每个 client 的个性化 classifier、以及 server 的 encoder
    tmp = ResNet18Split(k_layers_encoder=k_enc).to(device)
    group_encoder_state = [tmp.encoder_state() for _ in range(5)]     # 组内共享 encoder
    client_classifier_state = [tmp.classifier_state() for _ in range(num_clients)]
    server_encoder_state = tmp.encoder_state()                         # 仅 server 持有

    round_client_avg_accs = []  # 记录每轮 average client test acc

    # ===== 联邦阶段：组内聚合更新组内 encoder；server 仅更新自己的 encoder =====
    for r in tqdm(range(rounds), desc="Federated rounds"):
        all_client_enc_updates = []

        for g, clients in enumerate(groups):
            enc_updates_this_group = []
            for cid in tqdm(clients, desc=f"Round {r+1} Group {g}", leave=False):
                model = ResNet18Split(k_layers_encoder=k_enc).to(device)
                # 本轮 client 使用“本组 encoder”作为初始化
                model.load_encoder_state(group_encoder_state[g])
                model.load_classifier_state(client_classifier_state[cid])

                enc_sd, cls_sd = local_train(model, trainloaders[cid],
                                             epochs=local_epochs, lr=0.01, device=device)

                # 更新 client 的个性化 classifier
                client_classifier_state[cid] = cls_sd
                # 收集本组与全局聚合用的 encoder
                enc_updates_this_group.append(enc_sd)
                all_client_enc_updates.append(enc_sd)

            # —— 组内聚合：只更新该组的 encoder —— #
            _group_enc = fedavg_encoder(enc_updates_this_group)
            group_encoder_state[g] = _group_enc

        # —— 全局聚合：只更新 server 自己的 encoder（不下发给 clients）—— #
        server_encoder_state = fedavg_encoder(all_client_enc_updates)
        tqdm.write(f"[Fed] Round {r+1} done. Updated: group_encoders & server_encoder (k={k_enc}, enc_dim={enc_dim}).")

        # ===== 每轮评估：average client test acc =====
        client_accs = []
        for g, clients in enumerate(groups):
            for cid in clients:
                eval_model = ResNet18Split(k_layers_encoder=k_enc).to(device)
                eval_model.load_encoder_state(group_encoder_state[g])          # 该轮后的组内 encoder
                eval_model.load_classifier_state(client_classifier_state[cid]) # 最新的个性化 classifier
                acc = test_full_model(eval_model, testloaders[cid], device=device)
                client_accs.append(acc)
        avg_acc = float(np.mean(client_accs))
        round_client_avg_accs.append(avg_acc)
        tqdm.write(f"[Round {r+1}] Avg Client Test Acc = {avg_acc:.4f}")

    # ===== 服务器分类器阶段（用 server 最终 encoder 抽特征）=====
    final_encoder_model = ResNet18Split(k_layers_encoder=k_enc).to(device)
    final_encoder_model.load_encoder_state(server_encoder_state)
    encoder_only = final_encoder_model.encoder  # 仅 encoder

    # 服务器训练集（并集）的“GAP向量”特征
    all_train_feats, all_train_labels = [], []
    for cid in range(num_clients):
        feats, labels = extract_features_with_encoder(encoder_only, trainloaders[cid], device)
        all_train_feats.append(feats); all_train_labels.append(labels)
    all_train_feats = torch.cat(all_train_feats, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)

    # 服务器测试集（并集）的“GAP向量”特征
    all_test_feats, all_test_labels = [], []
    for cid in range(num_clients):
        feats, labels = extract_features_with_encoder(encoder_only, testloaders[cid], device)
        all_test_feats.append(feats); all_test_labels.append(labels)
    all_test_feats = torch.cat(all_test_feats, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)

    # ======= 额外：导出“空间特征” N×C×H×W（供 ResNet50 tail 使用）=======
    os.makedirs("./server_features_spatial", exist_ok=True)
    # 训练集空间特征
    sp_train_feats, sp_train_labels = [], []
    for cid in range(num_clients):
        feats_sp, labels_sp = extract_spatial_features_with_encoder(encoder_only, trainloaders[cid], device)
        sp_train_feats.append(feats_sp); sp_train_labels.append(labels_sp)
    sp_train_feats = torch.cat(sp_train_feats, dim=0)   # [Ntr, C, H, W]
    sp_train_labels = torch.cat(sp_train_labels, dim=0) # [Ntr]

    # 测试集空间特征
    sp_test_feats, sp_test_labels = [], []
    for cid in range(num_clients):
        feats_sp, labels_sp = extract_spatial_features_with_encoder(encoder_only, testloaders[cid], device)
        sp_test_feats.append(feats_sp); sp_test_labels.append(labels_sp)
    sp_test_feats = torch.cat(sp_test_feats, dim=0)     # [Nte, C, H, W]
    sp_test_labels = torch.cat(sp_test_labels, dim=0)   # [Nte]

    C, H, W = sp_train_feats.shape[1], sp_train_feats.shape[2], sp_train_feats.shape[3]
    spatial_save_path = f"./server_features_spatial/{dataset}_ratio{ratio:.2f}_k{k_enc}_C{C}_H{H}_W{W}_train{sp_train_feats.shape[0]}_test{sp_test_feats.shape[0]}.npz"
    np.savez_compressed(
        spatial_save_path,
        X_train=sp_train_feats.numpy().astype(np.float32),
        y_train=sp_train_labels.numpy().astype(np.int64),
        X_test=sp_test_feats.numpy().astype(np.float32),
        y_test=sp_test_labels.numpy().astype(np.int64),
        C=np.array([C], dtype=np.int32),
        H=np.array([H], dtype=np.int32),
        W=np.array([W], dtype=np.int32),
        ratio=np.array([ratio], dtype=np.float32),
        k_enc=np.array([k_enc], dtype=np.int32),
    )
    tqdm.write(f"[Saved] Spatial features saved to: {spatial_save_path}")
    # ===============================================================

    # ======= 保持原有：保存“GAP向量”特征到磁盘（便于快速对比） =======
    os.makedirs("./server_features", exist_ok=True)
    vec_save_path = f"./server_features/{dataset}_ratio{ratio:.2f}_k{k_enc}_enc{enc_dim}_train{all_train_feats.shape[0]}_test{all_test_feats.shape[0]}.npz"
    np.savez_compressed(
        vec_save_path,
        X_train=all_train_feats.numpy().astype(np.float32),
        y_train=all_train_labels.numpy().astype(np.int64),
        X_test=all_test_feats.numpy().astype(np.float32),
        y_test=all_test_labels.numpy().astype(np.int64),
        ratio=np.array([ratio], dtype=np.float32),
        k_enc=np.array([k_enc], dtype=np.int32),
        enc_dim=np.array([enc_dim], dtype=np.int32)
    )
    tqdm.write(f"[Saved] Vector (GAP) features saved to: {vec_save_path}")

    # ======= 用“GAP向量”构建 DataLoader 训练/评估 server 端 MLP 分类器（原逻辑不变） =======
    server_train_loader = DataLoader(TensorDataset(all_train_feats, all_train_labels),
                                     batch_size=256, shuffle=True, num_workers=2)
    server_test_loader  = DataLoader(TensorDataset(all_test_feats,  all_test_labels),
                                     batch_size=256, shuffle=False, num_workers=2)

    # 服务器端 MLP 分类器训练 30 个 epoch（每轮打印训练 acc & 立即在 test 上评估）
    server_head = ServerMLPClassifier(in_dim=enc_dim, num_classes=10).to(device)
    server_train_acc_hist, server_test_acc_hist = train_classifier_epochs(
        server_head, server_train_loader, server_test_loader, epochs=30, device=device, lr=0.01
    )

    # ===== 训练后打印汇总 =====
    print("\n=== Server classifier TRAIN acc per epoch ===")
    for i, a in enumerate(server_train_acc_hist, 1):
        print(f"Epoch {i}: {a:.4f}")
    print("server_train_acc_list =", [float(a) for a in server_train_acc_hist])

    print("\n=== Server classifier TEST acc per epoch ===")
    for i, a in enumerate(server_test_acc_hist, 1):
        print(f"Epoch {i}: {a:.4f}")
    print("server_test_acc_list  =", [float(a) for a in server_test_acc_hist])

    # 最终在测试并集上的 acc（与上面列表最后一项一致）
    final_test_acc = server_test_acc_hist[-1]
    print(f"\n=== Final server classifier test acc on union(test) ===\n{final_test_acc:.4f}")

    # 每轮 average client test acc 列表
    print("\n=== All Rounds Avg Client Test Acc ===")
    for i, a in enumerate(round_client_avg_accs, 1):
        print(f"Round {i}: {a:.4f}")
    print("round_client_avg_accs =", [float(a) for a in round_client_avg_accs])

if __name__ == "__main__":
    main()