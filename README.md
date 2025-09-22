# FedTEST

fedavg.py：经典FEDAVG算法
group.py：分组聚合FEDAVG算法
split.py：encoder + classifier算法，运行方式python split.py --ratio 0.75/0.5/0.25
split最后客户端feature会存储在
- server_features（全局池化后的特征 [N, 256, 1, 1]）
- server_features（空间特征，不做GAP 对于0.75是 [N, 256, 2, 2]）