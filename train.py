import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob

from model import OthelloNet

class OthelloDataset(Dataset):
    """自己対戦データを読み込み、オセロの対称性（8パターン）を利用して拡張するデータセット"""
    def __init__(self, data_dir):
        self.raw_data = []
        # フォルダ内のすべての .pkl ファイルを検索して合体させる
        file_list = glob.glob(os.path.join(data_dir, "*.pkl"))
        for file_path in file_list:
            with open(file_path, "rb") as f:
                self.raw_data.extend(pickle.load(f))
            
        self.extended_data = []
        self._augment_data()

    def _augment_data(self):
        """1つの局面データを回転・反転させて8倍に増殖させます"""
        for state, policy, value in self.raw_data:
            # stateの形状: (2, 8, 8)
            # policyの形状: (64,) -> 2次元 (8, 8) に戻して一緒に回転させる
            policy_2d = policy.reshape(8, 8)
            
            for i in range(4):
                # 1. 0度, 90度, 180度, 270度の回転
                rot_state = np.rot90(state, i, axes=(1, 2))
                rot_policy = np.rot90(policy_2d, i)
                
                self.extended_data.append((
                    rot_state.copy(),
                    rot_policy.flatten().copy(),
                    value
                ))
                
                # 2. 左右反転（鏡状態）を加えた4パターン
                flip_state = np.flip(rot_state, axis=2)
                flip_policy = np.flip(rot_policy, axis=1)
                
                self.extended_data.append((
                    flip_state.copy(),
                    flip_policy.flatten().copy(),
                    value
                ))

    def __len__(self):
        return len(self.extended_data)

    def __getitem__(self, idx):
        state, policy, value = self.extended_data[idx]
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy),
            torch.FloatTensor([value]) # MSELossに合わせるため形状を (1,) にする
        )


def train_model(data_dir, model_path, epochs=10, batch_size=1024, lr=0.001):
    """溜まった自己対戦データを使ってモデルを訓練し、最新モデルを保存します"""
    if not os.path.exists(data_dir):
        print(f"学習データが見つかりません: {data_dir}。先に自己対戦を行ってください。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 訓練開始 (使用デバイス: {device}) ===")
    
    dataset = OthelloDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"読み込みファイル数: {len(glob.glob(os.path.join(data_dir, '*.pkl')))} 個")
    print(f"オリジナル局面数: {len(dataset.raw_data)} -> 8倍拡張後: {len(dataset)}")

    # モデルの読み込み
    model = OthelloNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    model.train() # 訓練モード

    # オプティマイザと損失関数（AlphaZero本家仕様）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss() # Value用（平均二乗誤差）

    for epoch in range(epochs):
        sum_total_loss = 0.0
        sum_policy_loss = 0.0
        sum_value_loss = 0.0
        
        for states, count_targets, value_targets in dataloader:
            states = states.to(device)
            count_targets = count_targets.to(device)
            value_targets = value_targets.to(device)

            # 順伝播
            p_out, v_out = model(states)

            # 1. Policyロス: MCTS訪問比率（教師）とNN出力（生のLogits）の交差エントロピー
            loss_policy = torch.nn.functional.cross_entropy(p_out, count_targets)

            # 2. Valueロス: 本家仕様のMSE（平均二乗誤差）
            loss_value = mse_loss(v_out, value_targets)

            # 総損失（PolicyとValueのロスの合算）
            loss = loss_policy + loss_value

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_total_loss += loss.item()
            sum_policy_loss += loss_policy.item()
            sum_value_loss += loss_value.item()

        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {sum_total_loss/len(dataloader):.4f} "
              f"(Policy: {sum_policy_loss/len(dataloader):.4f}, Value: {sum_value_loss/len(dataloader):.4f})")

    # 訓練終了後、モデルを上書き保存
    torch.save(model_state := model.state_dict(), model_path)
    print(f"最新のモデルを {model_path} に保存しました。\n")


if __name__ == "__main__":
    DATA_PATH = "data/self_play_batch.pkl"
    MODEL_PATH = "current_model.pth"
    
    # 単体テスト用（データがあれば回ります）
    train_model(DATA_PATH, MODEL_PATH, epochs=3, batch_size=32)