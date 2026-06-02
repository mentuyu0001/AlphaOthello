import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class OthelloNet(nn.Module):
    """Dual Network"""
    def __init__(self, num_residual_blocks=5, num_channels=64):
        super().__init__()
        
        # 1. 入力層（2チャネルの盤面をnum_channelsチャネルに拡張）
        self.conv_input = nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 2. 残差タワー
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_residual_blocks)])
        
        # 3. Policyヘッド
        self.conv_policy = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 8 * 8, 64)  # 64マス分のLogitsを出力
        
        # 4. Valueヘッド
        self.conv_value = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 64)
        self.fc_value2 = nn.Linear(64, 1)  # -1〜1の1値を出力

    def forward(self, x):
        # xの形状: (Batch_size, 2, 8, 8)
        
        # 共通の特徴抽出
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policyの計算
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)  # 出力はLogits（Softmaxは損失関数側で処理）
        
        # Valueの計算
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v))  # tanhで -1.0 〜 1.0 に収束させる
        
        return p, v


# --- 動作確認用のメイン処理 ---
if __name__ == "__main__":
    print("OthelloNet の動作確認を開始します。")
    
    # 疑似的なミニバッチデータ（バッチサイズ4、2チャネル、8x8盤面）
    dummy_input = torch.randn(4, 2, 8, 8)
    
    # モデルのインスタンス化（残差ブロック5個、チャンネル数64）
    model = OthelloNet(num_residual_blocks=5, num_channels=64)
    model.eval()  # 推論モード
    
    with torch.no_grad():
        policy, value = model(dummy_input)
        
    print(f"入力形状:  {dummy_input.shape}")
    print(f"Policy出力形状 (各マスの評価値): {policy.shape}")  # (4, 64)
    print(f"Value出力形状 (局面の勝敗予測値):  {value.shape}")   # (4, 1)
    print("\nValueの予測値(-1.0 〜 1.0 に収まっているか確認）:")
    print(value.squeeze(-1).tolist())