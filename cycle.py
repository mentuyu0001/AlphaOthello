import os
import glob
import random
import uuid
import multiprocessing as mp
import torch

from model import OthelloNet
from self_play import run_parallel_self_play_cycle
from train import train_model

def get_lr(cycle_count):
    """学習率のスケジューリング（例: サイクルが進むごとに少しずつ減らす）"""
    if cycle_count < 35:
        return 0.001
    elif cycle_count < 65:
        return 0.0005
    elif cycle_count < 100:
        return 0.00025
    elif cycle_count < 150:
        return 0.0001
    elif cycle_count < 250:
        return 0.00005
    else:
        return 0.00001
        

def run_infinite_cycle():
    # --- ご要望の各種パラメータ ---
    TOTAL_FILES_TARGET = 15        # プールしておく合計ファイル数
    FILES_TO_REPLACE = 3           # 1サイクルで入れ替えるファイル数
    NUM_GAMES_PER_FILE = 250       # 1ファイルあたりの試合数
    NUM_SIMULATIONS = 500          # MCTSシミュレーション回数
    EPOCHS = 3                    # 学習のエポック数
    BATCH_SIZE = 1024              # 学習のバッチサイズ
    
    # コア数 - 8 (最低1プロセスは確保する安全設計)
    NUM_PROCESSES = max(1, mp.cpu_count() - 8)
    
    MODEL_PATH = "current_model.pth"
    DATA_DIR = "data_buffer"       # 15個のファイルが溜まるフォルダ
    COUNT_FILE = "cycle_count.txt"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 初期モデル作成
    if not os.path.exists(MODEL_PATH):
        print(">>> 初期モデルを作成します...")
        dummy_model = OthelloNet()
        torch.save(dummy_model.state_dict(), MODEL_PATH)

    cycle_count = 1
    if os.path.exists(COUNT_FILE):
        try:
            with open(COUNT_FILE, "r") as f:
                cycle_count = int(f.read().strip())
                print(f">>> 前回までの記録を読み込みました。累計 第{cycle_count}サイクル から再開します。")
        except ValueError:
            print(">>> 記録ファイルが破損しているため、サイクル1から開始します。")
            cycle_count = 1
    
    # === 無限ループ開始 ===
    while True:
        print("\n" + "="*50)
        print(f"サイクル {cycle_count} 開始")
        print("="*50)
        
        # 1. データフォルダの現状確認
        current_files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
        num_current_files = len(current_files)
        
        files_to_generate = 0
        
        if num_current_files < TOTAL_FILES_TARGET:
            # 初回など、15ファイルに満たない場合は足りない分だけ作る
            files_to_generate = TOTAL_FILES_TARGET - num_current_files
            print(f"[フェーズ1] データ不足を検知。新たに {files_to_generate} ファイルを生成します。")
        else:
            # 既に15ファイルある場合は、ランダムに5つ選んで削除し、5つ新規作成する
            files_to_generate = FILES_TO_REPLACE
            files_to_delete = random.sample(current_files, FILES_TO_REPLACE)
            print(f"[フェーズ1] ランダムに {FILES_TO_REPLACE} ファイルを削除します。")
            for f in files_to_delete:
                os.remove(f)
                
        # 2. セルフプレイ（指定されたファイル数だけ順番に作成）
        force_cpu=True
        device_name = "CPU" if force_cpu else ("GPU (CUDA)" if torch.cuda.is_available() else "CPU")
        print(f"=== 自己対戦データ生成開始 ===")
        print(f"使用デバイス: {device_name}")
        print(f"総試合数: {NUM_GAMES_PER_FILE} | 1手あたりのMCTSシミュレーション回数: {NUM_SIMULATIONS}")
        print(f"使用プロセス数: {NUM_PROCESSES}")

        for i in range(files_to_generate):
            print(f"\n--- データ生成 {i+1}/{files_to_generate} ---")
            # 重複しないランダムなファイル名を生成
            new_filename = f"batch_{uuid.uuid4().hex[:8]}.pkl"
            out_path = os.path.join(DATA_DIR, new_filename)
            
            run_parallel_self_play_cycle(
                model_path=MODEL_PATH,
                num_games=NUM_GAMES_PER_FILE,
                num_simulations=NUM_SIMULATIONS,
                num_processes=NUM_PROCESSES,
                output_path=out_path,
                force_cpu=force_cpu
            )
            
        # 3. 学習フェーズ
        # ※ train_model内はデフォルトでGPU(cuda)を探して使用します
        print(f"\n[フェーズ2] ニューラルネットワークの学習 (Epochs: {EPOCHS}, Batch: {BATCH_SIZE})")
        train_model(
            data_dir=DATA_DIR,
            model_path=MODEL_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=get_lr(cycle_count)
        )

        cycle_count += 1
        with open(COUNT_FILE, "w") as f:
            f.write(str(cycle_count))
        
        print(f"サイクル {cycle_count} 完了！")

if __name__ == "__main__":
    mp.freeze_support()
    run_infinite_cycle()