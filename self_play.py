import os
import pickle
import time  # 時間計測用に追加
import multiprocessing as mp
import numpy as np
import torch
import pyrev

from othello_env import OthelloEnv
from model import OthelloNet
from mcts import MCTS

def collect_one_game(model_path, num_simulations, temperature_threshold=10, force_cpu=False):
    """1試合を自己対戦させ、データを生成する関数"""

    torch.set_num_threads(1)

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = OthelloNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    mcts = MCTS(model)
    env = OthelloEnv()
    
    game_history = []
    state = env.reset()
    turn_count = 0
    
    while not env.is_gameover():
        visit_counts = mcts.search(env, num_simulations, is_self_play=True)
        
        sum_N = np.sum(visit_counts)
        if sum_N == 0:
            break
            
        policy = visit_counts / sum_N
        current_color = env.position.side_to_move
        game_history.append((state, policy, current_color))
        
        if turn_count < temperature_threshold:
            action = np.random.choice(64, p=policy)
        else:
            action = np.argmax(visit_counts)
            
        state, done = env.step(action)
        turn_count += 1
        
    final_black_score = env.position.get_score_from(pyrev.BLACK)
    if final_black_score > 0:
        absolute_winner = pyrev.BLACK
    elif final_black_score < 0:
        absolute_winner = pyrev.WHITE
    else:
        absolute_winner = None
        
    game_data = []
    for s, p, color in game_history:
        if absolute_winner is None:
            v = 0.0
        elif color == absolute_winner:
            v = 1.0
        else:
            v = -1.0
        game_data.append((s, p, v))
        
    return game_data


def run_parallel_self_play(model_path, num_games, num_simulations, num_processes, output_path, force_cpu=False):
    """並列で自己対戦を実行し、かかった時間を計測します"""
    device_name = "CPU" if force_cpu else ("GPU (CUDA)" if torch.cuda.is_available() else "CPU")
    print(f"=== 自己対戦データ生成開始 ===")
    print(f"使用デバイス: {device_name}")
    print(f"総試合数: {num_games} | 1手あたりのMCTSシミュレーション回数: {num_simulations}")
    print(f"使用プロセス数: {num_processes}")
    
    # 計測開始
    start_time = time.time()
    
    # argsに force_cpu を追加して各プロセスに渡す
    args = [(model_path, num_simulations, 10, force_cpu) for _ in range(num_games)]
    all_dataset = []
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(collect_one_game, args)
        for game_data in results:
            all_dataset.extend(game_data)
            
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_dataset, f)
        
    # 計測終了
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"=== 自己対戦データ生成完了 ===")
    print(f"生成局面数: {len(all_dataset)}")
    print(f"⏱️ 所要時間: {elapsed_time:.2f} 秒 (1試合平均: {elapsed_time/num_games:.2f} 秒)")

def run_parallel_self_play_cycle(model_path, num_games, num_simulations, num_processes, output_path, force_cpu=False):
    """並列で自己対戦を実行し、かかった時間を計測します"""
    # argsに force_cpu を追加して各プロセスに渡す
    args = [(model_path, num_simulations, 10, force_cpu) for _ in range(num_games)]
    all_dataset = []
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(collect_one_game, args)
        for game_data in results:
            all_dataset.extend(game_data)
            
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_dataset, f)


if __name__ == "__main__":
    mp.freeze_support()
    
    MODEL_PATH = "current_model.pth"
    OUTPUT_PATH = "data/self_play_batch.pkl"
    NUM_GAMES = 250
    NUM_SIMULATIONS = 200
    NUM_PROCESSES = max(4, mp.cpu_count()-8)
    
    if not os.path.exists(MODEL_PATH):
        dummy_model = OthelloNet()
        torch.save(dummy_model.state_dict(), MODEL_PATH)
        
    # --- 検証用スイッチ ---
    # ここを True にすると CPU、False にすると GPU で動きます
    USE_CPU_ONLY = True 
    
    run_parallel_self_play(
        model_path=MODEL_PATH,
        num_games=NUM_GAMES,
        num_simulations=NUM_SIMULATIONS,
        num_processes=NUM_PROCESSES,
        output_path=OUTPUT_PATH,
        force_cpu=USE_CPU_ONLY
    )