import os
import multiprocessing as mp
import numpy as np
import torch
import pyrev

from othello_env import OthelloEnv
from model import OthelloNet
from mcts import MCTS

# --- 思考ロジック ---

def get_random_move(legal_moves):
    return np.random.choice(legal_moves)

def get_rule_based_move(env, legal_moves):
    corners = [pyrev.A1, pyrev.H1, pyrev.A8, pyrev.H8]
    # 1. 角が取れるなら最優先
    for move in legal_moves:
        if move in corners:
            return move
            
    # 2. 一番多く裏返せる手を選ぶ
    best_move = legal_moves[0]
    max_flips = -1
    for move in legal_moves:
        flips_bitboard = int(env.position.calc_flip_discs(np.int8(move)))
        flips_count = bin(flips_bitboard).count('1')
        if flips_count > max_flips:
            max_flips = flips_count
            best_move = move
    return best_move


# --- 1試合を実行するワーカー関数 ---

def play_eval_game(model_path, p_black, p_white, num_simulations):
    """
    指定されたプレイヤー構成で1試合だけ行い、結果を返します。
    戻り値: 1 (黒勝ち), -1 (白勝ち), 0 (引き分け)
    """
    # プロセスごとに独立してモデルをロード（安全な並列化）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    mcts = None

    if "alphazero" in (p_black, p_white):
        model = OthelloNet()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        mcts = MCTS(model)

    env = OthelloEnv()
    env.reset()

    while not env.is_gameover():
        legal_moves = env.get_legal_moves()
        
        # 現在の手番がどちらのプレイヤーか判定
        current_player = p_black if env.position.side_to_move == pyrev.BLACK else p_white

        if current_player == "alphazero":
            # ガチ対戦なので is_self_play=False (ノイズなし)
            visit_counts = mcts.search(env, num_simulations, is_self_play=False)
            action = np.argmax(visit_counts)
        elif current_player == "random":
            action = get_random_move(legal_moves)
        elif current_player == "rule":
            action = get_rule_based_move(env, legal_moves)

        env.step(action)

    # 黒視点での勝敗を計算
    score = env.position.get_score_from(pyrev.BLACK)
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


# --- 評価実行メイン関数 ---

def run_evaluation(model_path, opponent, num_games_per_side=50, num_simulations=50):
    num_processes = min(8, mp.cpu_count())
    print("="*50)
    print(f"🏆 AlphaZero 性能評価ベンチマーク")
    print(f"対戦相手: {opponent.upper()}")
    print(f"試合数: 先手{num_games_per_side}試合 / 後手{num_games_per_side}試合 (計{num_games_per_side * 2}試合)")
    print(f"MCTSシミュレーション回数: {num_simulations}")
    print(f"並列プロセス数: {num_processes}")
    print("="*50)
    
    print("\n[前半] AlphaZero (先手/黒) vs オポーネント (後手/白) を実行中...")
    args_black = [(model_path, "alphazero", opponent, num_simulations) for _ in range(num_games_per_side)]
    with mp.Pool(processes=num_processes) as pool:
        results_black = pool.starmap(play_eval_game, args_black)
        
    print("[後半] オポーネント (先手/黒) vs AlphaZero (後手/白) を実行中...")
    args_white = [(model_path, opponent, "alphazero", num_simulations) for _ in range(num_games_per_side)]
    with mp.Pool(processes=num_processes) as pool:
        results_white = pool.starmap(play_eval_game, args_white)

    # --- 集計 ---
    az_wins_as_black = results_black.count(1)
    az_losses_as_black = results_black.count(-1)
    draws_as_black = results_black.count(0)
    
    az_wins_as_white = results_white.count(-1) # 白勝ち(-1)がAZの勝ち
    az_losses_as_white = results_white.count(1)
    draws_as_white = results_white.count(0)
    
    total_games = num_games_per_side * 2
    total_wins = az_wins_as_black + az_wins_as_white
    total_losses = az_losses_as_black + az_losses_as_white
    total_draws = draws_as_black + draws_as_white
    win_rate = (total_wins / total_games) * 100

    print("\n" + "="*50)
    print(f"📊 最終結果レポート: 勝率 {win_rate:.1f}%")
    print("="*50)
    print(f"🟢 総合: {total_wins}勝 {total_losses}敗 {total_draws}分")
    print(f"⚫ 先手(黒)時: {az_wins_as_black}勝 {az_losses_as_black}敗 {draws_as_black}分")
    print(f"⚪ 後手(白)時: {az_wins_as_white}勝 {az_losses_as_white}敗 {draws_as_white}分")
    print("="*50)


if __name__ == "__main__":
    mp.freeze_support()
    
    # 評価設定
    MODEL_PATH = "current_model.pth"
    
    # "random" または "rule" を指定してください
    OPPONENT = "random" 
    
    NUM_GAMES_PER_SIDE = 50  # 黒50戦、白50戦（計100戦）
    NUM_SIMULATIONS = 50     # AlphaZeroの思考の深さ
    
    if not os.path.exists(MODEL_PATH):
        print(f"モデル {MODEL_PATH} がありません。学習を回してから実行してください。")
    else:
        run_evaluation(MODEL_PATH, OPPONENT, NUM_GAMES_PER_SIDE, NUM_SIMULATIONS)