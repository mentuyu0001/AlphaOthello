import os
import random
import numpy as np
import torch
import pyrev

from othello_env import OthelloEnv
from model import OthelloNet
from mcts import MCTS

# --- 各種プレイヤーの思考ロジック ---

def get_human_move(env, legal_moves):
    """人間のプレイヤーからの入力を受け付けます"""
    while True:
        valid_strs = [pyrev.coord_to_str(np.int8(m)) for m in legal_moves]
        print(f"あなたの番です。合法手: {', '.join(valid_strs)}")
        move_str = input("着手を入力してください (例: F5) > ").strip().upper()
        
        try:
            move = pyrev.parse_coord_str(move_str)
            if move in legal_moves:
                return move
            else:
                print("そこには打てません。合法手の中から選んでください。")
        except Exception:
            print("入力フォーマットが正しくありません。")

def get_random_move(env, legal_moves):
    """ランダムに手を決定します"""
    return random.choice(legal_moves)

def get_rule_based_move(env, legal_moves):
    """ルールベースAI: 角を優先し、それ以外は一番多くひっくり返せる手を選ぶ"""
    # 四隅の座標（A1, H1, A8, H8）
    corners = [pyrev.A1, pyrev.H1, pyrev.A8, pyrev.H8]
    
    # 1. 角が取れるなら最優先で取る
    for move in legal_moves:
        if move in corners:
            return move
            
    # 2. 角以外なら、一番多く裏返せる手を選ぶ（貪欲法）
    best_move = legal_moves[0]
    max_flips = -1
    
    for move in legal_moves:
        # calc_flip_discsは裏返る石のビットボード(64bit整数)を返すので、'1'の数を数える
        flips_bitboard = int(env.position.calc_flip_discs(np.int8(move)))
        flips_count = bin(flips_bitboard).count('1')
        
        if flips_count > max_flips:
            max_flips = flips_count
            best_move = move
            
    return best_move

def get_alphazero_move(env, legal_moves, mcts, num_simulations=50):
    """AlphaZero(MCTS)による本気の思考"""
    # ガチ対戦なので is_self_play=False にしてノイズを消す
    visit_counts = mcts.search(env, num_simulations, is_self_play=False)
    # 最も多くシミュレーションされた（＝最善と判断した）手を選ぶ
    return np.argmax(visit_counts)


# --- メイン対戦ループ ---

def play_game():
    print("="*40)
    print("オセロAI テスト対戦アリーナ")
    print("="*40)
    
    players = {
        "1": ("人間", get_human_move),
        "2": ("ランダムAI", get_random_move),
        "3": ("ルールベースAI (角優先＋強欲)", get_rule_based_move),
        "4": ("AlphaZero", get_alphazero_move)
    }
    
    print("\nプレイヤーを選択してください:")
    for k, (name, _) in players.items():
        print(f"  {k}: {name}")
        
    p_black = input("黒番(先手) を選んでください [1-4] > ").strip()
    p_white = input("白番(後手) を選んでください [1-4] > ").strip()
    
    if p_black not in players or p_white not in players:
        print("正しい番号を選択してください。終了します。")
        return

    name_black, func_black = players[p_black]
    name_white, func_white = players[p_white]
    
    # AlphaZeroが選ばれた場合のみ、モデルとMCTSをロードする
    mcts = None
    if p_black == "4" or p_white == "4":
        model_path = "current_model.pth"
        if not os.path.exists(model_path):
            print(f"エラー: {model_path} が見つかりません。先に学習を行ってください。")
            return
            
        print("\nAlphaZeroの頭脳をロード中...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OthelloNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        mcts = MCTS(model)
        
    env = OthelloEnv()
    env.reset()
    turn_count = 1
    
    print(f"\n【対戦開始】 黒(*): {name_black} vs 白(O): {name_white}\n")
    
    while not env.is_gameover():
        env.render()
        legal_moves = env.get_legal_moves()
        current_color_str = "黒(*)" if env.position.side_to_move == pyrev.BLACK else "白(O)"
        
        print(f"--- ターン {turn_count} : {current_color_str} の手番 ---")
        
        # 現在の手番に応じたプレイヤー関数を呼び出す
        if env.position.side_to_move == pyrev.BLACK:
            if p_black == "4":
                action = func_black(env, legal_moves, mcts, num_simulations=50) # 強さはここで調整
            else:
                action = func_black(env, legal_moves)
        else:
            if p_white == "4":
                action = func_white(env, legal_moves, mcts, num_simulations=50)
            else:
                action = func_white(env, legal_moves)
                
        action_str = pyrev.coord_to_str(np.int8(action))
        print(f">>> {current_color_str} は {action_str} に着手しました\n")
        
        env.step(action)
        turn_count += 1
        
    # --- 終局処理 ---
    print("="*40)
    print("ゲーム終了！")
    env.render()
    
    black_discs = env.position.get_disc_count_of(pyrev.BLACK)
    white_discs = env.position.get_disc_count_of(pyrev.WHITE)
    print(f"\n最終スコア - 黒(*): {black_discs}枚 | 白(O): {white_discs}枚")
    
    if black_discs > white_discs:
        print(f"🏆 黒({name_black}) の勝利！")
    elif white_discs > black_discs:
        print(f"🏆 白({name_white}) の勝利！")
    else:
        print("🤝 引き分け！")


if __name__ == "__main__":
    play_game()