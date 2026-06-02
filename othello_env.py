import numpy as np
import pyrev

class OthelloEnv:
    def __init__(self):
        self.position = pyrev.Position()

    def reset(self):
        """ゲームを初期状態にリセットし、初期盤面テンソルを返します。"""
        self.position = pyrev.Position()
        return self.get_state()

    def get_state(self):
        """
        現在の盤面状態をニューラルネットワークに入力可能なテンソル形式で返します。
        形状: (2, 8, 8) - チャネルファースト (PyTorch標準)
        ・チャネル 0: 現在の手番の石の配置 (1.0: 石あり, 0.0: なし)
        ・チャネル 1: 相手の手番の石の配置 (1.0: 石あり, 0.0: なし)
        """
        state = np.zeros((2, 8, 8), dtype=np.float32)

        # PyRevの高速なイテレータを使って石の座標を取得し、テンソルに配置
        for coord in self.position.get_player_disc_coords():
            state[0, coord // 8, coord % 8] = 1.0

        for coord in self.position.get_opponent_disc_coords():
            state[1, coord // 8, coord % 8] = 1.0

        return state

    def get_legal_moves(self):
        """現在の手番の合法手リスト(0〜63の整数配)を返します。"""
        # BoardCoordinateIterator を Python のリストに変換
        return list(self.position.get_legal_moves())

    def step(self, action):
        """
        指定されたアクション(0〜63の座標)を実行します。
        戻り値: (次の状態, ゲーム終了フラグ)
        """
        # 合法手チェックと着手
        # do_move_at は合法手なら着手して True を返し、手番を交代します
        success = self.position.do_move_at(np.int8(action))
        if not success:
            raise ValueError(f"非合法な着手です: {action} (文字列: {pyrev.coord_to_str(np.int8(action))})")

        # パスの自動処理
        # 着手後にゲームが終了しておらず、かつ次のプレイヤーが打てる手がない（＝パスが必要）場合、
        # 自動的にパスを実行して手番をさらに交代させます。連続パスによる終局までループします。
        while not self.position.is_gameover() and self.position.can_pass():
            self.position.do_pass()

        return self.get_state(), self.position.is_gameover()

    def is_gameover(self):
        """ゲームが終了しているかどうかを返します。"""
        return self.position.is_gameover()

    def get_game_result(self):
        """
        Valueの3クラス分類(クロスエントロピー)を見据えたゲーム結果を返します。
        戻り値:
            1: 現在の手番のプレイヤーの勝ち (石差がプラス)
            0: 引き分け (石差が0)
            -1: 現在の手番のプレイヤーの負け (石差がマイナス)
            None: ゲーム進行中
        """
        if not self.position.is_gameover():
            return None

        # get_score() は現在の手番から見た石差を返します
        score = self.position.get_score()
        if score > 0:
            return 1  # 勝ち
        elif score == 0:
            return 0  # 引き分け
        else:
            return -1  # 負け

    def render(self):
        """盤面をコンソールに表示します（デバッグ用）。"""
        print(self.position)


# --- 動作確認用のメイン処理 ---
if __name__ == "__main__":
    print("OthelloEnv の動作確認を開始します。")
    env = OthelloEnv()
    state = env.reset()
    
    print("初期盤面:")
    env.render()
    
    # ランダムにプレイして終局まで動かすテスト
    import random
    
    turn_count = 1
    while not env.is_gameover():
        legal_moves = env.get_legal_moves()
        # パスは環境側で自動処理されるため、ここには必ず打てる手が存在します
        action = random.choice(legal_moves)
        
        action_str = pyrev.coord_to_str(np.int8(action))
        current_color = "黒(*)" if env.position.side_to_move == pyrev.BLACK else "白(O)"
        print(f"\n--- 手番 {turn_count}: {current_color} が {action_str} ({action}) に着手 ---")
        
        state, done = env.step(action)
        env.render()
        turn_count += 1
        
    print("\n=== ゲーム終了 ===")
    result = env.get_game_result()
    # 最後の状態での手番プレイヤーから見た結果が表示されるため、実際の最終石数を確認
    black_discs = env.position.get_disc_count_of(pyrev.BLACK)
    white_discs = env.position.get_disc_count_of(pyrev.WHITE)
    print(f"最終石数 - 黒(*): {black_discs}, 白(O): {white_discs}")
    
    if result == 1:
        winner = "現在の手番の勝ち"
    elif result == 0:
        winner = "引き分け"
    else:
        winner = "現在の手番の負け"
    print(f"結果コード: {result} ({winner})")