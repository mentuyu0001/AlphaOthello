import os
import threading
import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch
import pyrev

# Windows環境でのTkinterのズレ・ボケ（DPIスケーリング問題）を強制解除
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from othello_env import OthelloEnv
from model import OthelloNet
from mcts import MCTS

class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AlphaZero Othello vs Human")
        
        # 盤面の余白（座標用）を追加したため、ウィンドウサイズを少し拡大
        self.root.geometry("580x840")
        self.root.resizable(False, False)

        self.model_path = "current_model.pth"
        self.num_simulations = 400
        self.mcts = None
        self.ai_color = None
        self.human_color = None
        
        self.env = OthelloEnv()
        self.env.reset()
        self.is_ai_thinking = False
        self.game_started = False
        
        self.setup_ui()
        
        self.info_label.config(text="AlphaZeroの脳みそをロード中...")
        self.root.update()
        self.load_ai()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=5)
        control_frame.pack()

        tk.Label(control_frame, text="手番を選んで対戦スタート", font=("Arial", 12)).pack(pady=2)
        
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack()
        
        self.btn_black = tk.Button(btn_frame, text="人間が先手 (黒●)", font=("Arial", 12), width=15,
                                   command=lambda: self.start_game(pyrev.BLACK))
        self.btn_black.pack(side=tk.LEFT, padx=10)
        
        self.btn_white = tk.Button(btn_frame, text="人間が後手 (白○)", font=("Arial", 12), width=15,
                                   command=lambda: self.start_game(pyrev.WHITE))
        self.btn_white.pack(side=tk.LEFT, padx=10)

        # --- ★評価値UIレイアウト（ご要望の通りに修正） ---
        self.eval_frame = tk.Frame(self.root, pady=10)
        self.eval_frame.pack()
        
        # 上段：形勢テキスト（真上に残す）
        self.eval_status_label = tk.Label(self.eval_frame, text="形勢: 互角", font=("Arial", 13, "bold"))
        self.eval_status_label.pack(side=tk.TOP, pady=(0, 5))
        
        # 下段：[You 〇%] [=== バー ===] [〇% AI]
        self.bar_container = tk.Frame(self.eval_frame)
        self.bar_container.pack(side=tk.TOP)

        # 左側（人間）
        self.human_pct_label = tk.Label(self.bar_container, text="You  50%", font=("Arial", 11, "bold"), fg="#0000cc", width=10, anchor="e")
        self.human_pct_label.pack(side=tk.LEFT, padx=5)

        # 中央（バー本体: AIを赤背景、人間を青で左から塗る）
        self.bar_width = 300
        self.eval_canvas = tk.Canvas(self.bar_container, width=self.bar_width, height=22, bg="#ff4d4d", highlightthickness=1, highlightbackground="gray")
        self.eval_canvas.pack(side=tk.LEFT)
        self.eval_rect = self.eval_canvas.create_rectangle(0, 0, self.bar_width / 2, 22, fill="#4d79ff", width=0)

        # 右側（AI）
        self.ai_pct_label = tk.Label(self.bar_container, text="50%  AI", font=("Arial", 11, "bold"), fg="#cc0000", width=10, anchor="w")
        self.ai_pct_label.pack(side=tk.LEFT, padx=5)
        # -----------------------------------

        self.info_label = tk.Label(self.root, text="モデルを読み込んでいます...", font=("Arial", 14, "bold"), fg="black")
        self.info_label.pack(pady=5)

        # 盤面描画用キャンバス（座標A-H, 1-8を描くための余白を設定）
        self.cell_size = 60
        self.board_size = self.cell_size * 8
        self.offset = 30 # 盤面周りの余白（ここに座標を書く）
        self.canvas_size = self.board_size + (self.offset * 2)
        
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="#ececec", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

    def load_ai(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("エラー", f"モデルファイル {self.model_path} が見つかりません！")
            self.root.destroy()
            return
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OthelloNet()
        model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        self.model = model
        self.info_label.config(text="準備完了！先手か後手を選んでください。")
        self.draw_board()

    def start_game(self, human_color):
        self.human_color = human_color
        self.ai_color = pyrev.WHITE if human_color == pyrev.BLACK else pyrev.BLACK
        self.mcts = MCTS(self.model)
        
        self.env.reset()
        self.game_started = True
        self.is_ai_thinking = False
        
        self.btn_black.config(state=tk.DISABLED)
        self.btn_white.config(state=tk.DISABLED)
        
        self.update_ui()
        self.process_turn()

    def _get_state_tensor(self, pos):
        state = np.zeros((2, 8, 8), dtype=np.float32)
        for coord in pos.get_player_disc_coords():
            state[0, coord // 8, coord % 8] = 1.0
        for coord in pos.get_opponent_disc_coords():
            state[1, coord // 8, coord % 8] = 1.0
            
        device = next(self.model.parameters()).device
        return torch.from_numpy(state).float().unsqueeze(0).to(device)

    def update_eval_bar(self):
        if not self.game_started or not hasattr(self, 'model'):
            return

        state_tensor = self._get_state_tensor(self.env.position)
        with torch.no_grad():
            _, v = self.model(state_tensor)
            val = v.item()

        current_turn = self.env.position.side_to_move
        if current_turn != self.human_color:
            val = -val

        win_rate = (val + 1.0) / 2.0
        human_percent = int(win_rate * 100)
        ai_percent = 100 - human_percent

        # バーの長さを更新（左から人間=青が伸びる）
        human_bar_width = int(self.bar_width * win_rate)
        self.eval_canvas.coords(self.eval_rect, 0, 0, human_bar_width, 22)

        # 左右のパーセンテージ表示を更新（ご要望のレイアウト）
        self.human_pct_label.config(text=f"You  {human_percent}%")
        self.ai_pct_label.config(text=f"{ai_percent}%  AI")

        # 上部のテキストを更新
        if human_percent >= 55:
            self.eval_status_label.config(text="形勢: あなたの優勢", fg="#0000cc")
        elif human_percent <= 45:
            self.eval_status_label.config(text="形勢: AIの優勢", fg="#cc0000")
        else:
            self.eval_status_label.config(text="形勢: 互角", fg="black")

    def draw_board(self):
        self.canvas.delete("all")
        
        # 盤面の背景（緑）を描画（座標余白の内側）
        self.canvas.create_rectangle(self.offset, self.offset, 
                                     self.offset + self.board_size, self.offset + self.board_size, 
                                     fill="#006400", outline="black", width=2)
        
        # --- ★追加：座標（A-H, 1-8）の描画 ---
        cols = "ABCDEFGH"
        rows = "12345678"
        for i in range(8):
            # 上部のアルファベット (A-H)
            cx = self.offset + i * self.cell_size + self.cell_size / 2
            cy = self.offset / 2
            self.canvas.create_text(cx, cy, text=cols[i], font=("Arial", 11, "bold"), fill="black")
            
            # 左部の数字 (1-8)
            cx2 = self.offset / 2
            cy2 = self.offset + i * self.cell_size + self.cell_size / 2
            self.canvas.create_text(cx2, cy2, text=rows[i], font=("Arial", 11, "bold"), fill="black")
        
        # グリッド線の描画
        for i in range(1, 8):
            # 縦線
            x = self.offset + i * self.cell_size
            self.canvas.create_line(x, self.offset, x, self.offset + self.board_size, fill="black")
            # 横線
            y = self.offset + i * self.cell_size
            self.canvas.create_line(self.offset, y, self.offset + self.board_size, y, fill="black")

        if not hasattr(self, 'env'):
            return

        pos = self.env.position
        current_turn = pos.side_to_move
        
        for coord in pos.get_player_disc_coords():
            color = "black" if current_turn == pyrev.BLACK else "white"
            self.draw_disc(coord, color)
            
        for coord in pos.get_opponent_disc_coords():
            color = "white" if current_turn == pyrev.BLACK else "black"
            self.draw_disc(coord, color)

        if self.game_started and not self.is_ai_thinking and current_turn == self.human_color:
            for move in pos.get_legal_moves():
                self.draw_legal_move_guide(move)

    def draw_disc(self, coord, color):
        coord = int(coord)
        col = coord % 8
        row = coord // 8
        # 余白(offset)を足して中心座標を計算
        cx = self.offset + col * self.cell_size + self.cell_size / 2
        cy = self.offset + row * self.cell_size + self.cell_size / 2
        r = self.cell_size / 2 - 5
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="black", width=1)

    def draw_legal_move_guide(self, coord):
        coord = int(coord)
        col = coord % 8
        row = coord // 8
        # 余白(offset)を足して中心座標を計算
        cx = self.offset + col * self.cell_size + self.cell_size / 2
        cy = self.offset + row * self.cell_size + self.cell_size / 2
        r = 5
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="yellow", outline="black")

    def update_ui(self):
        self.draw_board()
        self.update_eval_bar()
        
        if self.env.is_gameover():
            return

        current_turn = self.env.position.side_to_move
        black_count = self.env.position.get_disc_count_of(pyrev.BLACK)
        white_count = self.env.position.get_disc_count_of(pyrev.WHITE)
        score_text = f"黒● {black_count}  :  白○ {white_count}"

        if current_turn == self.human_color:
            self.info_label.config(text=f"あなたの番です\n{score_text}", fg="#0000cc")
        else:
            self.info_label.config(text=f"AlphaZeroが思考中...\n{score_text}", fg="#cc0000")
        self.root.update()

    def process_turn(self):
        if self.env.is_gameover():
            self.handle_gameover()
            return

        if self.env.position.side_to_move == self.ai_color:
            self.is_ai_thinking = True
            self.update_ui()
            threading.Thread(target=self.ai_worker, daemon=True).start()
        else:
            self.is_ai_thinking = False
            self.update_ui()

    def ai_worker(self):
        visit_counts = self.mcts.search(self.env, self.num_simulations, is_self_play=False)
        action = np.argmax(visit_counts)
        self.root.after(0, self.apply_move, action)

    def on_click(self, event):
        if not self.game_started or self.is_ai_thinking or self.env.is_gameover():
            return
            
        if self.env.position.side_to_move != self.human_color:
            return

        # クリック座標から余白(offset)を引いてマスを計算
        col = int((event.x - self.offset) // self.cell_size)
        row = int((event.y - self.offset) // self.cell_size)
        
        # 盤面の外（余白部分）をクリックした場合は無視
        if 0 <= col < 8 and 0 <= row < 8:
            action = row * 8 + col
            if action in self.env.get_legal_moves():
                self.apply_move(action)

    def apply_move(self, action):
        flip = self.env.position.calc_flip_discs(np.int8(action))
        self.env.position.do_move(np.int8(action), flip)
        
        while not self.env.is_gameover() and self.env.position.can_pass():
            self.env.position.do_pass()
            
        self.process_turn()

    def handle_gameover(self):
        self.game_started = False
        self.draw_board()
        self.update_eval_bar()
        
        black_count = self.env.position.get_disc_count_of(pyrev.BLACK)
        white_count = self.env.position.get_disc_count_of(pyrev.WHITE)
        
        self.info_label.config(text=f"ゲーム終了！\n黒● {black_count}  :  白○ {white_count}", fg="black")
        
        if black_count > white_count:
            winner = "黒●の勝ち！"
        elif white_count > black_count:
            winner = "白○の勝ち！"
        else:
            winner = "引き分け！"
            
        messagebox.showinfo("結果発表", f"黒: {black_count}枚\n白: {white_count}枚\n\n{winner}")
        
        self.btn_black.config(state=tk.NORMAL)
        self.btn_white.config(state=tk.NORMAL)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    torch.set_num_threads(1)
    
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()