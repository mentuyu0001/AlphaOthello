import numpy as np
import torch
import pyrev

def get_state_tensor(position, device):
    """pyrev.Position から 指定デバイス上の NN入力用テンソル (1, 2, 8, 8) を生成"""
    state = np.zeros((2, 8, 8), dtype=np.float32)
    for coord in position.get_player_disc_coords():
        state[0, coord // 8, coord % 8] = 1.0
    for coord in position.get_opponent_disc_coords():
        state[1, coord // 8, coord % 8] = 1.0
    
    # 確実に指定されたデバイス（GPU等）へ直接送り込む
    return torch.from_numpy(state).float().unsqueeze(0).to(device)

class Node:
    """MCTSの木を構成するノードクラス"""
    def __init__(self, parent=None, action_taken=None):
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.is_expanded = False
        
        self.N = np.zeros(64, dtype=np.float32)
        self.W = np.zeros(64, dtype=np.float32)
        self.Q = np.zeros(64, dtype=np.float32)
        self.P = np.zeros(64, dtype=np.float32)

    def expand(self, legal_moves, raw_logits):
        """ノードを展開し、合法手のみに絞ってPolicyを正規化"""
        self.is_expanded = True
        mask = np.full(64, -np.inf, dtype=np.float32)
        mask[legal_moves] = raw_logits[legal_moves]
        e_x = np.exp(mask - np.max(mask))
        self.P = e_x / e_x.sum()

    def get_best_action(self, legal_moves, c_puct=1.0):
        """PUCTアルゴリズムに基づき、次に探索すべき最適な手を返します"""
        sum_N = np.sum(self.N)
        U = c_puct * self.P * (np.sqrt(sum_N + 1.0) / (1.0 + self.N))
        puct_scores = self.Q + U
        
        final_scores = np.full(64, -np.inf, dtype=np.float32)
        final_scores[legal_moves] = puct_scores[legal_moves]
        
        return np.argmax(final_scores)


class MCTS:
    """モンテカルロ木探索を管理するクラス"""
    def __init__(self, model, c_puct=1.0):
        self.model = model
        self.model.eval()  # evalモードは初期化時に固定して無駄な呼び出しを削減
        self.c_puct = c_puct
        
        # 【超重要】外部からデバイスを受け取るのをやめ、
        # 渡されたモデルが現在載っているデバイス（CPU or GPU）を自動で100%正確に取得する
        self.device = next(model.parameters()).device

    def search(self, env, num_simulations, is_self_play=False):
        root_pos = env.position.copy()
        root_node = Node()
        
        legal_moves = list(root_pos.get_legal_moves())
        if not legal_moves:
            return np.zeros(64, dtype=np.float32)

        # 自動検知した正確なデバイスを強制的に適用
        state_tensor = get_state_tensor(root_pos, device=self.device)
        with torch.no_grad():
            p, v = self.model(state_tensor)
        
        raw_logits = p.squeeze(0).cpu().numpy()
        root_node.expand(legal_moves, raw_logits)
        
        if is_self_play:
            epsilon = 0.25
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                root_node.P[move] = (1 - epsilon) * root_node.P[move] + epsilon * noise[i]

        for _ in range(num_simulations):
            node = root_node
            scratch_pos = root_pos.copy()
            
            search_path = [node]
            actions_path = []
            colors_path = [scratch_pos.side_to_move]
            
            # 1. セレクション
            while node.is_expanded and not scratch_pos.is_gameover():
                current_legal_moves = list(scratch_pos.get_legal_moves())
                action = node.get_best_action(current_legal_moves, self.c_puct)
                
                action_int8 = np.int8(action)
                flip = scratch_pos.calc_flip_discs(action_int8)
                scratch_pos.do_move(action_int8, flip)
                
                while not scratch_pos.is_gameover() and scratch_pos.can_pass():
                    scratch_pos.do_pass()
                    
                if action not in node.children:
                    node.children[action] = Node(parent=node, action_taken=action)
                
                node = node.children[action]
                search_path.append(node)
                actions_path.append(action)
                colors_path.append(scratch_pos.side_to_move)
            
            # 2. エバリュエーション & エクスパンション
            if scratch_pos.is_gameover():
                score = scratch_pos.get_score()
                if score > 0:
                    v_val = 1.0
                elif score == 0:
                    v_val = 0.0
                else:
                    v_val = -1.0
            else:
                # 自動検知した正確なデバイスを強制的に適用
                state_tensor = get_state_tensor(scratch_pos, device=self.device)
                with torch.no_grad():
                    p, v = self.model(state_tensor)
                v_val = v.item()
                
                current_legal_moves = list(scratch_pos.get_legal_moves())
                node.expand(current_legal_moves, p.squeeze(0).cpu().numpy())
            
            # 3. バックプロパゲーション
            for i in reversed(range(len(actions_path))):
                parent_node = search_path[i]
                act = actions_path[i]
                
                if colors_path[i] != colors_path[i+1]:
                    v_val = -v_val
                
                parent_node.W[act] += v_val
                parent_node.N[act] += 1
                parent_node.Q[act] = parent_node.W[act] / parent_node.N[act]

        return root_node.N