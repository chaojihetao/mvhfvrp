import torch
import torch.nn as nn
from tensordict import TensorDict
import torch.nn.functional as F

from implement.utils import gather_by_index

# === 工具函数：线性层初始化 ===
def linear_layer(input_dim, output_dim, std=1e-2, bias=True):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim,bias=bias)
    nn.init.normal_(linear.weight, std=std)
    nn.init.zeros_(linear.bias)
    return linear

# === 问题特征编码器 ===
class ProblemNet(nn.Module):
    def __init__(self, input_dim, output_dim, head_num):
        super().__init__()

        layer1 = nn.Linear(input_dim, output_dim,bias=False)
        nn.init.uniform_(layer1.weight)

        self.model = nn.Sequential(
            layer1,
            nn.LayerNorm(output_dim),
            linear_layer(output_dim,output_dim),
            nn.ReLU(),
            linear_layer(output_dim, output_dim//head_num), # task embedding
            nn.LayerNorm(output_dim//head_num),
            linear_layer(output_dim//head_num, input_dim*output_dim),
        )

    def forward(self, problem_feature):
        # problem_feature: [B, D_problem]
        B, D = problem_feature.shape
        output = self.model(problem_feature).view(B, D, -1)  # [B, D_problem, H]
        return output

# === SwiGLU Feed-Forward Network ===
class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size, dim_feedforward):
        super().__init__()
        # SwiGLU 三个线性层
        # 1. Gate projection
        self.w1 = nn.Linear(hidden_size, dim_feedforward)
        # 2. Value projection
        self.w2 = nn.Linear(hidden_size, dim_feedforward)
        # 3. Output projection
        self.w3 = nn.Linear(dim_feedforward, hidden_size)

    def forward(self, x):
        # SwiGLU 逻辑: (Swish(xW1) * xW2) W3, 其中，F.silu 就是 Swish 激活函数
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        output = self.w3(gate * value)
        return output

# === Self-Attention ===
class SelfAttentionLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, hidden_size, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.scale = self.head_dim ** -0.5
        self.hidden_size = hidden_size
        self.dim_feedforward = hidden_size * 4

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, self.dim_feedforward)  # 前馈网络层(SwiGLU)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, src):
        # src: [B, N, H]
        B, N, H = src.shape
        
        # 1. 线性投影并分头
        # [B, N, H] -> [B, N, n_head, head_dim] -> [B, n_head, N, head_dim]
        q = self.q_proj(src).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(B, N, self.n_head, self.head_dim).transpose(1, 2)

        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # [B, n_head, N, head_dim] @ [B, n_head, head_dim, N] -> [B, n_head, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. 聚合 Value
        # [B, n_head, N, N] @ [B, n_head, N, head_dim] -> [B, n_head, N, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # 5. 拼接头并输出
        # [B, n_head, N, head_dim] -> [B, N, n_head, head_dim] -> [B, N, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, H)
        attn_output = self.out_proj(attn_output)

        # 6. Add & Norm
        src = self.norm1(src + attn_output)
        ffn_output = self.ffn(src)
        src = self.norm2(src + ffn_output)

        return src

# === Cross-Attention ===
class CrossAttentionLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, hidden_size, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.scale = self.head_dim ** -0.5
        self.hidden_size = hidden_size
        self.dim_feedforward = hidden_size * 4

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, self.dim_feedforward)  # 前馈网络层(SwiGLU)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, src, kv):
        # q: [B, V, H]
        # kv: [B, N, H]
        B, V, H = src.shape
        B, N, H = kv.shape
        
        # 1. 线性投影并分头
        # [B, N, H] -> [B, N, n_head, head_dim] -> [B, n_head, N, head_dim]
        q = self.q_proj(src).view(B, V, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, N, self.n_head, self.head_dim).transpose(1, 2)

        # 2. 计算注意力分数 (Scaled Dot-Product Attention)
        # [B, n_head, V, head_dim] @ [B, n_head, head_dim, N] -> [B, n_head, V, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. 聚合 Value
        # [B, n_head, V, N] @ [B, n_head, N, head_dim] -> [B, n_head, V, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # 5. 拼接头并输出
        # [B, n_head, V, head_dim] -> [B, V, n_head, head_dim] -> [B, V, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, V, H)
        attn_output = self.out_proj(attn_output)

        # 6. Add & Norm
        src = self.norm1(src + attn_output)
        ffn_output = self.ffn(src)
        src = self.norm2(src + ffn_output)

        return src

# === Node Self-Attention Encoder ===
class NodeSelfAttentionEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, n_head):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(hidden_size, n_head)  for _ in range(num_layers)])
    
    def forward(self, all_node_emb):
        out = all_node_emb
        for layer in self.layers:
            out = layer(out)
        return out  # [B, 1+N, H]

# === Vehicle Self-Attention Encoder ===
class VehicleSelfAttentionEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, n_head):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(hidden_size, n_head)  for _ in range(num_layers)])
    
    def forward(self, all_vehicle_node_emb):
        out = all_vehicle_node_emb
        for layer in self.layers:
            out = layer(out)
        return out  # [B, 1+V, H]

# === Vehicle and Node Cross Attation Encoder ===
class VehicleNodeCrossAttentionEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, n_head):
        super().__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(hidden_size, n_head)  for _ in range(num_layers)])
    
    def forward(self, vehicle_node_emb, all_node_emb):
        out = vehicle_node_emb  # [B, V, H]
        for layer in self.layers:
            out = layer(out, all_node_emb)
        return out  # [B, V, H]

# === Global Node Encoder ===
class GlobalNodeEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, n_head):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttentionLayer(hidden_size, n_head)  for _ in range(num_layers)])
        self.layers2 = nn.ModuleList([SelfAttentionLayer(hidden_size, n_head) for _ in range(num_layers)])
        self.layers1combine = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]) 
        self.layers2combine = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)]) 

    def forward(self, problem_emb, vehicle_emb, global_node_emb):
        '''
        problem_emb: [B, D, H]
        vehicle_emb: [B, V, H]
        global_node_emb: [B, 1+V+N, H]
        '''
        B, N, H = global_node_emb.shape
        B, D, H = problem_emb.shape
        B, V, H = vehicle_emb.shape

        out = global_node_emb  # [B, 1+V+N, H]
        out2 = global_node_emb # [B, 1+V+N, H]
        for i, layer in enumerate(self.layers):
            if i==0:
                out2 = torch.cat((out2, problem_emb, vehicle_emb), dim=1)   # [B, (1+V+N)+D+V, H]
            out = layer(out)
            out2 = self.layers2[i](out2)

            # combine
            out = out + self.layers1combine[i](out2[:, :N])  # [B, (1+V+N), H] = [B, (1+V+N), H] + [B, (1+V+N), H]
            if i != len(self.layers)-1:
                out2_ = out2[:, :N] + self.layers2combine[i](out)    # [B, (1+V+N), H] = [B, (1+V+N), H] + [B, (1+V+N), H]
                out2_ = torch.cat((out2_, out2[:, -D-V:]), dim=1)     # [B, (1+V+N)+D+V, H]
                out2 = out2_
        return out[:, :N]  # [B, (1+V+N), H]

# === Pointer Network Decoder ===
class PointerDecoder(nn.Module):
    def __init__(self, cur_status_dim, hidden_size, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.scale = self.head_dim ** -0.5

        self.Wq = nn.Linear(hidden_size+cur_status_dim, hidden_size, bias=False)   # hidden_size + cur_status_dim -> hidden_size
        # self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)     # 已经预计算，节省显存
        # self.Wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.combine = nn.Linear(hidden_size, hidden_size)

    def forward(self, 
                q_global,
                k_global,
                v_global,
                k_node,
                v_node,
                k_vehicle,
                v_vehicle,
                k_vehicle_node,
                v_vehicle_node,
                context_feature, 
                cache_pointer, 
                illegal_mask=None):
        '''
        q_global: [B, P, H]  P = num_starts or pomo size
        k_global: [B, n_head, 1+V+N, head_dim]
        v_global: [B, n_head, 1+V+N, head_dim]
        k_node: [B, n_head, 1+N, head_dim]
        v_node: [B, n_head, 1+N, head_dim]
        k_vehicle: [B, n_head, 1+V, head_dim]
        v_vehicle: [B, n_head, 1+V, head_dim]
        k_vehicle_node: [B, n_head, V, head_dim]
        v_vehicle_node: [B, n_head, V, head_dim]
        context_feature: [B, P, D_curr-2]
        cache_pointer: [B, H, 1+V+N]
        illegal_mask: [B*P, 1+V+N]
        
        '''
        # 获取维度信息
        B, P, H = q_global.shape
        B, n_head, V_N, head_dim = k_global.shape # V_N = 1+V+N
        B, _, V, _ = k_vehicle_node.shape
        illegal_mask = illegal_mask.view(B, P, V_N)  # [B*P, 1+V+N] -> [B, P, 1+V+N]

        # 1. 线性投影并分头
        q = torch.cat([q_global, context_feature], dim=-1)  # [B, P, H + D_curr-2]
        q = self.Wq(q)  # [B, P, H]
        q = q.view(B, P, self.n_head, self.head_dim)    # [B, P, n_head, head_dim]
        q = q.permute(0, 2, 1, 3)                       # [B, n_head, P, head_dim]

        # 1.1 q_global and global k,v
        scores_global = torch.matmul(q, k_global.transpose(-2, -1)) * self.scale  # [B, n_head, P, 1+V+N]
        if illegal_mask is not None:
            mask_expanded = illegal_mask.unsqueeze(1)   # [B, 1, P, 1+V+N]
            scores_global = scores_global.masked_fill(mask_expanded, float('-inf'))
        attn_weights_global = torch.softmax(scores_global, dim=-1)  # [B, n_head, P, 1+V+N]

        context_global = torch.matmul(attn_weights_global, v_global)  # [B, n_head, P, head_dim]
        context_global = context_global.permute(0, 2, 1, 3).contiguous()  # [B, P, n_head, head_dim]
        context_global = context_global.view(B, P, H)  # [B, P, H]

        # 1.2 q_global and node k,v
        scores_node = torch.matmul(q, k_node.transpose(-2, -1)) * self.scale  # [B, n_head, P, 1+N]
        attn_weights_node = torch.softmax(scores_node, dim=-1)  # [B, n_head, P, 1+N]
        context_node = torch.matmul(attn_weights_node, v_node)  # [B, n_head, P, head_dim]
        context_node = context_node.permute(0, 2, 1, 3).contiguous()  # [B, P, n_head, head_dim]
        context_node = context_node.view(B, P, H)  # [B, P, H]

        # 1.3 q_global and vehicle k,v
        scores_vehicle = torch.matmul(q, k_vehicle.transpose(-2, -1)) * self.scale  # [B, n_head, P, 1+V]
        attn_weights_vehicle = torch.softmax(scores_vehicle, dim=-1)  # [B, n_head, P, 1+V]
        context_vehicle = torch.matmul(attn_weights_vehicle, v_vehicle)  # [B, n_head, P, head_dim]
        context_vehicle = context_vehicle.permute(0, 2, 1, 3).contiguous()  # [B, P, n_head, head_dim]
        context_vehicle = context_vehicle.view(B, P, H)  # [B, P, H]

        # 1.4 q_global and vehicle_node k,v
        scores_vehicle_node = torch.matmul(q, k_vehicle_node.transpose(-2, -1)) * self.scale  # [B, n_head, P, V]
        attn_weights_vehicle_node = torch.softmax(scores_vehicle_node, dim=-1)  # [B, n_head, P, V]
        context_vehicle_node = torch.matmul(attn_weights_vehicle_node, v_vehicle_node)  # [B, n_head, P, head_dim]
        context_vehicle_node = context_vehicle_node.permute(0, 2, 1, 3).contiguous()  # [B, P, n_head, head_dim]
        context_vehicle_node = context_vehicle_node.view(B, P, H)  # [B, P, H]

        # 2. 合并四个部分的上下文
        context = context_global + context_node + context_vehicle + context_vehicle_node  # [B, P, H]
        mha_output = self.combine(context)  # [B, P, H]

        # 5. Pointer Network
        logits_pointer = torch.matmul(mha_output, cache_pointer)  # [B, P, 1+V+N]
        logits = logits_pointer.view(B*P, V_N)  # [B*P, 1+V+N]

        attn_scores = logits.float()
        return attn_scores

# === 完整的 Transformer 模型 ===
class TransformerModel(nn.Module):
    """完整的神经网络模型，包含特征提取器、价值头和策略头。
    """
    # ==== 模型初始化构建 ====
    def __init__(self, hidden_size: int = 256,
                       n_head: int = 8,
                       encoder_num_layers: int = 6,
                       state_feature_dims: dict = None,
                       ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.state_feature_dims = state_feature_dims

        depot_dim = state_feature_dims['depot_features']
        node_dim = state_feature_dims['node_features']
        vehicle_dim = state_feature_dims['vehicle_features']

        # === 1. 问题特征编码器 ===
        self.problem_net = ProblemNet(
            input_dim=state_feature_dims['problem_feature'],    # D
            output_dim=hidden_size,                             # H
            head_num=n_head
        )   # output: [B, D, H]

        # 车辆特征单独作为问题特征的一部分
        self.vehicle_embedder = nn.Linear(vehicle_dim, hidden_size)

        # === 2. 特征嵌入器 ===
        self.depot_embedder = nn.Linear(depot_dim, hidden_size)                          # 1
        self.vehicle_node_embedder = nn.Linear(depot_dim + vehicle_dim, hidden_size)     # V
        self.node_embedder = nn.Linear(node_dim, hidden_size)    # 不包含 depot           # N
        
        # === 3. Encoder ===
        self.node_self_attention_encoder = NodeSelfAttentionEncoder(num_layers=encoder_num_layers, hidden_size=hidden_size, n_head=n_head)
        self.vehicle_self_attention_encoder = VehicleSelfAttentionEncoder(num_layers=encoder_num_layers, hidden_size=hidden_size, n_head=n_head)
        self.vehicle_node_cross_attention_encoder = VehicleNodeCrossAttentionEncoder(num_layers=encoder_num_layers, hidden_size=hidden_size, n_head=n_head)
        self.global_node_encoder = GlobalNodeEncoder(num_layers=encoder_num_layers, hidden_size=hidden_size, n_head=n_head)

        # === 4. Pointer Decoder ===
        self.pointer_decoder = PointerDecoder(
            cur_status_dim=state_feature_dims['current_feature'],  # 减去当前选择节点、车辆的索引维度
            hidden_size=hidden_size,
            n_head=n_head
        )

        # === 5. key, value 投影层(提前缓存) ===
        self.Wk_n = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv_n = nn.Linear(hidden_size, hidden_size, bias=False)
        self._n_k_mha = None
        self._n_v_mha = None

        self.Wk_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self._v_k_mha = None
        self._v_v_mha = None

        self.Wk_nv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv_nv = nn.Linear(hidden_size, hidden_size, bias=False)
        self._nv_k_mha = None
        self._nv_v_mha = None

        self.Wk_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self._g_k_mha = None
        self._g_v_mha = None

        self._global_node_cache = None

    # ==== 2. 前期特征编码 ====
    def feature(self, td: TensorDict) -> dict:
        """
        负责静态信息的编码 (Pre-computation)
        """
        state_feat = td["state_feature"]
        problem_feature = state_feat["problem_feature"]  # [B, D]
        depot_feature = state_feat["depot_features"]    # [B, 1, D_depot]
        node_feature = state_feat["node_features"]      # [B, N, D_node]
        vehicle_feature = state_feat["vehicle_features"]  # [B, V, D_vehicle]
        depot_vehicle_feature = depot_feature.expand(-1, vehicle_feature.size(1), -1)  # [B, V, D_depot]
        vehicle_node_feature = torch.cat([depot_vehicle_feature, vehicle_feature], dim=-1)  # [B, V, D_depot + D_vehicle]

        # 1. node self attention encoder
        depot_emb = self.depot_embedder(depot_feature)  # [B, 1, H]
        node_emb = self.node_embedder(node_feature)     # [B, N, H]
        all_node_emb = torch.cat([depot_emb, node_emb], dim=1)  # [B, 1+N, H]
        all_node_feature_encoded = self.node_self_attention_encoder(all_node_emb)  # [B, 1+N, H]

        # 2. vehicle self attention encoder
        vehicle_node_emb = self.vehicle_node_embedder(vehicle_node_feature)  # [B, V, H]
        all_vehicle_node_emb = torch.cat([depot_emb, vehicle_node_emb], dim=1)  # [B, 1+V, H]
        all_vehicle_feature_encoded = self.vehicle_self_attention_encoder(all_vehicle_node_emb)  # [B, 1+V, H]

        # 3. node and vehicle cross attention encoder
        vehicle_node_cross_encoded = self.vehicle_node_cross_attention_encoder(
            vehicle_node_emb = vehicle_node_emb,  # [B, V, H]
            all_node_emb = all_node_emb           # [B, 1+N, H]
        )   # [B, V, H]

        # 4. global node encoder
        global_node_emb = torch.cat([depot_emb, vehicle_node_emb, node_emb], dim=1)  # [B, 1+V+N, H]
        problem_emb = self.problem_net(problem_feature)    # [B, D, H]
        vehicle_emb = self.vehicle_embedder(vehicle_feature)  # [B, V, H]

        global_node_feature_encoded = self.global_node_encoder(
            problem_emb=problem_emb,
            vehicle_emb=vehicle_emb,
            global_node_emb=global_node_emb
        )   # [B, 1+V+N, H]

        # 5. 缓存结果
        # 5.1 node self attention encoder
        B, N_1, H = all_node_feature_encoded.shape
        self._n_k_mha = self.Wk_n(all_node_feature_encoded).view(B, N_1, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, 1+N, head_dim]
        self._n_v_mha = self.Wv_n(all_node_feature_encoded).view(B, N_1, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, 1+N, head_dim]

        # 5.2 vehicle self attention encoder
        B, V_1, H = all_vehicle_feature_encoded.shape
        self._v_k_mha = self.Wk_v(all_vehicle_feature_encoded).view(B, V_1, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, 1+V, head_dim]
        self._v_v_mha = self.Wv_v(all_vehicle_feature_encoded).view(B, V_1, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, 1+V, head_dim]

        # 5.3 vehicle and node cross attention encoder
        B, V, H = vehicle_node_cross_encoded.shape
        self._nv_k_mha = self.Wk_nv(vehicle_node_cross_encoded).view(B, V, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, V, head_dim]
        self._nv_v_mha = self.Wv_nv(vehicle_node_cross_encoded).view(B, V, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, V, head_dim]

        # 5.4 global node encoder
        B, V_N, H = global_node_feature_encoded.shape  # V_N = 1+V+N
        self._g_k_mha = self.Wk_g(global_node_feature_encoded).view(B, V_N, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, 1+V+N, head_dim]
        self._g_v_mha = self.Wv_g(global_node_feature_encoded).view(B, V_N, self.n_head, H // self.n_head).transpose(1, 2)  # [B, n_head, 1+V+N, head_dim]
        self._global_node_cache = global_node_feature_encoded  # [B, 1+V+N, H]

    # ==== 3. 策略网络前向计算 ====
    def policy(self, current_feature, illegal_mask) -> torch.Tensor:
        """
        策略网络的前向计算，生成动作的注意力分数。
        current_feature: [B*P, D_curr], P = num_starts or pomo size
        illegal_mask: [B*P, 1+V+N], True表示不可行
        return: attn_scores: [B*P, (1+V+N)]
        """

        # ==== 1. Feature Extraction ====
        k_global = self._g_k_mha  # [B, n_head, 1+V+N, head_dim]
        v_global = self._g_v_mha  # [B, n_head, 1+V+N, head_dim]
        k_node = self._n_k_mha    # [B, n_head, 1+N, head_dim]
        v_node = self._n_v_mha    # [B, n_head, 1+N, head_dim]
        k_vehicle = self._v_k_mha  # [B, n_head, 1+V, head_dim]
        v_vehicle = self._v_v_mha  # [B, n_head, 1+V, head_dim]
        k_vehicle_node = self._nv_k_mha  # [B, n_head, V, head_dim]
        v_vehicle_node = self._nv_v_mha  # [B, n_head, V, head_dim]
        global_node_feature_encoded = self._global_node_cache  # [B, 1+V+N, H]


        # 1.1 expand for POMO and get context
        B, V_N, H = global_node_feature_encoded.shape  # V_N = 1+V+N
        B_P, D_curr = current_feature.shape
        P = B_P // B  # P = num_starts or pomo size
        current_feature = current_feature.view(B, P, D_curr)  # [B, P, D_curr]

        # context gather
        current_action_idx = current_feature[..., :1]  # [B, P, 1], 当前动作对应的节点索引
        global_node_feature_encoded_expanded = global_node_feature_encoded.unsqueeze(1).expand(-1, P, -1, -1)  # [B, P, 1+V+N, H]
        q_global = gather_by_index(
            global_node_feature_encoded_expanded,  # [B, P, 1+V+N, H]
            current_action_idx.long(),  # [B, P, 1]
            dim=2
        )  # [B, P, H]

        # context embeddings
        cur_status_feature = current_feature[..., 2:]  # 除去第一个索引特征的车辆状态, [B, P, D_curr-2]

        # pointer decoder
        score = self.pointer_decoder(
            q_global=q_global,  # [B, P, H]
            k_global=k_global,
            v_global=v_global,
            k_node=k_node,
            v_node=v_node,
            k_vehicle=k_vehicle,
            v_vehicle=v_vehicle,
            k_vehicle_node=k_vehicle_node,
            v_vehicle_node=v_vehicle_node,
            context_feature=cur_status_feature,  # [B, P, D_curr-2]
            cache_pointer=global_node_feature_encoded.transpose(1, 2),  # [B, H, 1+V+N]
            illegal_mask=illegal_mask
        )  # [B*P, 1+V+N]

        # mask
        score = torch.masked_fill(score, illegal_mask, float('-inf'))

        return score