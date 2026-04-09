import torch
from typing import Optional
from tensordict.tensordict import TensorDict

from framework.env import EnvBase
from implement.generator import MTHFVRPGenerator
from implement.utils import gather_by_index, get_distance_by_matrix


class MTHFVRPEnv(EnvBase):

    def __init__(
        self,
        generator: MTHFVRPGenerator = None,
        batch_size: Optional[list] = None,
        device: str = "cpu",
    ):
        super().__init__(batch_size=batch_size)

        assert generator is not None, "Either generator or generator_params must be provided."
        self.generator = generator
        self.device = device

    def to(self, device: str):
        self.device = device
        return self

    def _reset(
        self,
        problem: TensorDict,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        td = problem
        device = td.device
        batch_size = td.batch_size
        # Demands: linehaul (C) and backhaul (B). Backhaul defaults to 0
        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]],
            dim=1,
        )
        demand_backhaul = td.get(
            "demand_backhaul",
            torch.zeros_like(td["demand_linehaul"]),
        )
        demand_backhaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), demand_backhaul], dim=1
        )
        # Backhaul class (MB). 1 is the default backhaul class
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32),
        )

        # Time windows (TW). Defaults to [0, inf] and service time to 0
        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

        # Open (O) route. Defaults to 0
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # Distance limit (L). Defaults to inf
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        # Distance Matrix
        distance_matrix = td.get('distance_matrix', torch.zeros((*batch_size, td["locs"].shape[1], td["locs"].shape[1])))

        # Heterogeneous fleet (HF). Defaults to False
        heterogeneous_fleet = td.get(
            "heterogeneous_fleet", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # 构建route_vehicle的模板，形状为 (total_routes,)
        num_vehicle_types = td["vehicle_capacity"].shape[-1]
        route_vehicle_template = []
        for vehicle_type_idx, count in enumerate(td["available_vehicles"][0]):
            count_int = int(count.item())
            route_vehicle_template.extend([vehicle_type_idx] * count_int)

        route_vehicle_template = torch.tensor(route_vehicle_template, dtype=torch.long, device=device)  # (total_routes,)
        
        # 维护状态信息
        state = TensorDict(
            {
                # local info
                "locs": td["locs"],
                "distance_matrix": distance_matrix,
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "time_windows": time_windows,
                "service_time": service_time,
                "open_route": open_route,
                "heterogeneous_fleet": heterogeneous_fleet,
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),

                # vehicle info
                "available_vehicles": td.get(
                    "available_vehicles",
                    torch.ones((*batch_size, num_vehicle_types), dtype=torch.int32, device=device),
                ),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", 
                    torch.ones((*batch_size, num_vehicle_types), device=device)
                ),
                "vehicle_fixed_cost": td.get(
                    "vehicle_fixed_cost", 
                    torch.zeros((*batch_size, num_vehicle_types), device=device)
                ),
                "vehicle_variable_cost": td.get(
                    "variable_cost", 
                    torch.ones((*batch_size, num_vehicle_types), device=device)
                ),

                # current route info
                "current_action": torch.zeros(*batch_size, dtype=torch.long, device=device),  # 当前选择的节点
                "current_vehicle_type": torch.zeros(*batch_size, dtype=torch.long, device=device),  # 当前选择的车辆类型

                "vehicle_used_num": torch.zeros((*batch_size, num_vehicle_types), dtype=torch.float32, device=device),  # 车辆被使用的次数
                "current_length": torch.zeros(*batch_size, dtype=torch.float32, device=device),  # 当前路径长度
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),  # 当前路径时间
                "current_used_capacity_lh": torch.zeros(*batch_size, dtype=torch.float32, device=device),  # 当前路径线haul使用容量
                "current_used_capacity_bh": torch.zeros(*batch_size, dtype=torch.float32, device=device),  # 当前路径回haul使用容量

                "visited": torch.zeros((*batch_size, td["locs"].shape[-2]), dtype=torch.bool, device=device),  # 访问状态
            },
            batch_size=batch_size,
            device=device,
        )

        state = self.get_action_mask(state)     # Get the legal action mask
        return state
    
    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        # ==== 准备数据 ====
        B, plus_N, _ = td["locs"].shape
        N = plus_N - 1  # 客户节点数量
        V = td["vehicle_capacity"].shape[-1]

        # === 当前状态判定 ===
        curr_action = td["current_action"]  # [B]
        at_physical_depot = (curr_action == 0)                        # 状态 A, 处在depot
        at_virtual_vehicle = (curr_action >= 1) & (curr_action <= V)    # 状态 B, 处在vehicle type
        at_customer = (curr_action > V)                               # 状态 C, 处在customer node

        # 实际节点索引
        real_curr_node = (curr_action - V).clamp(min=0) * at_customer.long()  # [B]
        depot_idx = torch.zeros(B, dtype=torch.long, device=td.device) # [B]
        all_nodes = torch.arange(1, plus_N, device=td.device).unsqueeze(0).expand(B, N)  # [B, N]

        # 距离
        d_ij = get_distance_by_matrix(td['distance_matrix'], real_curr_node.unsqueeze(-1), all_nodes).squeeze()  # [B, N]
        d_j0 = get_distance_by_matrix(td['distance_matrix'], all_nodes, depot_idx.unsqueeze(-1)).squeeze()  # [B, N]

        # ==== Mask: [1 + V + N] ==== 
        # Not visited (V)
        is_unvisited = ~td['visited'][:, 1:] # [B, N]

        # Time constraint (TW)
        speed = td["speed"] # [B, 1]
        arrival_time = td["current_time"].unsqueeze(-1) + (d_ij / speed) # [B, N]
        can_reach_cust = arrival_time <= td["time_windows"][:, 1:, 1]

        service_t = td["service_time"][:, 1:]
        start_service_t = torch.max(arrival_time, td["time_windows"][:, 1:, 0])
        can_back_depot = (start_service_t + service_t + (d_j0 / speed)) * (~td["open_route"]) <= td["time_windows"][:, :1, 1]

        # Distance limit (L)
        can_finish_dist = (td["current_length"].unsqueeze(-1) + d_ij + (d_j0 * ~td["open_route"])) <= td["distance_limit"]

        # Capacity constraints linehaul (C) and backhaul (B)
        v_type = td["current_vehicle_type"]
        v_cap = gather_by_index(td["vehicle_capacity"], v_type).unsqueeze(-1) # [B, 1]
        not_exceed_lh = (td["current_used_capacity_lh"].unsqueeze(-1) + td["demand_linehaul"][:, 1:]) <= v_cap
        not_exceed_bh = (td["current_used_capacity_bh"].unsqueeze(-1) + td["demand_backhaul"][:, 1:]) <= v_cap        

        # Backhaul class 1 (classical backhaul) (B)
        lh_missing = ((td["demand_linehaul"][:, 1:] * is_unvisited).sum(dim=-1, keepdim=True) > 0) # [B, 1]
        is_carrying_bh = (td["current_used_capacity_bh"] > 0).unsqueeze(-1) # [B, 1]
    
        meets_bh_1 = (lh_missing & not_exceed_lh & ~is_carrying_bh & (td["demand_linehaul"][:, 1:] > 0)) | \
                    (not_exceed_bh & (td["demand_backhaul"][:, 1:] > 0))
        
        # Backhaul class 2 (mixed pickup and delivery / mixed backhaul) (MB)
        meets_bh_2 = not_exceed_lh & not_exceed_bh & \
                    (td["demand_linehaul"][:, 1:] <= (v_cap - td["current_used_capacity_bh"].unsqueeze(-1)))

        backhaul_class = td["backhaul_class"]  # [B, 1]
        meets_demand =(
            (backhaul_class == 1) & meets_bh_1 |
            (backhaul_class == 2) & meets_bh_2
        )

        # 合法客户节点
        valid_customers = is_unvisited & can_reach_cust & can_back_depot & can_finish_dist & meets_demand  # [B, N]

        # 构建mask
        # 0: depot
        # 1~V: vehicle types 
        # V+1~V+N: customer nodes
        final_mask = torch.zeros((B, 1 + V + N), dtype=torch.bool, device=td.device)

        # ==== 1. 状态 A: 处在 depot，可以选择 vehicle type ====
        num_avail = td["available_vehicles"] - td["vehicle_used_num"]
        final_mask[:, 1:1+V] = at_physical_depot.unsqueeze(-1) & (num_avail > 0)

        # ==== 2. 状态 B或C， 选择 customer node ====
        can_go_cust = at_virtual_vehicle | at_customer
        final_mask[:, 1+V:] = can_go_cust.unsqueeze(-1) & valid_customers

        # ==== 3. 状态C， 选择回 depot ====
        final_mask[:, 0] = at_customer

        td.set("legal_action_mask", final_mask)  # [B, 1+V+N]
        return td
    
    def _step(self, td: TensorDict, action: int) -> TensorDict:
        # 1. ==== 解码动作 =====
        # action 是一个 [B] 大小的张量，值为 [0, 1+V+N]
        # 0 代表 depot
        # 1~V 代表 vehicle type
        # V+1~V+N 代表 customer nodes
        B = action.shape[0]
        V = td["vehicle_capacity"].shape[-1]
        N = td["locs"].shape[-2] - 1

        # 解析动作类型
        is_return_depot = (action == 0)                     # 选择回 depot
        is_select_vehicle = (action >= 1) & (action <= V)   # 选择 vehicle type
        is_select_customer = (action > V)                   # 选择 customer node

        # 物理节点索引
        target_phys_idx = (action - V).clamp(min=0) * is_select_customer.long()  # [B]

        prev_action = td["current_action"]
        prev_is_cust = (prev_action > V).float()
        prev_phys_idx = (prev_action - V).clamp(min=0) * prev_is_cust.long()

        # 2. ==== 计算物理增量（距离、时间、容量） ====
        dist = get_distance_by_matrix(td['distance_matrix'], prev_phys_idx, target_phys_idx)  # [B]

        # 需求与时间窗
        dem_lh = gather_by_index(td["demand_linehaul"], target_phys_idx)
        dem_bh = gather_by_index(td["demand_backhaul"], target_phys_idx)
        service_t = gather_by_index(td["service_time"], target_phys_idx)
        tw = gather_by_index(td["time_windows"], target_phys_idx) # [B, 2]
        travel_t = dist / td["speed"].squeeze(-1)

        # 3. === 更新状态 ====
        # 3.1 更新车辆类型 (仅在 is_select_vehicle 时生效)
        new_v_type = (action - 1).clamp(min=0)
        td["current_vehicle_type"] = (1 - is_select_vehicle.long()) * td["current_vehicle_type"] + is_select_vehicle.long() * new_v_type        

        # 更新车辆计数
        delta_used = torch.zeros_like(td["vehicle_used_num"])
        delta_used[is_select_vehicle, new_v_type[is_select_vehicle]] = 1.0
        td["vehicle_used_num"] = td["vehicle_used_num"] + delta_used

        # 3.2 更新路径状态
        reset_mask = is_return_depot   

        td["current_length"] = (td["current_length"] + dist) * (1 - reset_mask.long())  # 回 depot 清零

        arrival_t = td["current_time"] + travel_t
        ready_t = torch.max(arrival_t, tw[..., 0]) + service_t
        td["current_time"] = (ready_t * is_select_customer) * (1 - reset_mask.long()) # 选车和回场都会清零或保持零

        td["current_used_capacity_lh"] = (td["current_used_capacity_lh"] + dem_lh) * (1 - reset_mask.long())
        td["current_used_capacity_bh"] = (td["current_used_capacity_bh"] + dem_bh) * (1 - reset_mask.long())

        # 3.3 更新访问状态
        td["visited"].scatter_(1, target_phys_idx.unsqueeze(-1), True)
        td["visited"][:, 0] = False # 保护位
        td["current_action"] = action

        # ==== 4. 处理完结状态并计算奖励 ====
        # 4.1 处理open route 情况
        is_open = td['open_route'].squeeze(-1)  # [B]
        is_return_depot_phys = (target_phys_idx == 0)  # [B], 物理上回 depot
        force_zero_dist_mask = is_open & is_return_depot_phys # [B], 强制回 depot 时, open route 情况下，距离为0
        dist = torch.where(force_zero_dist_mask, torch.zeros_like(dist), dist)  # [B]

        # 4.2 正常情况下，奖励为负的总成本（变动成本 + 固定成本）
        # 对于不可行完结的实例，额外施加未访问节点的惩罚
        vehicle_var_cost = gather_by_index(td["vehicle_variable_cost"], td["current_vehicle_type"])  # [B]
        vehicle_fixed_cost = gather_by_index(td["vehicle_fixed_cost"], td["current_vehicle_type"])  # [B]
        variable_cost = dist * vehicle_var_cost  # [B]
        fixed_cost = vehicle_fixed_cost * is_select_vehicle.float()  # [B], 仅在选择 vehicle type 时计算固定成本
        reward = -(variable_cost + fixed_cost).float().unsqueeze(-1)  # [B, 1]

        # 更新可行动作掩码
        td = self.get_action_mask(td)
        mask = td['legal_action_mask']  # [B, 1+V+N]

        # 可行完结条件
        all_visited = td['visited'][:, 1:].all(dim=-1)  # [B]
        at_depot_or_open = ((td['current_action'] == 0) | td['open_route'].squeeze(-1))  # [B]
        feasible_done = all_visited & at_depot_or_open  # [B]

        # 不可行完结条件
        has_legal_actions = mask.any(dim=-1)  # [B]
        infeasible_done = ~has_legal_actions & ~feasible_done  # [B]

        if feasible_done.any():
            # 对于已完成的实例，允许停留在 Depot (Node 0) 作为 dummy action
            mask[feasible_done, 0] = True
            mask[feasible_done, 1:] = False

        if infeasible_done.any():
            unvisited_mask = ~td['visited'][:, 1:]  # [B, N]

            # 最大成本参数
            max_fix = td['vehicle_fixed_cost'].max(dim=-1)[0].unsqueeze(-1)  # [B, 1]
            max_var = td['vehicle_variable_cost'].max(dim=-1)[0].unsqueeze(-1)  # [B, 1]

            # 获取距离
            dist_0_i = td['distance_matrix'][:, 0, 1:]  # [B, N]
            dist_i_0 = td['distance_matrix'][:, 1:, 0]  # [B, N]
            round_trip_dist = dist_0_i + dist_i_0  # [B, N]

            node_penalty = round_trip_dist * max_var  + max_fix  # [B, N]
            total_unvisited_penalty = (unvisited_mask.float() * node_penalty).sum(dim=-1, keepdim=True) # [B, 1]
            penalty_to_apply = torch.where(
                infeasible_done.unsqueeze(-1), 
                total_unvisited_penalty, 
                torch.zeros_like(total_unvisited_penalty)
            )   # [B, 1]

            reward = reward - penalty_to_apply # [B, 1] 

            # 将所有node都设为visited，允许停留在 Depot (Node 0) 作为 dummy action
            mask[infeasible_done, 0] = True
            mask[infeasible_done, 1:] = False
            td['visited'][infeasible_done, 1:] = True

        td['legal_action_mask'] = mask
        done = feasible_done | infeasible_done  # [B]

        return td, reward, done


    @staticmethod
    def get_global_features(td: TensorDict) -> TensorDict:
        # problem features, [B, 1, H_problem]
        problem_style_feature = torch.zeros(td.batch_size[0], 6, device=td.device)
        problem_style_feature[:, 0] = 1.0  # Capacity (C)

        problem_style_feature[:, 1] = td['open_route'].float().squeeze(-1)  # Open Route (O)

        has_tw = (td['time_windows'][:, 1:, 1] < float('inf')).any(dim=-1)
        problem_style_feature[:, 2] = has_tw.float()  # Time Windows (TW)

        has_limit = (td['distance_limit'] < float('inf')).squeeze(-1)
        problem_style_feature[:, 3] = has_limit.float()  # Distance Limit (L)

        has_backhaul = (td['demand_backhaul'] > 0).any(dim=-1)
        problem_style_feature[:, 4] = has_backhaul.float()  # Backhaul (B)

        has_heterogeneous_fleet = td['heterogeneous_fleet'].any(dim=-1).squeeze(-1)
        problem_style_feature[:, 5] = has_heterogeneous_fleet.float()  # Heterogeneous Fleet (HF)

        problem_features = problem_style_feature  # [B, H_problem]

        # node features, [B, N, H_node]
        depot_features = torch.cat(
                    [
                        td["open_route"].float()[..., None],           # [B, 1, 1], 是否开放
                        td["locs"][:, :1, :],                          # [B, 1, 2], depot位置
                        td["time_windows"][:, :1, 1:2],                # [B, 1, 1], depot时间窗结束时间
                    ],
                    -1,
                )

        node_features = torch.cat(
                    (
                        td["locs"][..., 1:, :],                 # [B, N, 2], 节点位置
                        td["demand_linehaul"][..., 1:, None],   # [B, N, 1], 线haul需求
                        td["demand_backhaul"][..., 1:, None],   # [B, N, 1], 回程需求
                        td["time_windows"][..., 1:, :],         # [B, N, 2], 节点时间窗
                        td["service_time"][..., 1:, None],      # [B, N, 1], 节点服务时间
                    ),
                    -1,
                )

        vehicle_features = torch.cat(
                    (
                        td['available_vehicles'][..., None],        # [B, V, 1], 可用车辆数量
                        td["vehicle_capacity"][..., None],          # [B, V, 1], 车辆容量
                        td["vehicle_fixed_cost"][..., None],        # [B, V, 1], 车辆固定成本
                        td["vehicle_variable_cost"][..., None],     # [B, V, 1], 车辆可变成本
                    ),
                    -1,
                )

        # ==== 获取状态特征 ====
        state_feature = {
            "problem_feature": torch.nan_to_num(problem_features, posinf=0),  # [B, H_problem]
            "depot_features": torch.nan_to_num(depot_features, posinf=0),     # [B, 1, H_depot]
            "node_features": torch.nan_to_num(node_features, posinf=0),     # [B, N, H_node]
            "vehicle_features": torch.nan_to_num(vehicle_features, posinf=0), # [B, V, H_vehicle]
        }
        
        return TensorDict({
            "state_feature": state_feature,
        },
        batch_size=td.batch_size,
        device=td.device,
        ) 

    @staticmethod
    def get_current_feature_and_mask(td: TensorDict) -> torch.Tensor:
        # current route features, [B, D_current]
        # legal actions mask, [B, 1+V+N]

        # current_info
        current_action = td["current_action"]  # [B]
        current_vehicle = td["current_vehicle_type"]  # [B]

        vehicle_available_num = td["available_vehicles"] - td["vehicle_used_num"]  # [B, V]
        remaining_vehicle_available_num = gather_by_index(vehicle_available_num, current_vehicle)  # [B]

        current_vehicle_capacity = gather_by_index(td['vehicle_capacity'], current_vehicle)  # [B]
        remaining_capacity_lh = current_vehicle_capacity - td["current_used_capacity_lh"]  # [B]
        remaining_capacity_bh = current_vehicle_capacity - td["current_used_capacity_bh"]  # [B]

        current_time = td["current_time"]  # [B]
        current_length = td["current_length"]  # [B]

        route_current_info = torch.cat(
            [
                current_action[..., None],                                       # [B, 1], 当前选择节点
                current_vehicle[..., None],                                      # [B, 1], 当前选择车辆类型
                remaining_vehicle_available_num[..., None],                      # [B, 1], 当前选择车辆剩余可用数量
                remaining_capacity_lh[..., None],                                # [B, 1], 线haul剩余容量
                remaining_capacity_bh[..., None],                                # [B, 1], 回haul剩余容量
                current_time[..., None],                                         # [B, 1], 当前路径时间
                current_length[..., None],                                       # [B, 1], 当前路径长度
            ],
            -1,
        )  # [B, D_current]

        current_features = torch.nan_to_num(route_current_info, posinf=0)

        # illegal actions mask
        illegal_actions_mask = ~td["legal_action_mask"]  # [B, 1+V+N]

        return current_features, illegal_actions_mask

    @staticmethod
    def select_start_nodes(td):
        B = td.batch_size[0]
        V = td["vehicle_capacity"].shape[-1]
        N = td["locs"].shape[-2] - 1
        
        # 随机生成起始节点索引和车辆类型索引，大小为 num_starts
        num_starts = N
        v_idx = torch.randint(1, V+1, (num_starts,), device=td.device)  # (num_starts,), 车辆类型索引， [1, V]
        c_idx = torch.arange(1+V, N + 1 + V, device=td.device)  # (num_starts,), 客户节点索引， [V+1, V+N]
        
        return num_starts, v_idx, c_idx
