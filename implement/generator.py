import random
from typing import Callable, Tuple, Union

import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from framework.gen import Generator, get_sampler
from framework.utils.io import save_tensordict_to_npz
from framework.utils.pylogger import get_pylogger
from implement.utils import get_distance

log = get_pylogger(__name__)


VARIANT_GENERATION_PRESETS = {
    # 包含所有特征的组合
    "no_hf_all": {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "HF": 0.0},
    "hf_all":    {"O": 0.5, "TW": 0.5, "L": 0.5, "B": 0.5, "HF": 1.0},
    
    # 基础变体
    "cvrp":    {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0, "HF": 0.0},
    "ovrp":    {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0, "HF": 0.0},
    "vrpb":    {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0, "HF": 0.0},
    "vrpl":    {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0, "HF": 0.0},
    "vrptw":   {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0, "HF": 0.0},
    "ovrptw":  {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0, "HF": 0.0},
    "ovrpb":   {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0, "HF": 0.0},
    "ovrpl":   {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0, "HF": 0.0},
    "vrpbl":   {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0, "HF": 0.0},
    "vrpbtw":  {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0, "HF": 0.0},
    "vrpltw":  {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0, "HF": 0.0},
    "ovrpbl":  {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0, "HF": 0.0},
    "ovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0, "HF": 0.0},
    "ovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0, "HF": 0.0},
    "vrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0, "HF": 0.0},
    "ovrpbltw":{"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0, "HF": 0.0},

    "hfcvrp":    {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 0.0, "HF": 1.0},
    "hfovrp":    {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 0.0, "HF": 1.0},
    "hfvrpb":    {"O": 0.0, "TW": 0.0, "L": 0.0, "B": 1.0, "HF": 1.0},
    "hfvrpl":    {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 0.0, "HF": 1.0},
    "hfvrptw":   {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 0.0, "HF": 1.0},
    "hfovrptw":  {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 0.0, "HF": 1.0},
    "hfovrpb":   {"O": 1.0, "TW": 0.0, "L": 0.0, "B": 1.0, "HF": 1.0},
    "hfovrpl":   {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 0.0, "HF": 1.0},
    "hfvrpbl":   {"O": 0.0, "TW": 0.0, "L": 1.0, "B": 1.0, "HF": 1.0},
    "hfvrpbtw":  {"O": 0.0, "TW": 1.0, "L": 0.0, "B": 1.0, "HF": 1.0},
    "hfvrpltw":  {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 0.0, "HF": 1.0},
    "hfovrpbl":  {"O": 1.0, "TW": 0.0, "L": 1.0, "B": 1.0, "HF": 1.0},
    "hfovrpbtw": {"O": 1.0, "TW": 1.0, "L": 0.0, "B": 1.0, "HF": 1.0},
    "hfovrpltw": {"O": 1.0, "TW": 1.0, "L": 1.0, "B": 0.0, "HF": 1.0},
    "hfvrpbltw": {"O": 0.0, "TW": 1.0, "L": 1.0, "B": 1.0, "HF": 1.0},
    "hfovrpbltw":{"O": 1.0, "TW": 1.0, "L": 1.0, "B": 1.0, "HF": 1.0},
}


class MTHFVRPGenerator(Generator):
    """Mutil-Type (HF) VRP Generator.
    Class to generate instances of the Heterogeneous Fleet Vehicle Routing Problem (VRP).
    
    支持主要变体特征的组合：
    - O (Open routes): 开放路径，车辆不需要返回depot
    - TW (Time Windows): 时间窗约束
    - L (Distance Limits): 距离限制约束  
    - B (Backhaul): 回程约束，支持linehaul和backhaul客户
    - HF (Heterogeneous Fleet): 异构车队模式，使用多种不同容量和成本的车辆类型
    
    可通过variant_preset参数选择预定义的问题变体，如：
    - "cvrp": 经典容量约束VRP (所有变体特征都为0)
    - "vrptw": 带时间窗VRP (TW=1.0, HF=0.0)
    - "hfcvrp": 异构车队CVRP (HF=1.0, 其他=0.0)
    - 等等...

    Args:
        num_loc: Number of locations to generate
        vehicle_num: Number of vehicles
        min_loc: Minimum location value
        max_loc: Maximum location value
        loc_distribution: Distribution to sample locations from
        random_seed: Random seed
        min_demand: Minimum demand value
        max_demand: Maximum demand value
        min_backhaul: Minimum backhaul value
        max_backhaul: Maximum backhaul value
        scale_demand: Scale demand values (by default, generate between 1 and 10)
        max_time: Maximum time window value (at depot)
        backhaul_ratio: Fraction of backhauls (e.g. 0.2 means 20% of nodes are backhaul)
        backhaul_class: which type of backhaul to use:
                1: classic backhaul (VRPB), linehauls must be served before backhauls in a route
                2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order
        sample_backhaul_class: whether to sample backhaul class
        max_distance_limit: Maximum distance limit
        speed: Speed of vehicle. Defaults to 1
        prob_open: Probability of open route feature
        prob_time_window: Probability of time window feature
        prob_limit: Probability of distance limit feature
        prob_backhaul: Probability of backhaul feature
        prob_hf: Probability of heterogeneous fleet feature
        variant_preset: Predefined variant preset name
        use_combinations: Whether to use feature combinations
        subsample: Whether to subsample problems
        **kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        num_loc: int = 20,
        vehicle_num: int = 10,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        min_demand: int = 1,
        max_demand: int = 10,
        min_backhaul: int = 1,
        max_backhaul: int = 10,
        scale_demand: bool = True,
        max_time: float = 4.6,
        backhaul_ratio: float = 0.2,
        backhaul_class: int = 1,
        sample_backhaul_class: bool = False,
        max_distance_limit: float = 2.8,  # 2sqrt(2) ~= 2.8
        speed: float = 1.0,
        prob_open: float = 0.5,
        prob_time_window: float = 0.5,
        prob_limit: float = 0.5,
        prob_backhaul: float = 0.5,
        prob_hf: float = 0.5,
        variant_preset=None,
        subsample=True,
        random_seed: int = None,
        **kwargs,
    ) -> None:
        # Location distribution
        self.num_loc = num_loc
        self.vehicle_num = vehicle_num
        self.min_loc = min_loc
        self.max_loc = max_loc
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )       # 一个抽样工具，针对不同的参数进行抽样生成。

        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_backhaul = min_backhaul
        self.max_backhaul = max_backhaul
        self.scale_demand = scale_demand
        self.backhaul_ratio = backhaul_ratio
        self.random_seed = random_seed

        assert backhaul_class in (
            1,
            2,
        ), "Backhaul class must be in [1, 2]. We don't use class 0 for efficiency since it is a subset"
        self.backhaul_class = backhaul_class
        self.sample_backhaul_class = sample_backhaul_class

        self.max_time = max_time
        self.max_distance_limit = max_distance_limit
        self.speed = speed

        if variant_preset is not None:
            log.info(f"Using variant generation preset {variant_preset}")
            variant_probs = VARIANT_GENERATION_PRESETS.get(variant_preset)
            assert (
                variant_probs is not None
            ), f"Variant generation preset {variant_preset} not found. \
                Available presets are {VARIANT_GENERATION_PRESETS.keys()} with probabilities {VARIANT_GENERATION_PRESETS.values()}"
        else:
            variant_probs = {
                "O": prob_open,
                "TW": prob_time_window,
                "L": prob_limit,
                "B": prob_backhaul,
                "HF": prob_hf,
            }
        # check probabilities
        for key, prob in variant_probs.items():
            assert 0 <= prob <= 1, f"Probability {key} must be between 0 and 1"
        self.variant_probs = variant_probs
        self.variant_preset = variant_preset
        self.subsample = subsample


    def _generate(self, batch_size) -> TensorDict:
        if self.random_seed is None:
            random.seed()  # 使用系统时间或其他随机源生成随机种子
            random_seed = random.randint(0, 10000)
        else:
            random_seed = self.random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Locations
        locs = self.generate_locations(batch_size=batch_size, num_loc=self.num_loc)

        # Distance Matrix
        distance_matrix = self.generate_distance_matrix(locs=locs)

        # Vehicle capacity (C, B) - applies to both linehaul and backhaul
        heterogeneous_fleet = torch.ones((*batch_size, 1), dtype=torch.bool)
        vehicle_info_hf = self.get_vehicle_capacity_and_costs(vehicle_num=self.vehicle_num, heterogeneous_fleet=True)
        available_vehicles = torch.tensor(vehicle_info_hf['available_vehicles'], dtype=torch.float32).repeat(*batch_size, 1)
        vehicle_capacity = torch.tensor(vehicle_info_hf['capacity'], dtype=torch.float32).repeat(*batch_size, 1)
        vehicle_fixed_cost = torch.tensor(vehicle_info_hf['fixed_cost'], dtype=torch.float32).repeat(*batch_size, 1)
        vehicle_variable_cost = torch.tensor(vehicle_info_hf['variable_cost'], dtype=torch.float32).repeat(*batch_size, 1)

        # linehaul demand / delivery (C) and backhaul / pickup demand (B)
        demand_linehaul, demand_backhaul = self.generate_demands(
            batch_size=batch_size, num_loc=self.num_loc
        )

        backhaul_class = self.generate_backhaul_class(
            shape=(*batch_size, 1), sample=self.sample_backhaul_class
        )

        # Open (O)
        open_route = self.generate_open_route(shape=(*batch_size, 1))

        # Time windows (TW)
        speed = self.generate_speed(shape=(*batch_size, 1))
        time_windows, service_time = self.generate_time_windows(
            locs=locs,
            speed=speed,
        )

        # Distance limit (L)
        distance_limit = self.generate_distance_limit(shape=(*batch_size, 1), locs=locs)

        # scaling
        if self.scale_demand:
            demand_backhaul /= vehicle_info_hf['base_capacity']
            demand_linehaul /= vehicle_info_hf['base_capacity']

        # Put all variables together
        td = TensorDict(
            {
                "locs": locs,
                "distance_matrix": distance_matrix,
                "demand_backhaul": demand_backhaul,  # (C)
                "demand_linehaul": demand_linehaul,  # (B)
                "backhaul_class": backhaul_class,  # (B)
                "distance_limit": distance_limit,  # (L)
                "time_windows": time_windows,  # (TW)
                "service_time": service_time,  # (TW)
                'available_vehicles': available_vehicles, # available vehicles number
                "vehicle_capacity": vehicle_capacity,  # (C)
                "vehicle_fixed_cost": vehicle_fixed_cost,  # Heterogeneous vehicles (HF)
                "variable_cost": vehicle_variable_cost,  # Heterogeneous vehicles (HF)
                "open_route": open_route,  # (O)
                "heterogeneous_fleet": heterogeneous_fleet,  # (HF)
                "speed": speed,  # common
            },
            batch_size=batch_size,
        )

        if self.subsample:
            # Subsample problems based on given instructions
            td = self.subsample_problems(td)
            return td
        else:
            # Not subsampling problems, i.e. return tensordict with all attributes
            return td

    def subsample_problems(self, td):
        """Create subproblems starting from seed probabilities depending on their variant.
        If random seed sampled in [0, 1] in batch is greater than prob, remove the constraint
        thus, if prob high, it is less likely to remove the constraint (i.e. prob=0.9, 90% chance to keep constraint)
        """
        batch_size = td.batch_size[0]

        variant_probs = torch.tensor(list(self.variant_probs.values()))

        if self.variant_preset == "hf_all" or self.variant_preset == "no_hf_all":
            num_variants = 5
            base_count = batch_size // num_variants
            remainder = batch_size % num_variants
            
            # Create indices: [0, 0, ..., 1, 1, ..., 2, 2, ..., 3, 3, ..., 4, 4, ...]
            indices_list = []
            for i in range(num_variants):
                count = base_count + (1 if i < remainder else 0)
                indices_list.extend([i] * count)
            
            indices = torch.tensor(indices_list, dtype=torch.long)
            indices = indices[torch.randperm(batch_size)]
            
            # Keep mask initialization
            keep_mask = torch.zeros(batch_size, 5, dtype=torch.bool)
            if self.variant_preset == "hf_all":
                keep_mask[:, 4] = True  # HF is always True (index 4)
            elif self.variant_preset == "no_hf_all":
                keep_mask[:, 4] = False # HF is always False (index 4)
            
            # 0: C=True, always keep
            keep_mask[indices == 1, 0] = True  # 1: O=True
            keep_mask[indices == 2, 2] = True  # 2: L=True
            keep_mask[indices == 3, 3] = True  # 3: B=True
            keep_mask[indices == 4, 1] = True  # 4: TW=True
        else:
            # in a batch, multiple variants combinations can be picked
            keep_mask = torch.rand(batch_size, 5) < variant_probs  # O, TW, L, B, HF

        td = self._default_open(td, ~keep_mask[:, 0])
        td = self._default_time_window(td, ~keep_mask[:, 1])
        td = self._default_distance_limit(td, ~keep_mask[:, 2])
        td = self._default_backhaul(td, ~keep_mask[:, 3])
        td = self._default_hf(td, ~keep_mask[:, 4])  # Handle HF feature

        return td

    @staticmethod
    def _default_open(td, remove):
        td["open_route"][remove] = False
        return td

    @staticmethod
    def _default_time_window(td, remove):
        default_tw = torch.zeros_like(td["time_windows"])
        default_tw[..., 1] = float("inf")
        td["time_windows"][remove] = default_tw[remove]
        td["service_time"][remove] = torch.zeros_like(td["service_time"][remove])
        return td

    @staticmethod
    def _default_distance_limit(td, remove):
        td["distance_limit"][remove] = float("inf")
        return td

    @staticmethod
    def _default_backhaul(td, remove):
        # by default, where there is a backhaul, linehaul is 0. therefore, we add backhaul to linehaul
        # and set backhaul to 0 where we want to remove backhaul
        td["demand_linehaul"][remove] = (
            td["demand_linehaul"][remove] + td["demand_backhaul"][remove]
        )
        td["demand_backhaul"][remove] = 0
        return td

    @staticmethod
    def _default_hf(td, remove):
        """Remove heterogeneous fleet feature by converting to homogeneous fleet.
        
        When HF is disabled, all vehicles should have:
        - Same capacity (use the middle type)
        - Zero fixed cost
        - Same variable cost (normalized to 1.0)
        """

        td["heterogeneous_fleet"][remove] = False
        # Get the homogeneous fleet configuration
        vehicle_info_homog = MTHFVRPGenerator.get_vehicle_capacity_and_costs(
            vehicle_num=td['available_vehicles'].shape[-1],
            heterogeneous_fleet=False
        )
        
        # Convert to tensors with the same device
        homog_capacity = torch.tensor(vehicle_info_homog['capacity'], dtype=td['vehicle_capacity'].dtype, device=td['vehicle_capacity'].device)
        homog_fixed = torch.tensor(vehicle_info_homog['fixed_cost'], dtype=td['vehicle_fixed_cost'].dtype, device=td['vehicle_fixed_cost'].device)
        homog_variable = torch.tensor(vehicle_info_homog['variable_cost'], dtype=td['variable_cost'].dtype, device=td['variable_cost'].device)

        # Apply homogeneous configuration where remove mask is True
        td["vehicle_capacity"][remove] = homog_capacity.expand_as(td["vehicle_capacity"][remove])
        td["vehicle_fixed_cost"][remove] = homog_fixed.expand_as(td["vehicle_fixed_cost"][remove])
        td["variable_cost"][remove] = homog_variable.expand_as(td["variable_cost"][remove])
        
        return td

    @staticmethod
    def get_vehicle_capacity_and_costs(vehicle_num: int = 10, heterogeneous_fleet: bool = True):

        # vehicle 
        if heterogeneous_fleet:
            # heterogeneous fleet with 5 types of vehicles, with different capacities and costs. The costs are normalized to be between 0 and 1, and the capacity is scaled accordingly.
            vehicle_types = [
                {'id': 0, 'capacity': 1.0, 'fixed_cost': 0.03, 'variable_cost': 0.28},
                {'id': 1, 'capacity': 1.4, 'fixed_cost': 0.056, 'variable_cost': 0.40},
                {'id': 2, 'capacity': 2.0,   'fixed_cost': 0.065, 'variable_cost': 0.40},
                {'id': 3, 'capacity': 4.0,   'fixed_cost': 0.065, 'variable_cost': 0.56},
                {'id': 4, 'capacity': 6.0,   'fixed_cost': 0.08, 'variable_cost': 0.80},
            ]
        else:
            # homogeneous fleet
            vehicle_types = [
                {'id': 0, 'capacity': 1.0, 'fixed_cost': 0.0, 'variable_cost': 1.0},
                {'id': 1, 'capacity': 1.0, 'fixed_cost': 0.0, 'variable_cost': 1.0},
                {'id': 2, 'capacity': 1.0, 'fixed_cost': 0.0, 'variable_cost': 1.0},
                {'id': 3, 'capacity': 1.0, 'fixed_cost': 0.0, 'variable_cost': 1.0},
                {'id': 4, 'capacity': 1.0, 'fixed_cost': 0.0, 'variable_cost': 1.0},
            ]

        # base capacity for scaling demand, set to the capacity of the middle type of vehicle (or the only type if homogeneous)
        base_capacity = 30

        vehicle_capacity = [vt['capacity'] for vt in vehicle_types]
        vehicle_fixed_cost = [vt['fixed_cost'] for vt in vehicle_types]
        vehicle_variable_cost = [vt['variable_cost'] for vt in vehicle_types]

        # assign available vehicles based on probabilities. 
        probabilities = [0.1, 0.15, 0.5, 0.1, 0.15]
        available_vehicles = np.random.multinomial(vehicle_num, probabilities).tolist()

        return {'available_vehicles': available_vehicles,
                'capacity': vehicle_capacity,
                'fixed_cost': vehicle_fixed_cost,
                'variable_cost': vehicle_variable_cost,
                'base_capacity': base_capacity}

    def generate_locations(self, batch_size, num_loc) -> torch.Tensor:
        """Generate seed locations.

        Returns:
            locs: [B, N+1, 2] where the first location is the depot.
        """
        locs = torch.FloatTensor(*batch_size, num_loc + 1, 2).uniform_(
            self.min_loc, self.max_loc
        )
        return locs

    def generate_demands(self, batch_size: int, num_loc: int) -> torch.Tensor:
        """Classical lineahul demand / delivery from depot (C) and backhaul demand / pickup to depot (B) generation.
        Initialize the demand for nodes except the depot, which are added during reset.
        Demand sampling Following Kool et al. (2019), demands as integers between 1 and 10.
        Generates a slightly different distribution than using torch.randint.

        Returns:
            linehaul_demand: [B, N]
            backhaul_demand: [B, N]
        """
        linehaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_demand - 1, self.max_demand - 1
        )
        linehaul_demand = (linehaul_demand.int() + 1).float()
        # Backhaul demand sampling
        backhaul_demand = torch.FloatTensor(*batch_size, num_loc).uniform_(
            self.min_backhaul - 1, self.max_backhaul - 1
        )
        backhaul_demand = (backhaul_demand.int() + 1).float()
        is_linehaul = torch.rand(*batch_size, num_loc) > self.backhaul_ratio
        backhaul_demand = (
            backhaul_demand * ~is_linehaul
        )  # keep only values where they are not linehauls
        linehaul_demand = linehaul_demand * is_linehaul
        return linehaul_demand, backhaul_demand

    def generate_time_windows(
        self,
        locs: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)
            speed: [B]

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """

        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.5
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (self.max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i / speed
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), self.max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]

    def generate_distance_limit(
        self, shape: Tuple[int, int], locs: torch.Tensor
    ) -> torch.Tensor:
        """Generates distance limits (L).
        The distance lower bound is dist_lower_bound = 2 * max(depot_to_location_distance),
        then the max can be max_lim = min(max_distance_limit, dist_lower_bound + EPS). Ensures feasible yet challenging
        constraints, with each instance having a unique, meaningful limit

        Returns:
            distance_limit: [B, 1]
        """
        max_dist = torch.max(torch.cdist(locs[:, 0:1], locs[:, 1:]).squeeze(-2), dim=1)[0]
        dist_lower_bound = 2 * max_dist + 1e-6
        max_distance_limit = torch.maximum(
            torch.full_like(dist_lower_bound, self.max_distance_limit),
            dist_lower_bound + 1e-6,
        )

        # We need to sample from the `distribution` module to get the same distribution with a tensor as input
        return torch.distributions.Uniform(dist_lower_bound, max_distance_limit).sample()[
            ..., None
        ]

    def generate_open_route(self, shape: Tuple[int, int]):
        """Generate open route flags (O). Here we could have a sampler but we simply return True here so all
        routes are open. Afterwards, we subsample the problems.
        """
        return torch.ones(shape, dtype=torch.bool)

    def generate_speed(self, shape: Tuple[int, int]):
        """We simply generate the speed as constant here"""
        # in this version, the speed is constant but this class may be overridden
        return torch.full(shape, self.speed, dtype=torch.float32)

    def generate_backhaul_class(self, shape: Tuple[int, int], sample: bool = False):
        """Generate backhaul class (B) for each node. If sample is True, we sample the backhaul class
        otherwise, we return the same class for all nodes.
        - Backhaul class 1: classic backhaul (VRPB), linehauls must be served before backhauls in a route (every customer is either, not both)
        - Backhaul class 2: mixed backhaul (VRPMPD or VRPMB), linehauls and backhauls can be served in any order (every customer is either, not both)
        """
        if sample:
            return torch.randint(1, 3, shape, dtype=torch.float32)
        else:
            return torch.full(shape, self.backhaul_class, dtype=torch.float32)

    def generate_distance_matrix(self, locs: torch.Tensor) -> torch.Tensor:
        """Generate distance matrix based on locs.
        
        Args:
            locs: [B, N+1, 2] where the first location is the depot.

        Returns:
            distance_matrix: [B, N+1, N+1]
        """
        locs_1 = locs.unsqueeze(2)  # [B, N+1, 1, 2]
        locs_2 = locs.unsqueeze(1)  # [B, 1, N+1, 2]
        distance_matrix = get_distance(locs_1, locs_2)    # [B, N+1, N+1]
            
        return distance_matrix

    @staticmethod
    def save_data(td: TensorDict, path, compress: bool = False):
        save_tensordict_to_npz(td, path)

    @staticmethod
    def print_presets():
        for key, value in VARIANT_GENERATION_PRESETS.items():
            print(f"{key}: {value}")

    @staticmethod
    def available_variants(*args, **kwargs):
        # remove 'all', 'single_feat' from the list
        return list(VARIANT_GENERATION_PRESETS.keys())[3:]
