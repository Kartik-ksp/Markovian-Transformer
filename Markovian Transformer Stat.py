import os
import math
import warnings
import copy
import traceback
from dataclasses import dataclass, field, fields
from typing import Optional, Tuple, List, Literal, Type, ContextManager, Dict, Any
import re # For pytest.warns match

# --- Determinism Control ---
FORCE_DETERMINISTIC_TESTING = os.getenv("FORCE_DETERMINISTIC_TESTING", "false").lower() == "true"
if FORCE_DETERMINISTIC_TESTING:
    print("INFO: FORCE_DETERMINISTIC_TESTING is True. Attempting to enable GLOBAL deterministic algorithms.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

if hasattr(torch,'_dynamo'):
    try:
        # torch._dynamo.config.verbose = True
        # import logging
        # torch._dynamo.config.log_level = logging.DEBUG
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_scalar_outputs = True
    except AttributeError:
        pass
    except Exception as e:
        print(f"Warning: Error setting torch._dynamo.config options: {e}")

if FORCE_DETERMINISTIC_TESTING:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    try:
        strict_determinism = torch.__version__ >= "1.8.0"
        torch.use_deterministic_algorithms(True, warn_only=not strict_determinism)
        print(f"INFO: PyTorch deterministic algorithms flag SET to True (strict: {strict_determinism}).")
    except RuntimeError as e:
        print(f"WARNING: Could not enforce all PyTorch deterministic algorithms: {e}")
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_available() and "CUBLAS" in str(e).upper():
             print("       CUBLAS_WORKSPACE_CONFIG was not set correctly BEFORE PyTorch import.")
        print("       Continuing... operations might be non-deterministic or error.")
    except AttributeError:
        print("WARNING: torch.use_deterministic_algorithms not available. Determinism not enforced.")
elif hasattr(torch, 'are_deterministic_algorithms_enabled') and torch.are_deterministic_algorithms_enabled():
    print("WARNING: PyTorch deterministic algorithms appear globally enabled.")
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(False)
            print("        Attempted to disable global determinism for this run.")
        except Exception as e:
            print(f"        Attempt to set use_deterministic_algorithms(False) failed: {e}")

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_fn

# --- Constants ---
EPSILON_CLAMP = 1e-9 
RFF_EPSILON = 1e-6 

# --- Pytest Import and Fallback ---
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    class DummyContextManagerImpl:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): return False

    class _pytest_warning_recorder(warnings.catch_warnings):
        def __init__(self, expected_warning_type: Type[Warning], match_pattern: Optional[Any]): # Match pattern can be str or compiled regex
            super().__init__(record=True)
            self.expected_warning_type = expected_warning_type
            self.match_pattern = match_pattern
            self.filtered_warnings_list: List[warnings.WarningMessage] = []

        def __enter__(self) -> List[warnings.WarningMessage]:
            self._all_recorded_warnings_from_super = super().__enter__() 
            return self.filtered_warnings_list 

        def __exit__(self, exc_type, exc_val, exc_tb):
            super().__exit__(exc_type, exc_val, exc_tb) 
            if self._all_recorded_warnings_from_super:
                for w_msg in self._all_recorded_warnings_from_super: 
                    if issubclass(w_msg.category, self.expected_warning_type):
                        message_str = str(w_msg.message)
                        if self.match_pattern is None:
                            self.filtered_warnings_list.append(w_msg)
                        elif isinstance(self.match_pattern, str): # if match is a string regex
                            if re.search(self.match_pattern, message_str): 
                                self.filtered_warnings_list.append(w_msg)
                        elif hasattr(self.match_pattern, 'search'): # if match is a compiled regex object
                            if self.match_pattern.search(message_str): # type: ignore
                                self.filtered_warnings_list.append(w_msg)
            return False 

    class pytest: # type: ignore
        @staticmethod
        def raises(expected_exception: Type[BaseException], match: Optional[str] = None) -> ContextManager[None]:
            return DummyContextManagerImpl()
        @staticmethod
        def warns(expected_warning: Type[Warning], match: Optional[Any] = None) -> ContextManager[List[warnings.WarningMessage]]: # match can be str or compiled regex
            return _pytest_warning_recorder(expected_warning, match) 
        @staticmethod
        def skip(reason: str = ""): print(f"INFO: Pytest skip called: {reason}"); return True
        @staticmethod 
        def fail(reason: str = ""): raise AssertionError(reason)


AttentionType = Literal["markovian", "qi_rff", "hybrid_split_heads", "hybrid_parallel_sum"]

@dataclass
class MarkovianTransformerConfig:
    vocab_size: int = 1000
    num_output_classes: int = 50
    embed_size: int = 256
    num_layers: int = 6
    heads: int = 8 
    
    max_order: int = 3
    layer_max_orders: Optional[List[int]] = None 
    head_specific_transitions: bool = True
    token_specific_order_gate: bool = True
    order_gate_variant: Literal["mlp", "deep_mlp"] = "mlp"
    use_bidirectional_markov: bool = True 
    constrain_transitions: bool = True
    transition_softplus_arg_offset: float = 1e-3
    
    ffn_hidden_dim_multiplier: float = 4.0
    dropout_rate: float = 0.1
    param_init_std: float = 0.02
    max_seq_len: int = 512
    
    use_absolute_pos_embeddings: bool = True
    max_rel_pos_bias_range: int = field(init=False)

    ffn_variant: Literal["swiglu", "gelu_ffn"] = "swiglu"
    tie_output_to_embedding_weights: bool = False
    use_gru_layer: bool = True
    gru_hidden_size: Optional[int] = None

    use_activation_checkpointing: bool = False
    activation_checkpointing_reentrant: bool = False 
    compile_model: bool = True
    torch_compile_mode: Optional[str] = "default"
    torch_compile_options: Optional[Dict[str, Any]] = None
    
    label_smoothing: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr_ratio: float = 0.1

    attention_type_per_layer: Optional[List[AttentionType]] = None 
    default_attention_type: AttentionType = "hybrid_parallel_sum" 
    
    hybrid_attention_markov_head_ratio: float = 0.5 
                                                    
    qi_attention_rff_dim_ratio: float = 1.0 
    qi_attention_redraw_rff_on_train: bool = False 

    use_adapters: bool = False
    adapter_bottleneck_dim_ratio: float = 0.25
    is_multimodal: bool = False
    image_feature_dim: int = 768 
    num_image_patches: int = 256 
    num_cross_attention_layers: int = 2
    multimodal_fusion_type: Literal["sequential_cross_attn", "interleaved_cross_attn"] = "sequential_cross_attn"

    sparsity_regularization_coeff: float = 0.0
    sparsity_target_modules: List[str] = field(default_factory=lambda: ["Linear", "Embedding"])


    def __post_init__(self):
        if not isinstance(self.param_init_std, float) or self.param_init_std <= 0: raise ValueError("param_init_std > 0")
        if not isinstance(self.transition_softplus_arg_offset, float): raise ValueError("transition_softplus_arg_offset float")
        if self.constrain_transitions and not (1e-7 < self.transition_softplus_arg_offset < 0.5):
             warnings.warn(f"transition_softplus_arg_offset ({self.transition_softplus_arg_offset}) outside typical range.", UserWarning)
        if not (0.0 <= self.dropout_rate < 1.0): raise ValueError("dropout_rate in [0,1)")
        for name, val, low, high_exclusive in [("learning_rate",self.learning_rate,0.0,float('inf')), ("beta1",self.beta1,0.0,1.0), ("beta2",self.beta2,0.0,1.0)]:
            if not (low <= val < high_exclusive if high_exclusive != float('inf') else low <= val): raise ValueError(f"{name} bounds error")
        if self.weight_decay < 0: raise ValueError("weight_decay >=0")
        if self.grad_clip < 0: raise ValueError("grad_clip >=0")
        if not (0.0 <= self.min_lr_ratio <= 1.0): raise ValueError("min_lr_ratio in [0,1]")
        if self.vocab_size <= 0: raise ValueError("vocab_size >0")
        if self.num_output_classes <= 0: raise ValueError("num_output_classes >0")
        if self.embed_size <= 0: raise ValueError("embed_size >0")
        if self.num_layers < 0: raise ValueError("num_layers >=0")

        if self.num_layers > 0:
            if self.heads <= 0: raise ValueError("heads >0 if num_layers >0")
            if self.embed_size % self.heads != 0: raise ValueError("embed_size % heads == 0 if num_layers >0")
        elif self.num_layers == 0 :
            default_heads = MarkovianTransformerConfig.__dataclass_fields__['heads'].default # type: ignore
            if self.heads != default_heads: self.heads = default_heads

        if self.layer_max_orders is not None:
            if not isinstance(self.layer_max_orders, list): raise ValueError("layer_max_orders list")
            if len(self.layer_max_orders) != self.num_layers: raise ValueError("len(layer_max_orders) == num_layers")
            if not all(isinstance(o, int) and o >= 0 for o in self.layer_max_orders): raise ValueError("layer_max_orders non-neg int")
            
            markovian_orders_in_use = []
            effective_att_types = self.get_effective_attention_types() 
            for i, att_type in enumerate(effective_att_types):
                 if att_type == "markovian" or "hybrid" in att_type: 
                    markovian_orders_in_use.append(self.layer_max_orders[i])
            self.max_rel_pos_bias_range = max(markovian_orders_in_use) if markovian_orders_in_use else 1
        else:
            if not (isinstance(self.max_order, int) and self.max_order >= 0): raise ValueError("max_order non-neg int")
            self.max_rel_pos_bias_range = self.max_order
            self.layer_max_orders = [self.max_order] * self.num_layers
        self.max_rel_pos_bias_range = max(1, self.max_rel_pos_bias_range) 

        if self.ffn_hidden_dim_multiplier <= 0: raise ValueError("ffn_hidden_dim_multiplier >0")
        if self.max_seq_len <= 0: raise ValueError("max_seq_len >0")

        eff_max_order_all_layers = max(self.layer_max_orders) if self.layer_max_orders and self.num_layers > 0 else 0
        if self.num_layers > 0 and eff_max_order_all_layers > 0 and self.max_seq_len < eff_max_order_all_layers:
            warnings.warn(f"max_seq_len ({self.max_seq_len}) < max_layer_order ({eff_max_order_all_layers}).", UserWarning)

        if self.warmup_iters < 0: raise ValueError("warmup_iters >=0")
        if self.lr_decay_iters < 0: raise ValueError("lr_decay_iters >=0")
        if self.warmup_iters > 0 and self.lr_decay_iters > 0 and self.warmup_iters >= self.lr_decay_iters :
            warnings.warn(f"Warmup ({self.warmup_iters}) >= LR decay ({self.lr_decay_iters}).", UserWarning)

        if self.tie_output_to_embedding_weights and self.num_output_classes != self.vocab_size:
            warnings.warn(f"Tying enabled but num_classes ({self.num_output_classes}) != vocab_size ({self.vocab_size}).", UserWarning)

        if self.use_gru_layer:
            if self.gru_hidden_size is None: self.gru_hidden_size = self.embed_size
            elif self.gru_hidden_size <= 0: raise ValueError("gru_hidden_size >0")

        if not self.use_absolute_pos_embeddings and self.max_rel_pos_bias_range <= 1:
             warnings.warn("No abs_pos_emb and rel_pos_range <=1. Limited pos info.", UserWarning)

        if self.torch_compile_options is None: self.torch_compile_options = {}
        if self.torch_compile_mode is None: self.torch_compile_mode = "default"

        if self.attention_type_per_layer is not None and len(self.attention_type_per_layer) != self.num_layers:
            raise ValueError("If attention_type_per_layer is specified, its length must match num_layers.")
        
        if not (0.0 <= self.hybrid_attention_markov_head_ratio <= 1.0): 
            raise ValueError(f"hybrid_attention_markov_head_ratio ({self.hybrid_attention_markov_head_ratio}) must be in [0,1].")
        
        if self.use_adapters and not (0 < self.adapter_bottleneck_dim_ratio <= 1): raise ValueError("adapter_bottleneck_dim_ratio must be in (0, 1]")
        
        uses_qi_rff = False
        for att_type in self.get_effective_attention_types():
            if att_type == "qi_rff" or "hybrid" in att_type:
                uses_qi_rff = True
                break
        if uses_qi_rff and not (0 < self.qi_attention_rff_dim_ratio):
            raise ValueError("qi_attention_rff_dim_ratio must be > 0 if QI attention is used")
        
        if self.is_multimodal:
            if self.num_cross_attention_layers < 0: raise ValueError("num_cross_attention_layers must be non-negative.")
            if self.image_feature_dim <= 0: raise ValueError("image_feature_dim must be positive.")
            if self.num_image_patches <= 0: raise ValueError("num_image_patches must be positive.")
        if self.sparsity_regularization_coeff < 0: raise ValueError("sparsity_regularization_coeff cannot be negative.")

    def get_effective_attention_types(self) -> List[AttentionType]:
        if self.attention_type_per_layer is not None:
            return self.attention_type_per_layer
        return [self.default_attention_type] * self.num_layers
    
    def get_layer_attention_config(self, layer_idx: int) -> Tuple[AttentionType, int, int]:
        effective_types = self.get_effective_attention_types()
        if layer_idx >= len(effective_types):
            if self.num_layers == 0: return self.default_attention_type, 0, 0 
            raise IndexError(f"layer_idx {layer_idx} out of bounds for effective_attention_types (len {len(effective_types)})")
        att_type = effective_types[layer_idx]

        if att_type == "markovian": return "markovian", self.heads, 0
        elif att_type == "qi_rff": return "qi_rff", 0, self.heads
        elif att_type == "hybrid_split_heads":
            num_markov_heads = math.floor(self.heads * self.hybrid_attention_markov_head_ratio)
            num_qi_heads = self.heads - num_markov_heads
            if self.heads > 0 and num_markov_heads == 0 and num_qi_heads == 0: 
                 num_markov_heads = self.heads 
                 warnings.warn(f"L{layer_idx} hybrid_split_heads ratio led to 0/0 heads. Defaulting all to Markovian.", UserWarning)
            return "hybrid_split_heads", num_markov_heads, num_qi_heads
        elif att_type == "hybrid_parallel_sum": return "hybrid_parallel_sum", self.heads, self.heads
        else: raise ValueError(f"Unknown attention type '{att_type}' for layer {layer_idx}")

# ... (GELUFeedForward, SwiGLUFeedForward, LearnedRelativePositionalEmbeddings remain the same)
class GELUFeedForward(nn.Module):
    def __init__(self, config: MarkovianTransformerConfig):
        super().__init__()
        h = int(config.ffn_hidden_dim_multiplier * config.embed_size)
        h = max(8, (h + 7) // 8 * 8) if h > 0 else 0 
        self.fc1 = nn.Linear(config.embed_size, h)
        self.fc2 = nn.Linear(h, config.embed_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.activation = nn.GELU(approximate='tanh') 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0: return x
        return self.dropout(self.fc2(self.activation(self.fc1(x))))

class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: MarkovianTransformerConfig):
        super().__init__()
        h_approx = int(config.ffn_hidden_dim_multiplier * config.embed_size)
        ffn_dim = int(2 * h_approx / 3)
        ffn_dim = max(8, (ffn_dim + 7) // 8 * 8) if ffn_dim > 0 else 0 
        self.w1 = nn.Linear(config.embed_size, ffn_dim, bias=False)
        self.w3 = nn.Linear(config.embed_size, ffn_dim, bias=False) 
        self.w2 = nn.Linear(ffn_dim, config.embed_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0: return x
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class LearnedRelativePositionalEmbeddings(nn.Module):
    def __init__(self, max_distance: int, heads: int):
        super().__init__()
        if max_distance <= 0: raise ValueError(f"max_distance > 0, got {max_distance}")
        if heads <= 0: raise ValueError(f"heads > 0, got {heads}") # Check heads > 0
        self.heads, self.max_distance = heads, max_distance
        self.relative_attention_bias = nn.Embedding(2 * self.max_distance + 1, self.heads)
        nn.init.zeros_(self.relative_attention_bias.weight) 
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if seq_len < 0: raise ValueError(f"seq_len >= 0, got {seq_len}")
        if seq_len == 0: return torch.empty((1, self.heads, 0, 0), dtype=dtype, device=device)
        q_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(1)
        k_pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        rel_pos_unclamped = k_pos - q_pos
        rel_pos_clamped = torch.clamp(rel_pos_unclamped, -self.max_distance, self.max_distance)
        rel_pos_indices = rel_pos_clamped + self.max_distance
        bias = self.relative_attention_bias(rel_pos_indices) 
        return bias.permute(2, 0, 1).unsqueeze(0).to(dtype)


class ImageEmbedder(nn.Module):
    def __init__(self, config: MarkovianTransformerConfig):
        super().__init__()
        self.config = config
        self.feature_projection = nn.Linear(config.image_feature_dim, config.embed_size)
        # Conditional positional embedding based on global config, similar to text
        if config.use_absolute_pos_embeddings:
            self.pos_embed: Optional[nn.Embedding] = nn.Embedding(config.num_image_patches, config.embed_size)
            self.register_buffer('image_positions', torch.arange(0, config.num_image_patches, dtype=torch.long), persistent=False)
        else:
            self.pos_embed = None
            self.register_buffer('image_positions', None, persistent=False)

        self.layer_norm = nn.LayerNorm(config.embed_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        B, N_img, _ = image_features.shape # D_img not used after projection
        if N_img == 0:
            return torch.empty(B, 0, self.config.embed_size, device=image_features.device, dtype=image_features.dtype) # Added dtype

        if N_img > self.config.num_image_patches:
            warnings.warn(f"Num img patches ({N_img}) > max ({self.config.num_image_patches}). Truncating.", UserWarning)
            image_features = image_features[:, :self.config.num_image_patches, :]
            N_img = self.config.num_image_patches

        x = self.feature_projection(image_features)

        if self.pos_embed is not None and self.image_positions is not None:
            pos_ids = self.image_positions[:N_img].unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(pos_ids)

        x = self.layer_norm(x)
        return self.dropout(x)



class QuantumInspiredAttentionRFF(nn.Module):
    ### MODIFIED ###
    def __init__(self, config: MarkovianTransformerConfig, layer_max_order: int = 0, num_attn_heads: Optional[int] = None,
                 external_qkv_proj: bool = False): # external_qkv_proj flag
        super().__init__()
        self.config = config
        self.embed_size = config.embed_size
        self.parent_total_heads = config.heads # Total heads in the parent block
        self.heads = num_attn_heads if num_attn_heads is not None else config.heads
        self.external_qkv_proj = external_qkv_proj
        
        if self.heads == 0:
            self.head_dim = 0; self.num_rff = 0
            if not self.external_qkv_proj: # Only define these if not externally projected
                self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Identity(),)*4
            self.dropout = nn.Identity()
            print(f"INFO: QuantumInspiredAttentionRFF initialized with 0 heads. Will act as Identity.")
            return

        # head_dim is always based on the parent's total division of embed_size
        if self.embed_size % self.parent_total_heads != 0:
             raise ValueError(f"QI RFF: embed_size {self.embed_size} not divisible by parent_total_heads {self.parent_total_heads}")
        self.head_dim = self.embed_size // self.parent_total_heads
        
        self.num_rff = max(1, int(self.head_dim * config.qi_attention_rff_dim_ratio))
        self.redraw_rff_on_train = config.qi_attention_redraw_rff_on_train

        if not self.external_qkv_proj:
            self.q_proj = nn.Linear(config.embed_size, self.heads * self.head_dim) # Projects to C_slice for its heads
            self.k_proj = nn.Linear(config.embed_size, self.heads * self.head_dim)
            self.v_proj = nn.Linear(config.embed_size, self.heads * self.head_dim)
            self.out_proj = nn.Linear(self.heads * self.head_dim, config.embed_size) # Projects C_slice back to C
        else: # If QKV are external, these are not needed by this module for projection
            self.q_proj, self.k_proj, self.v_proj, self.out_proj = (None,)*4


        self.dropout = nn.Dropout(config.dropout_rate)

        if self.redraw_rff_on_train:
            self.rff_projection_matrix_param = nn.Parameter(self._create_rff_projection_matrix_tensor(), requires_grad=False)
        else:
            self.register_buffer("rff_projection_matrix_buffer", self._create_rff_projection_matrix_tensor(), persistent=True)
        
        print(f"INFO: QuantumInspiredAttentionRFF initialized for {self.heads} heads (parent_total_heads: {self.parent_total_heads}, head_dim: {self.head_dim}), {self.num_rff} RFFs per head. Redraw: {self.redraw_rff_on_train}. External QKV: {self.external_qkv_proj}")


    def _create_rff_projection_matrix_tensor(self) -> torch.Tensor:
        if self.head_dim == 0 or self.num_rff == 0: return torch.empty(0)
        return torch.randn(self.num_rff, self.head_dim)

    def _get_current_rff_projection_matrix(self, device, dtype) -> torch.Tensor:
        if self.head_dim == 0 or self.num_rff == 0: return torch.empty(0, device=device, dtype=dtype)
        if self.redraw_rff_on_train and self.training:
            return self._create_rff_projection_matrix_tensor().to(device, dtype)
        elif hasattr(self, 'rff_projection_matrix_param'):
            return self.rff_projection_matrix_param.to(device, dtype)
        else:
            return self.rff_projection_matrix_buffer.to(device, dtype)

    def _rff_transform(self, x: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        if projection_matrix.numel() == 0 : 
            return torch.empty(*x.shape[:-1], 0, device=x.device, dtype=x.dtype) 
        projected_x = torch.einsum('bhnd,md->bhnm', x, projection_matrix) # x is (B, H_slice, N, Dh)
        scale = self.num_rff ** -0.5 
        return torch.cat([torch.cos(projected_x), torch.sin(projected_x)], dim=-1) * scale

    ### MODIFIED ###
    def forward(self, x_input_for_gate_or_proj: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor]=None, 
                is_causal: bool=False,
                q_external: Optional[torch.Tensor]=None, 
                k_external: Optional[torch.Tensor]=None, 
                v_external: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        B, N, C_full_or_slice = x_input_for_gate_or_proj.shape # C_full_or_slice depends on context
        
        if N == 0 or self.heads == 0: 
            if self.external_qkv_proj: # Return shape expected by calling block
                return torch.empty(B, N, self.heads * self.head_dim, device=x_input_for_gate_or_proj.device, dtype=x_input_for_gate_or_proj.dtype)
            return torch.zeros_like(x_input_for_gate_or_proj) if N > 0 and self.heads > 0 else x_input_for_gate_or_proj

        H, Dh = self.heads, self.head_dim # H is num_attn_heads for this module

        q, k, v = q_external, k_external, v_external
        if not self.external_qkv_proj:
            assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None
            # Input x is (B, N, C_full)
            # Projections map to (B, N, H * Dh) where H is num_attn_heads for this module
            q_proj_out = self.q_proj(x_input_for_gate_or_proj) # (B, N, H*Dh)
            k_proj_out = self.k_proj(x_input_for_gate_or_proj) # (B, N, H*Dh)
            v_proj_out = self.v_proj(x_input_for_gate_or_proj) # (B, N, H*Dh)

            q = q_proj_out.view(B, N, H, Dh).permute(0,2,1,3) # (B, H, N, Dh)
            k = k_proj_out.view(B, N, H, Dh).permute(0,2,1,3) # (B, H, N, Dh)
            v = v_proj_out.view(B, N, H, Dh).permute(0,2,1,3) # (B, H, N, Dh)
        else:
            # q,k,v are already (B, H_slice, N, Dh_parent)
            if q is None or k is None or v is None:
                raise ValueError("External Q, K, V must be provided when external_qkv_proj is True.")
            # Ensure H matches self.heads for this module
            if q.shape[1] != H or k.shape[1] != H or v.shape[1] != H :
                raise ValueError(f"External QKV head dim mismatch. Expected {H}, got Q:{q.shape[1]}, K:{k.shape[1]}, V:{v.shape[1]}")
        
        current_rff_matrix = self._get_current_rff_projection_matrix(q.device, q.dtype)
        if current_rff_matrix.numel() == 0:
            if self.external_qkv_proj:
                return torch.zeros(B, N, H * Dh, device=q.device, dtype=q.dtype)
            out_projected = self.out_proj(torch.zeros_like(x_input_for_gate_or_proj)) # type: ignore
            return self.dropout(out_projected)

        q_prime = self._rff_transform(q, current_rff_matrix) 
        k_prime = self._rff_transform(k, current_rff_matrix) 

        v_masked = v
        if key_padding_mask is not None:
            # Mask is (B, N_kv), expand to (B, 1, N_kv, 1) for k_prime and (B, 1, N_kv, 1) for v
            expanded_k_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(-1).to(k_prime.dtype)
            k_prime = k_prime * expanded_k_mask
            expanded_v_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(-1).to(v.dtype)
            v_masked = v * expanded_v_mask

        if is_causal:
            kv_outer_prod_cumsum_current = torch.zeros(B, H, self.num_rff * 2, Dh, device=q.device, dtype=q.dtype)
            k_prime_cumsum_diag_current = torch.zeros(B, H, self.num_rff * 2, device=q.device, dtype=q.dtype)
            out_attention_list = []
            for i in range(N):
                ki = k_prime[:, :, i, :]    
                vi = v_masked[:, :, i, :]   
                current_kv_term = torch.einsum('bhm,bhd->bhmd', ki, vi)
                kv_outer_prod_cumsum_current = kv_outer_prod_cumsum_current + current_kv_term 
                k_prime_cumsum_diag_current = k_prime_cumsum_diag_current + ki 
                qi = q_prime[:, :, i, :]    
                numerator = torch.einsum('bhm,bhmd->bhd', qi, kv_outer_prod_cumsum_current)
                denominator = torch.einsum('bhm,bhm->bh', qi, k_prime_cumsum_diag_current).unsqueeze(-1)
                out_attention_list.append(numerator / denominator.clamp(min=RFF_EPSILON))
            if N > 0: out_attention_weighted_v = torch.stack(out_attention_list, dim=2) 
            else: out_attention_weighted_v = torch.empty(B, H, 0, Dh, device=q.device, dtype=q.dtype)
        else: 
            kv_prod = torch.einsum('bhnm,bhnd->bhmd', k_prime, v_masked) # (B,H,N_k,M) @ (B,H,N_k,Dh) -> (B,H,M,Dh)
            out_attention_non_norm = torch.einsum('bhnm,bhmd->bhnd', q_prime, kv_prod) # (B,H,N_q,M) @ (B,H,M,Dh) -> (B,H,N_q,Dh)
            sum_k_prime = torch.sum(k_prime, dim=2, keepdim=True) # (B,H,1,M)
            denominator = torch.einsum('bhnm,bhzm->bhn', q_prime, sum_k_prime) # (B,H,N_q,M) @ (B,H,1,M) -> (B,H,N_q)
            out_attention_weighted_v = out_attention_non_norm / denominator.unsqueeze(-1).clamp(min=RFF_EPSILON)

        # out_attention_weighted_v is (B, H, N, Dh)
        # Reshape to (B, N, H * Dh) = (B, N, C_slice)
        out_reshaped = out_attention_weighted_v.permute(0,2,1,3).reshape(B, N, H * Dh)
        
        if not self.external_qkv_proj:
            assert self.out_proj is not None
            out_projected = self.out_proj(out_reshaped) # (B, N, C_full)
            return self.dropout(out_projected)
        else:
            # Return C_slice, dropout is handled by parent block's residual dropout
            return out_reshaped # (B, N, C_slice)


class LearnableMarkovianAttention(nn.Module):
    ### MODIFIED ###
    def __init__(self, config: MarkovianTransformerConfig, layer_max_order: int, num_attn_heads: Optional[int] = None,
                 external_qkv_proj: bool = False): # external_qkv_proj flag
        super().__init__()
        self.config = config
        self.trans_config_use_bidir = config.use_bidirectional_markov
        self.parent_total_heads = config.heads # Total heads in the parent block
        self.heads = num_attn_heads if num_attn_heads is not None else config.heads # Actual heads this module operates on
        self.external_qkv_proj = external_qkv_proj
        
        if self.heads == 0 :
            self.head_dim = 0
            self.register_parameter('transition_forward', None); self.register_parameter('transition_backward', None)
            self.order_gate = nn.Identity()
            if not self.external_qkv_proj:
                 self.qkv = nn.Identity(); self.proj = nn.Identity()
            self.resid_dropout = nn.Identity(); self.relative_bias_layer = None
            print(f"INFO: LearnableMarkovianAttention initialized with 0 heads. Will act as Identity.")
            return

        # head_dim is always based on the parent's total division of embed_size
        if config.embed_size % self.parent_total_heads != 0:
             raise ValueError(f"MarkovAttn: embed_size ({config.embed_size}) must be divisible by parent_total_heads ({self.parent_total_heads})")
        self.head_dim = config.embed_size // self.parent_total_heads

        if layer_max_order < 0: raise ValueError(f"layer_max_order >= 0, got {layer_max_order}")
        self.layer_max_order = layer_max_order

        trans_shape = (self.heads, layer_max_order) if config.head_specific_transitions else (layer_max_order,)
        if layer_max_order > 0:
            self.transition_forward = nn.Parameter(torch.empty(*trans_shape)); nn.init.normal_(self.transition_forward, 0, config.param_init_std)
            if config.use_bidirectional_markov: self.transition_backward = nn.Parameter(torch.empty(*trans_shape)); nn.init.normal_(self.transition_backward, 0, config.param_init_std)
            else: self.register_parameter('transition_backward', None)
        else: self.register_parameter('transition_forward', None); self.register_parameter('transition_backward', None)

        # Order gate input dim depends on token_specific_order_gate
        # If token_specific, uses head_dim (of a single head).
        # If not token_specific, uses full embed_size of x_input_for_gate_or_proj.
        gate_input_dim_eff = self.head_dim if config.token_specific_order_gate else config.embed_size

        g_mid_dim, g_mid_dim_1, g_mid_dim_2 = 0,0,0 
        if config.order_gate_variant == "mlp": g_mid_dim = max(16, int((2.0 if config.token_specific_order_gate else 0.5) * gate_input_dim_eff)) if gate_input_dim_eff > 0 else 0
        elif config.order_gate_variant == "deep_mlp":
            g_mid_dim_1 = max(16, int((2.0 if config.token_specific_order_gate else 0.5) * gate_input_dim_eff)) if gate_input_dim_eff > 0 else 0
            g_mid_dim_2 = max(16, g_mid_dim_1 // 2) if g_mid_dim_1 > 0 else 0
        else: raise ValueError(f"Unknown order_gate_variant: {config.order_gate_variant}")
        self.n_dirs_gate_out = (2 if config.use_bidirectional_markov and layer_max_order > 0 else (1 if layer_max_order > 0 else 0))
        g_out_dim = self.n_dirs_gate_out * layer_max_order
        if layer_max_order == 0 and g_out_dim != 0: raise ValueError(f"g_out_dim ({g_out_dim}) must be 0 if LMO=0.")
        
        if layer_max_order > 0 and gate_input_dim_eff > 0 and g_out_dim > 0 and self.head_dim > 0: 
            if config.order_gate_variant == "mlp" and g_mid_dim > 0: self.order_gate = nn.Sequential(nn.Linear(gate_input_dim_eff,g_mid_dim), nn.GELU(), nn.Linear(g_mid_dim,g_out_dim))
            elif config.order_gate_variant == "deep_mlp" and g_mid_dim_1 > 0 and g_mid_dim_2 > 0: self.order_gate = nn.Sequential(nn.Linear(gate_input_dim_eff, g_mid_dim_1), nn.GELU(), nn.Linear(g_mid_dim_1, g_mid_dim_2), nn.GELU(), nn.Linear(g_mid_dim_2, g_out_dim))
            else: self.order_gate = nn.Linear(gate_input_dim_eff, g_out_dim) if gate_input_dim_eff > 0 and g_out_dim > 0 else nn.Identity()
        else: self.order_gate = nn.Identity()

        if not self.external_qkv_proj:
            # Projects to C_slice for its heads (self.heads * self.head_dim)
            # The QKV output here will be 3 * (self.heads * self.head_dim)
            self.qkv = nn.Linear(config.embed_size, 3 * self.heads * self.head_dim) 
            self.proj = nn.Linear(self.heads * self.head_dim, config.embed_size) # C_slice to C_full
        else:
            self.qkv, self.proj = None, None # Not used if QKV are external

        self.attn_dropout_p = config.dropout_rate
        self.resid_dropout = nn.Dropout(config.dropout_rate) # Note: if external_qkv_proj, this dropout might be redundant if parent block applies it
        
        # Relative bias layer needs to know how many heads it's generating biases for (self.heads)
        self.relative_bias_layer = LearnedRelativePositionalEmbeddings(max(1, config.max_rel_pos_bias_range), self.heads) if self.heads > 0 else None
        print(f"INFO: LearnableMarkovianAttention initialized for {self.heads} heads (parent_total_heads: {self.parent_total_heads}, head_dim: {self.head_dim}). External QKV: {self.external_qkv_proj}")


    ### MODIFIED ###
    def forward(self, x_input_for_gate_or_proj: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor]=None, 
                is_causal: bool=False,
                q_external: Optional[torch.Tensor]=None, 
                k_external: Optional[torch.Tensor]=None, 
                v_external: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        B, N, C_full_or_slice = x_input_for_gate_or_proj.shape
        
        if N == 0 or self.heads == 0:
            if self.external_qkv_proj: # Return shape expected by calling block
                return torch.empty(B, N, self.heads * self.head_dim, device=x_input_for_gate_or_proj.device, dtype=x_input_for_gate_or_proj.dtype)
            return torch.zeros_like(x_input_for_gate_or_proj) if N > 0 and self.heads > 0 else x_input_for_gate_or_proj

        H, Dh = self.heads, self.head_dim # H is num_attn_heads for this module

        q, k, v = q_external, k_external, v_external
        if not self.external_qkv_proj:
            assert self.qkv is not None
            # x_input_for_gate_or_proj is (B, N, C_full)
            # self.qkv projects C_full to 3 * (H * Dh)
            qkv_out = self.qkv(x_input_for_gate_or_proj) 
            # q,k,v will be (B, H, N, Dh)
            q, k, v = qkv_out.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4).unbind(0)
        else:
            # q,k,v are already (B, H_slice, N, Dh_parent)
            if q is None or k is None or v is None:
                raise ValueError("External Q, K, V must be provided when external_qkv_proj is True.")
            if q.shape[1] != H or k.shape[1] != H or v.shape[1] != H :
                 raise ValueError(f"External QKV head dim mismatch. Expected {H}, got Q:{q.shape[1]}, K:{k.shape[1]}, V:{v.shape[1]}")

        dyn_markov_bias_val = torch.zeros((B, H if self.config.head_specific_transitions else 1, N, N), device=q.device, dtype=q.dtype)
        if self.layer_max_order > 0:
            if self.config.token_specific_order_gate:
                # q is (B, H, N, Dh). We need (B, N, H, Dh) for permute, then reshape for gate
                q_for_gate = q.permute(0,2,1,3) # (B, N, H, Dh)
                q_gate_reshaped = q_for_gate.reshape(-1, Dh) # (B*N*H, Dh) -> this is wrong if H != self.parent_total_heads
                                                            # The gate input dim was set based on self.head_dim.
                                                            # If token_specific, it expects one head's q.
                                                            # This needs to be thought through. For now, assume q passed is correct for gate.
                                                            # The gate input is (B*N*H_slice, Dh_parent)
                # To make it work, we should iterate over heads if token_specific_order_gate
                # Or, the gate is shared across these H_slice heads.
                # Let's assume for now the gate is applied per head based on its q_external.
                # q.permute(0,2,1,3) gives (B, N, H, Dh)
                # .reshape(-1, Dh) gives (B*N*H, Dh)
                order_logits_flat = self.order_gate(q.permute(0,2,1,3).reshape(-1, Dh))
                order_logits = order_logits_flat.view(B,N,H,self.n_dirs_gate_out*self.layer_max_order).permute(0,2,1,3)
                order_weights = F.softmax(order_logits.view(B,H,N,self.n_dirs_gate_out,self.layer_max_order), dim=-1)
            else: 
                # Use the original x that was passed to the block for sequence-level gate
                gate_in_x = x_input_for_gate_or_proj # This should be the original full x (B,N,C_full)
                if key_padding_mask is not None:
                    non_pad_mask = (~key_padding_mask).unsqueeze(-1).to(gate_in_x.dtype) 
                    masked_x_sum = (gate_in_x * non_pad_mask).sum(dim=1) 
                    num_non_padded = non_pad_mask.sum(dim=1).clamp(min=EPSILON_CLAMP) 
                    gate_in = masked_x_sum / num_non_padded 
                else: gate_in = gate_in_x.mean(dim=1) 
                order_logits_seq = self.order_gate(gate_in) 
                order_weights = F.softmax(order_logits_seq.view(B,self.n_dirs_gate_out,self.layer_max_order),dim=-1).unsqueeze(1).unsqueeze(1)
            dyn_markov_bias_val = self.create_markovian_bias(N, order_weights, q.device)
        
        rel_bias = self.relative_bias_layer(N, q.device, q.dtype) if self.relative_bias_layer else 0.0
        
        scale_factor = Dh ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor # (B,H,N,N)
        
        # dyn_markov_bias_val can be (B, H_or_1, N, N). It needs to match scores' H.
        if dyn_markov_bias_val.shape[1] == 1 and H > 1 and self.config.head_specific_transitions:
            # This case occurs if head_specific_transitions=True but the order_gate was not token_specific
            # OR if the bias was meant to be shared. Given head_specific_transitions=True, it should apply per head.
            # The current create_markovian_bias produces H if head_specific_transitions=True based on self.heads
            # So, this path might not be hit if configured consistently.
             dyn_markov_bias_val = dyn_markov_bias_val.expand(-1, H, -1, -1)
        elif dyn_markov_bias_val.shape[1] != H and dyn_markov_bias_val.shape[1] != 1:
            raise ValueError(f"dyn_markov_bias_val head dim {dyn_markov_bias_val.shape[1]} incompatible with scores head dim {H}")

        scores = scores + dyn_markov_bias_val
        if self.relative_bias_layer: scores = scores + rel_bias 

        if key_padding_mask is not None: scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), torch.finfo(scores.dtype).min)
        if is_causal:
            causal_mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        
        probs = F.softmax(scores, dim=-1)
        if self.training and self.attn_dropout_p > 0: probs = F.dropout(probs, p=self.attn_dropout_p)
        
        out_attention_weighted_v = torch.matmul(probs, v) # (B, H, N, Dh)
        out_reshaped = out_attention_weighted_v.transpose(1, 2).reshape(B, N, H * Dh) # (B, N, C_slice)
        
        if not self.external_qkv_proj:
            assert self.proj is not None
            out_projected = self.proj(out_reshaped) # (B, N, C_full)
            # If not external, this module is standalone, so apply its residual dropout.
            return self.resid_dropout(out_projected) if self.training and self.config.dropout_rate > 0 else out_projected
        else:
            # Return C_slice. Parent block will handle projection and dropout.
            return out_reshaped


    def create_markovian_bias(self, seq_len: int, order_weights: torch.Tensor, device: torch.device) -> torch.Tensor:
        B = order_weights.shape[0]
        # n_bias_H should be self.heads (number of heads this module is responsible for)
        n_bias_H = self.heads if self.config.head_specific_transitions else 1

        if self.layer_max_order == 0 or self.heads == 0: return torch.zeros(B, n_bias_H, seq_len, seq_len, device=device, dtype=order_weights.dtype)
        if seq_len == 0: return torch.empty((B, n_bias_H, 0, 0), device=device, dtype=order_weights.dtype)
        if not isinstance(self.transition_forward,nn.Parameter):
            warnings.warn(f"Layer {self}: transition_forward not nn.Parameter (LMO={self.layer_max_order}). Zero bias.",UserWarning)
            return torch.zeros(B,n_bias_H,seq_len,seq_len,device=device,dtype=order_weights.dtype)
        
        tf_raw = self.transition_forward # Shape (H_slice, LMO) or (LMO,)
        tb_raw=None
        if self.trans_config_use_bidir and hasattr(self,'transition_backward') and isinstance(self.transition_backward,nn.Parameter): 
            tb_raw=self.transition_backward # Shape (H_slice, LMO) or (LMO,)

        # If not head specific transitions, but this module has H_slice heads,
        # the transitions (LMO,) need to be expanded to (1, LMO) for broadcasting with order_weights (B,H_slice,N,Dirs,LMO)
        if not self.config.head_specific_transitions: # transitions are (LMO,)
            tf_raw_eff = tf_raw.unsqueeze(0) # (1, LMO)
            tb_raw_eff = tb_raw.unsqueeze(0) if tb_raw is not None else None # (1, LMO)
        else: # transitions are (H_slice, LMO)
            tf_raw_eff = tf_raw
            tb_raw_eff = tb_raw
            # Ensure order_weights match H_slice (B,H_slice,N,Dirs,LMO)
            if order_weights.shape[1] != n_bias_H and n_bias_H > 0: # order_weights might be (B,1,N,Dirs,LMO) if not token specific gate
                if order_weights.shape[1] == 1 : # Expand order_weights if gate was not token specific
                     order_weights = order_weights.expand(-1, n_bias_H, -1, -1, -1)
                else:
                    raise ValueError(f"Order weights head dim {order_weights.shape[1]} mismatch with n_bias_H {n_bias_H}")


        tf = F.softplus(tf_raw_eff + self.config.transition_softplus_arg_offset) if self.config.constrain_transitions else tf_raw_eff
        n_dirs_trans=1; transitions=tf.unsqueeze(1) # (H_slice_or_1, 1, LMO)
        
        if self.trans_config_use_bidir:
            if tb_raw_eff is not None:
                tb=F.softplus(tb_raw_eff + self.config.transition_softplus_arg_offset) if self.config.constrain_transitions else tb_raw_eff
                transitions=torch.stack([tf,tb],dim=1); n_dirs_trans=2 # (H_slice_or_1, 2, LMO)
            else: warnings.warn(f"Layer {self}: Bidir Markov, but 'transition_backward' missing. Uni-dir bias.",UserWarning, stacklevel=2) 
        
        # order_weights is (B, H_slice_or_1_from_gate, N, Dirs_gate, LMO)
        # transitions is (H_slice_or_1_from_param, Dirs_trans, LMO)
        # We need to align them.
        
        ow_eff = order_weights # Current shape (B, H_gate, N, Dirs_gate, LMO)
                               # H_gate is H_slice if token_specific_gate, or 1 if global_gate (expanded to H_slice if head_specific_trans)
        
        n_dirs_w = ow_eff.shape[-2] # Dirs_gate
        if n_dirs_w != n_dirs_trans: # Dirs_gate vs Dirs_trans
            warnings.warn(f"Layer {self}: Dir mismatch! OrderW {n_dirs_w} dirs, Trans {n_dirs_trans} dirs. Adjusting.",UserWarning, stacklevel=2)
            if n_dirs_trans==1 and n_dirs_w==2: ow_eff=ow_eff[...,0:1,:]
            elif n_dirs_trans==2 and n_dirs_w==1:
                warnings.warn(f"Layer {self}: Duplicating uni-dir order_weights for bi-dir transitions.",UserWarning, stacklevel=2)
                ow_eff=ow_eff.repeat_interleave(2,dim=-2)
            else: raise RuntimeError(f"Layer {self}: Unhandled dir mismatch. OW_dirs: {n_dirs_w}, T_dirs: {n_dirs_trans}.")
        
        curr_n_dirs_from_ow = ow_eff.shape[-2]
        dyn_bias = torch.zeros(B,n_bias_H,seq_len,seq_len,device=device,dtype=ow_eff.dtype)
        
        # transitions: (H_param, Dirs, LMO)
        # ow_eff:      (B, H_gate, N, Dirs, LMO)
        # We want bias_vals: (B, n_bias_H, N, Dirs, LMO)
        # n_bias_H is self.heads (for this module) if head_specific_transitions, else 1.

        # Unsqueeze transitions to (1, H_param, 1, Dirs, LMO) for broadcasting with ow_eff
        trans_exp = transitions.unsqueeze(0).unsqueeze(2) 

        if self.config.head_specific_transitions:
            # ow_eff should be (B, n_bias_H, N, Dirs, LMO)
            # trans_exp should be (1, n_bias_H, 1, Dirs, LMO)
            if ow_eff.shape[1] != n_bias_H : # This means ow_eff came from a non-token-specific gate (shape B,1,N,D,LMO)
                if ow_eff.shape[1] == 1:
                    ow_eff_expanded = ow_eff.expand(-1, n_bias_H, -1, -1, -1)
                else:
                    raise ValueError("Head dimension mismatch for head_specific_transitions in bias calculation.")
            else:
                ow_eff_expanded = ow_eff
            
            if trans_exp.shape[1] != n_bias_H : # This means transitions were not head specific from param (shape 1,1,1,D,LMO)
                if trans_exp.shape[1] == 1:
                    trans_exp_expanded = trans_exp.expand(-1, n_bias_H, -1, -1, -1)
                else:
                     raise ValueError("Transition head dimension mismatch for head_specific_transitions in bias calculation.")
            else:
                trans_exp_expanded = trans_exp
            bias_vals = trans_exp_expanded * ow_eff_expanded # (B, n_bias_H, N, Dirs, LMO)
        else: # not head_specific_transitions
            # ow_eff is (B, 1 or H_slice_if_token_specific, N, Dirs, LMO)
            # trans_exp is (1, 1, 1, Dirs, LMO)
            # We need to average ow_eff over H if it's token_specific
            ow_for_non_head_specific = ow_eff.mean(dim=1, keepdim=True) if self.config.token_specific_order_gate and ow_eff.shape[1] > 1 else ow_eff
            # ow_for_non_head_specific is now (B, 1, N, Dirs, LMO)
            bias_vals = trans_exp * ow_for_non_head_specific # (B, 1, N, Dirs, LMO)
            if n_bias_H != 1: raise ValueError("n_bias_H should be 1 if not head_specific_transitions")
            

        for k_idx in range(self.layer_max_order):
            k_offset = k_idx+1; vals_f = bias_vals[...,0,k_idx] # (B, n_bias_H, N) or (B, n_bias_H) if not token_specific
            q_indices_f = torch.arange(seq_len-k_offset,device=device); k_indices_f = q_indices_f + k_offset
            if torch.compiler.is_compiling() or q_indices_f.numel() > 0:
                update_f = vals_f[...,q_indices_f] if vals_f.ndim == 3 and vals_f.shape[-1] == seq_len else vals_f # Token specific vs sequence specific
                if update_f.ndim < dyn_bias[:,:,q_indices_f,k_indices_f].ndim : update_f = update_f.unsqueeze(-1)
                dyn_bias[:,:,q_indices_f,k_indices_f] = dyn_bias[:,:,q_indices_f,k_indices_f] + update_f 
            
            if curr_n_dirs_from_ow == 2:
                vals_b = bias_vals[...,1,k_idx]
                q_indices_b = torch.arange(k_offset,seq_len,device=device); k_indices_b = q_indices_b - k_offset
                if torch.compiler.is_compiling() or q_indices_b.numel() > 0:
                    update_b = vals_b[...,q_indices_b] if vals_b.ndim == 3 and vals_b.shape[-1] == seq_len else vals_b
                    if update_b.ndim < dyn_bias[:,:,q_indices_b,k_indices_b].ndim: update_b = update_b.unsqueeze(-1)
                    dyn_bias[:,:,q_indices_b,k_indices_b] = dyn_bias[:,:,q_indices_b,k_indices_b] + update_b 
        return dyn_bias # (B, n_bias_H, N, N)

class Adapter(nn.Module):
    def __init__(self, embed_size: int, bottleneck_ratio: float, dropout_rate: float):
        super().__init__()
        bottleneck_size = max(1, int(embed_size * bottleneck_ratio))
        self.down_proj = nn.Linear(embed_size, bottleneck_size)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_size, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.up_proj.weight); nn.init.zeros_(self.up_proj.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.dropout(self.up_proj(self.activation(self.down_proj(x))))

class MarkovianTransformerBlock(nn.Module):
    def __init__(self, config: MarkovianTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.full_embed_size = config.embed_size
        self.total_heads = config.heads
        if self.full_embed_size % self.total_heads != 0:
            raise ValueError(f"Block Layer {layer_idx}: embed_size {self.full_embed_size} not divisible by total_heads {self.total_heads}")
        self.head_dim = self.full_embed_size // self.total_heads
        
        self.att_type, self.num_markov_heads, self.num_qi_heads = config.get_layer_attention_config(layer_idx)
        lmo_for_this_layer = config.layer_max_orders[layer_idx] if config.layer_max_orders and layer_idx < len(config.layer_max_orders) else config.max_order

        self.attn_markov: Optional[LearnableMarkovianAttention] = None
        self.attn_qi: Optional[QuantumInspiredAttentionRFF] = None
        
        ### MODIFIED ###
        self.qkv_block_internal: Optional[nn.Linear] = None # Used by hybrid_split_heads and potentially others if refactored
        self.proj_block_internal: Optional[nn.Linear] = None

        if self.att_type == "hybrid_split_heads":
            if not (self.num_markov_heads + self.num_qi_heads == config.heads):
                raise ValueError(f"Layer {layer_idx}: Sum of markov heads ({self.num_markov_heads}) and qi heads ({self.num_qi_heads}) "
                                 f"must equal total config.heads ({config.heads}) for hybrid_split_heads.")
            
            # These are always needed for hybrid_split_heads for the combined operation
            self.qkv_block_internal = nn.Linear(config.embed_size, 3 * config.embed_size)
            self.proj_block_internal = nn.Linear(config.embed_size, config.embed_size)
            
            if self.num_markov_heads > 0:
                self.attn_markov = LearnableMarkovianAttention(config, lmo_for_this_layer, 
                                                               num_attn_heads=self.num_markov_heads, 
                                                               external_qkv_proj=True)
            if self.num_qi_heads > 0:
                self.attn_qi = QuantumInspiredAttentionRFF(config, lmo_for_this_layer, 
                                                           num_attn_heads=self.num_qi_heads,
                                                           external_qkv_proj=True)
        elif self.att_type == "markovian":
            self.attn_markov = LearnableMarkovianAttention(config, lmo_for_this_layer, num_attn_heads=config.heads, external_qkv_proj=False)
        elif self.att_type == "qi_rff":
            self.attn_qi = QuantumInspiredAttentionRFF(config, lmo_for_this_layer, num_attn_heads=config.heads, external_qkv_proj=False)
        elif self.att_type == "hybrid_parallel_sum":
            # Each sub-module handles its own QKV and projection from full C, then outputs are summed.
            if self.num_markov_heads > 0 : # For parallel_sum, interpret num_markov_heads as total heads for this path
                 self.attn_markov = LearnableMarkovianAttention(config, lmo_for_this_layer, num_attn_heads=self.num_markov_heads, external_qkv_proj=False)
            else: # If ratio is 0 for markov, effectively it's not used
                 self.attn_markov = LearnableMarkovianAttention(config, lmo_for_this_layer, num_attn_heads=0, external_qkv_proj=False)


            if self.num_qi_heads > 0 : # For parallel_sum, interpret num_qi_heads as total heads for this path
                self.attn_qi = QuantumInspiredAttentionRFF(config, lmo_for_this_layer, num_attn_heads=self.num_qi_heads, external_qkv_proj=False)
            else: # If ratio is 0 for qi, effectively it's not used
                self.attn_qi = QuantumInspiredAttentionRFF(config, lmo_for_this_layer, num_attn_heads=0, external_qkv_proj=False)


        self.ln_1 = nn.LayerNorm(config.embed_size)
        self.ln_2 = nn.LayerNorm(config.embed_size)
        if config.ffn_variant=="swiglu": self.ffn=SwiGLUFeedForward(config)
        else: self.ffn=GELUFeedForward(config)

        self.gru,self.gru_proj = None,None
        if config.use_gru_layer:
            ghs = config.gru_hidden_size if config.gru_hidden_size is not None else config.embed_size
            self.gru = nn.GRU(config.embed_size,ghs,batch_first=False)
            if ghs!=config.embed_size: self.gru_proj=nn.Linear(ghs,config.embed_size)
        
        self.adapter_attn = Adapter(config.embed_size, config.adapter_bottleneck_dim_ratio, config.dropout_rate) if config.use_adapters else None
        self.adapter_ffn = Adapter(config.embed_size, config.adapter_bottleneck_dim_ratio, config.dropout_rate) if config.use_adapters else None
        self.resid_dropout = nn.Dropout(config.dropout_rate) ### NEW ### For hybrid_split_heads combined output

    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor]=None, 
                is_causal: bool=False,
                image_embeds_for_cross_attn: Optional[torch.Tensor] = None, 
                image_pad_mask_for_cross_attn: Optional[torch.Tensor] = None,
                cross_attention_module_txt_img: Optional[nn.MultiheadAttention] = None,
                cross_attention_module_img_txt: Optional[nn.MultiheadAttention] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B,N_txt,C_in = x.shape # C_in should be self.full_embed_size
        if N_txt == 0: 
            return x, image_embeds_for_cross_attn 

        x_normed_for_self_attn = self.ln_1(x)
        attn_out_final: torch.Tensor

        if self.att_type == "markovian":
            assert self.attn_markov is not None
            attn_out_final = self.attn_markov(x_normed_for_self_attn, key_padding_mask=key_padding_mask, is_causal=is_causal)
        elif self.att_type == "qi_rff":
            assert self.attn_qi is not None
            attn_out_final = self.attn_qi(x_normed_for_self_attn, key_padding_mask=key_padding_mask, is_causal=is_causal)
        
        ### MODIFIED ###
        elif self.att_type == "hybrid_split_heads":
            assert self.qkv_block_internal is not None and self.proj_block_internal is not None
            
            qkv_out = self.qkv_block_internal(x_normed_for_self_attn) # (B, N, 3*C)
            # q_all, k_all, v_all are (B, H_total, N, Dh)
            q_all, k_all, v_all = qkv_out.view(B, N_txt, 3, self.total_heads, self.head_dim).permute(2,0,3,1,4).unbind(0)
            
            outputs_from_heads = []
            if self.attn_markov and self.num_markov_heads > 0:
                q_m = q_all[:, :self.num_markov_heads, :, :]
                k_m = k_all[:, :self.num_markov_heads, :, :]
                v_m = v_all[:, :self.num_markov_heads, :, :]
                # Pass x_normed_for_self_attn for potential use by non-token-specific gate
                out_m_slice = self.attn_markov(x_normed_for_self_attn, key_padding_mask, is_causal, q_m, k_m, v_m) # (B, N, C_slice_m)
                outputs_from_heads.append(out_m_slice)

            if self.attn_qi and self.num_qi_heads > 0:
                q_q = q_all[:, self.num_markov_heads : self.num_markov_heads + self.num_qi_heads, :, :]
                k_q = k_all[:, self.num_markov_heads : self.num_markov_heads + self.num_qi_heads, :, :]
                v_q = v_all[:, self.num_markov_heads : self.num_markov_heads + self.num_qi_heads, :, :]
                out_q_slice = self.attn_qi(x_normed_for_self_attn, key_padding_mask, is_causal, q_q, k_q, v_q) # (B, N, C_slice_q)
                outputs_from_heads.append(out_q_slice)
            
            if not outputs_from_heads: # Should not happen if validation in config is correct
                attn_output_unprojected = torch.zeros_like(x_normed_for_self_attn)
            else:
                # Concatenate along the feature dimension (C_slice_m and C_slice_q sum to C_full)
                attn_output_unprojected = torch.cat(outputs_from_heads, dim=2) # (B, N, C_full)
            
            attn_out_final = self.proj_block_internal(attn_output_unprojected) # (B, N, C_full)
            # Apply dropout here for the combined output, as sub-modules skip it when external_qkv_proj=True
            if self.training and self.config.dropout_rate > 0:
                 attn_out_final = self.resid_dropout(attn_out_final)


        elif self.att_type == "hybrid_parallel_sum":
            out_m = torch.zeros_like(x_normed_for_self_attn)
            if self.attn_markov and self.attn_markov.heads > 0 : # Check if it was initialized with heads
                out_m = self.attn_markov(x_normed_for_self_attn, key_padding_mask, is_causal)
            
            out_q = torch.zeros_like(x_normed_for_self_attn)
            if self.attn_qi and self.attn_qi.heads > 0: # Check if it was initialized with heads
                 out_q = self.attn_qi(x_normed_for_self_attn, key_padding_mask, is_causal)
            
            attn_out_final = out_m + out_q # Summing outputs, each already (B,N,C_full) and dropout applied internally
        else: 
            attn_out_final = torch.zeros_like(x_normed_for_self_attn) # Should not be reached
        
        x = x + attn_out_final # Residual connection for self-attention
        if self.adapter_attn: x = x + self.adapter_attn(x) 

        # ... (rest of the forward method: cross-attention, GRU, FFN, adapters) ...
        if self.config.is_multimodal and self.config.multimodal_fusion_type == "interleaved_cross_attn" \
           and image_embeds_for_cross_attn is not None and image_embeds_for_cross_attn.shape[1] > 0 \
           and N_txt > 0 \
           and cross_attention_module_txt_img is not None:
            # For cross-attention, we need to decide if x or x_normed_for_self_attn (ln_1(x)) is better.
            # Standard transformer uses ln_1(x) for Q and K,V come from other modality.
            # Here, x is post-self-attention residual. Let's use a fresh normalization for query.
            txt_q_for_img_kv = self.ln_1(x) # Or a new LayerNorm instance if preferred
            cross_attn_out_txt, _ = cross_attention_module_txt_img(
                query=txt_q_for_img_kv, key=image_embeds_for_cross_attn, value=image_embeds_for_cross_attn, 
                key_padding_mask=image_pad_mask_for_cross_attn, 
                need_weights=False 
            )
            x = x + cross_attn_out_txt # Residual for cross-attention

        if self.gru: 
            if x.shape[1] > 0: 
                gru_in = x.permute(1,0,2)
                try:
                    gru_out_seq, _ = self.gru(gru_in)
                    gru_out_r = gru_out_seq.permute(1,0,2)
                    if self.gru_proj:
                        gru_out_r = self.gru_proj(gru_out_r)
                    x = x + gru_out_r # Residual for GRU
                except Exception as e:
                    print(f"Error in GRU layer: x.shape={x.shape}, gru_in.shape={gru_in.shape}, device={x.device}")
                    traceback.print_exc()
                    raise e
        
        x_normed_for_ffn = self.ln_2(x)
        ffn_out = self.ffn(x_normed_for_ffn)
        x = x + ffn_out # Residual for FFN
        if self.adapter_ffn: x = x + self.adapter_ffn(x) 

        updated_image_embeds = image_embeds_for_cross_attn
        if self.config.is_multimodal and self.config.multimodal_fusion_type == "interleaved_cross_attn" \
           and image_embeds_for_cross_attn is not None and image_embeds_for_cross_attn.shape[1] > 0 \
           and N_txt > 0 \
           and cross_attention_module_img_txt is not None: 
            img_q_for_txt_kv = image_embeds_for_cross_attn # Assuming image embeds are already appropriately normed or handled by MHA
            txt_kv_for_img_q = self.ln_2(x) # Use the output of FFN block (post ln_2) as K,V for image query
            cross_attn_out_img, _ = cross_attention_module_img_txt(
                query=img_q_for_txt_kv, key=txt_kv_for_img_q, value=txt_kv_for_img_q,
                key_padding_mask=key_padding_mask, 
                need_weights=False
            )
            if updated_image_embeds is not None: 
                 updated_image_embeds = updated_image_embeds + cross_attn_out_img # Residual for img-txt cross-attention
            
        return x, updated_image_embeds

# ... (ImageEmbedder, MarkovianTransformer, get_lr, log_parameter_stats, test functions, and __main__ remain largely the same)
# Minor adjustments might be needed in MarkovianTransformer _init_weights if proj_block_internal is used
# and needs special scaling, but for now, standard Linear init should apply.
# The test_activation_checkpointing might need adjustment if the new hybrid_split_heads logic changes assumptions
# about parameter names or requires more careful handling of random states if sub-modules behave differently now.

# For _init_weights in MarkovianTransformer:
# The proj_block_internal.weight would be something like 'layers.N.proj_block_internal.weight'
# It should probably get the same scaling as other attention projections.
# Original code:
# if 'layers' in pn and '.attn.' in pn and pn.endswith('.proj.weight'): # Standard attention projection in Markovian or QI
#      is_block_attn_proj = True
#
# Modified check in _init_weights for scaling:
#             is_block_attn_proj = False
#             if 'layers.' in pn:
#                 # Standard attention projection in Markovian or QI (when not external_qkv_proj)
#                 if ('.attn_markov.proj.weight' in pn or '.attn_qi.out_proj.weight' in pn):
#                     # Check if this layer's attention module is NOT external_qkv_proj
#                     try:
#                         layer_idx_str = pn.split('.')[1]
#                         if layer_idx_str.isdigit():
#                             layer_idx = int(layer_idx_str)
#                             block = self.layers[layer_idx]
#                             module_name, _, _ = pn.split('.')[2:5] # e.g. attn_markov
#                             attn_module_in_block = getattr(block, module_name, None)
#                             if attn_module_in_block and not attn_module_in_block.external_qkv_proj:
#                                 is_block_attn_proj = True
#                     except: pass # Best effort
                
#                 # The block's own combined projection for hybrid_split_heads
#                 if f".proj_block_internal.weight" in pn :
#                     try:
#                         layer_idx_str = pn.split('.')[1]
#                         if layer_idx_str.isdigit():
#                             layer_idx = int(layer_idx_str)
#                             if self.config.get_effective_attention_types()[layer_idx] == "hybrid_split_heads":
#                                 is_block_attn_proj = True
#                     except: pass


class MarkovianTransformer(nn.Module):
    def __init__(self, config: MarkovianTransformerConfig):
        super().__init__(); self.config=config
        self.token_embeddings=nn.Embedding(config.vocab_size,config.embed_size)
        if config.use_absolute_pos_embeddings:
            self.pos_embeddings=nn.Embedding(config.max_seq_len,config.embed_size)
            self.register_buffer('precomputed_positions', torch.arange(0, config.max_seq_len, dtype=torch.long), persistent=False)
        else: self.pos_embeddings = None; self.register_buffer('precomputed_positions', None, persistent=False)
        self.embed_dropout=nn.Dropout(config.dropout_rate)

        if config.is_multimodal:
            self.image_embedder = ImageEmbedder(config)
            if config.multimodal_fusion_type == "sequential_cross_attn" and config.num_cross_attention_layers > 0:
                self.text_to_image_cross_attns = nn.ModuleList()
                self.image_to_text_cross_attns = nn.ModuleList() 
                self.ln_img_seq_cross = nn.ModuleList()
                self.ln_txt_seq_cross = nn.ModuleList()
                for _ in range(config.num_cross_attention_layers):
                    self.text_to_image_cross_attns.append(nn.MultiheadAttention(config.embed_size, config.heads, dropout=config.dropout_rate, batch_first=True))
                    self.image_to_text_cross_attns.append(nn.MultiheadAttention(config.embed_size, config.heads, dropout=config.dropout_rate, batch_first=True))
                    self.ln_img_seq_cross.append(nn.LayerNorm(config.embed_size))
                    self.ln_txt_seq_cross.append(nn.LayerNorm(config.embed_size))

        self.interleaved_cross_attn_txt_img: Optional[nn.ModuleList] = None
        self.interleaved_cross_attn_img_txt: Optional[nn.ModuleList] = None
        if config.is_multimodal and config.multimodal_fusion_type == "interleaved_cross_attn" and config.num_layers > 0:
            self.interleaved_cross_attn_txt_img = nn.ModuleList()
            self.interleaved_cross_attn_img_txt = nn.ModuleList()
            for _ in range(config.num_layers):
                self.interleaved_cross_attn_txt_img.append(nn.MultiheadAttention(config.embed_size, config.heads, dropout=config.dropout_rate, batch_first=True))
                self.interleaved_cross_attn_img_txt.append(nn.MultiheadAttention(config.embed_size, config.heads, dropout=config.dropout_rate, batch_first=True))
        
        self.layers=nn.ModuleList([MarkovianTransformerBlock(config,i) for i in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embed_size) if config.num_layers > 0 or \
            (config.is_multimodal and (config.num_cross_attention_layers > 0 or config.multimodal_fusion_type == "interleaved_cross_attn")) \
            else nn.Identity()
        self.head=nn.Linear(config.embed_size,config.num_output_classes,bias=False)

        if config.tie_output_to_embedding_weights:
            if config.vocab_size==config.num_output_classes and config.embed_size==self.token_embeddings.embedding_dim:
                self.head.weight=self.token_embeddings.weight; print("INFO: Output head weights TIED.")
            else: warnings.warn(f"Weight tying requested but conditions not met. NOT tying.", UserWarning)
        
        self.apply(self._init_weights) # Apply generic inits first

        # Apply special scaling (Residual re-scaling)
        for pn,p in self.named_parameters(): 
            if not p.requires_grad: continue
            
            is_gru_proj = (config.use_gru_layer and 
                           config.gru_hidden_size is not None and 
                           config.gru_hidden_size != config.embed_size and 
                           p.ndim == 2 and 'gru_proj.weight' in pn)
            is_adapter_up_proj = config.use_adapters and 'adapter' in pn and 'up_proj.weight' in pn
            
            scale = math.sqrt(2 * config.num_layers) if config.num_layers > 0 else 1.0
            
            # Determine if this parameter is an attention projection weight that needs scaling
            is_scaled_attn_proj = False
            if 'layers.' in pn and p.ndim == 2:
                try:
                    layer_idx_str = pn.split('.')[1]
                    if layer_idx_str.isdigit():
                        layer_idx = int(layer_idx_str)
                        block = self.layers[layer_idx] # type: ignore
                        
                        # Case 1: Standalone attention module's own projection
                        # e.g., layers.0.attn_markov.proj.weight or layers.0.attn_qi.out_proj.weight
                        # This applies if the attention type is "markovian", "qi_rff", or "hybrid_parallel_sum"
                        # where sub-modules handle their own full projection.
                        if (pn.endswith('.attn_markov.proj.weight') and block.attn_markov and not block.attn_markov.external_qkv_proj) or \
                           (pn.endswith('.attn_qi.out_proj.weight') and block.attn_qi and not block.attn_qi.external_qkv_proj):
                            is_scaled_attn_proj = True
                        
                        # Case 2: Block's internal projection for hybrid_split_heads
                        # e.g., layers.0.proj_block_internal.weight
                        elif pn.endswith('.proj_block_internal.weight') and block.att_type == "hybrid_split_heads":
                            is_scaled_attn_proj = True
                except IndexError: pass # Parameter name doesn't match expected structure
                except AttributeError: pass # Module structure doesn't match

            if (is_scaled_attn_proj or pn.endswith('ffn.w2.weight') or is_gru_proj) and \
               not is_adapter_up_proj and scale > 0 :
                # print(f"DEBUG: Scaling param {pn} with std {config.param_init_std / scale:.4g}")
                torch.nn.init.normal_(p, mean=0.0, std=config.param_init_std / scale)
            
            if config.use_gru_layer: # Specific GRU initializations
                if 'gru.weight_ih_l0' in pn: torch.nn.init.xavier_uniform_(p)
                elif 'gru.weight_hh_l0' in pn: torch.nn.init.orthogonal_(p)
                elif 'gru.bias' in pn and p is not None: torch.nn.init.zeros_(p)
    
    def _init_weights(self, m: nn.Module): # General initialization
        if isinstance(m,nn.Linear):
            is_tied = (hasattr(self,'head') and m is self.head and hasattr(self.head,'weight') and hasattr(self,'token_embeddings') and self.head.weight is self.token_embeddings.weight and self.config.tie_output_to_embedding_weights)
            if is_tied: return
            
            is_adapter_up = False
            if self.config.use_adapters and hasattr(self, 'layers') and self.layers:
                 is_adapter_up = any( (isinstance(block, MarkovianTransformerBlock) and block.adapter_attn and m is block.adapter_attn.up_proj) or \
                                      (isinstance(block, MarkovianTransformerBlock) and block.adapter_ffn and m is block.adapter_ffn.up_proj) for block in self.layers) # type: ignore
            if is_adapter_up: return # Adapter up_proj weights are zero-initialized by Adapter itself.

            # Skip scaling here; will be handled by the loop in __init__
            # Standard init for non-scaled linear layers
            if hasattr(m,'weight') and m.weight.requires_grad: 
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.config.param_init_std)
            if hasattr(m,'bias') and m.bias is not None and m.bias.requires_grad: 
                torch.nn.init.zeros_(m.bias)

        elif isinstance(m,nn.Embedding):
            is_tied = (hasattr(self,'token_embeddings') and m is self.token_embeddings and hasattr(self,'head') and hasattr(self.head,'weight') and self.head.weight is self.token_embeddings.weight and self.config.tie_output_to_embedding_weights)
            if is_tied: return

            # Positional embeddings often have specific inits or are left to default, or sometimes normal like token embeddings
            if m is getattr(self, 'pos_embeddings', None) or \
               (self.config.is_multimodal and hasattr(self, 'image_embedder') and isinstance(self.image_embedder, ImageEmbedder) and m is self.image_embedder.pos_embed):
                # Typically, positional embeddings are initialized normally or with zeros, depending on the scheme.
                # Here, we'll use the same normal init as token embeddings unless a specific scheme is desired.
                 torch.nn.init.normal_(m.weight, mean=0.0, std=self.config.param_init_std)
            elif hasattr(m,'weight') and m.weight.requires_grad: # For token_embeddings if not tied, and other embeddings
                torch.nn.init.normal_(m.weight, mean=0.0, std=self.config.param_init_std)
        
        elif isinstance(m,nn.LayerNorm):
            if hasattr(m,'bias') and m.bias is not None: torch.nn.init.zeros_(m.bias)
            if hasattr(m,'weight') and m.weight is not None: torch.nn.init.ones_(m.weight)
    
    # Forward and get_sparsity_loss remain the same from Code 1
    def forward(self, 
                input_ids: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor]=None, 
                is_causal: bool=False,
                image_features: Optional[torch.Tensor]=None,
                image_padding_mask: Optional[torch.Tensor]=None
               ) -> torch.Tensor:
        B,N_txt=input_ids.shape
        if input_ids.dtype!=torch.long: raise TypeError(f"input_ids dtype must be torch.long, got {input_ids.dtype}")
        expected_dtype = self.head.weight.dtype if hasattr(self.head,'weight') and self.head.weight is not None else torch.float32
        if B == 0: return torch.empty((0, N_txt, self.config.num_output_classes), dtype=expected_dtype, device=input_ids.device)

        if not torch.compiler.is_compiling(): 
            if N_txt > 0 and input_ids.numel() > 0:
                min_v, max_v = input_ids.min().item(), input_ids.max().item()
                if not(0 <= min_v < self.config.vocab_size and 0 <= max_v < self.config.vocab_size): raise ValueError(f"input_ids out of vocab_range")
            if N_txt > self.config.max_seq_len: raise ValueError(f"Seq len {N_txt} > max_seq_len {self.config.max_seq_len}")
            if key_padding_mask is not None and key_padding_mask.shape != (B,N_txt): raise ValueError(f"key_padding_mask shape mismatch")
            if key_padding_mask is not None and key_padding_mask.dtype != torch.bool: raise ValueError(f"key_padding_mask dtype not bool")

        txt_x = self.token_embeddings(input_ids)
        if self.pos_embeddings is not None and N_txt > 0:
            pos_ids = self.precomputed_positions[:N_txt].unsqueeze(0) if self.precomputed_positions is not None else torch.arange(0,N_txt,device=input_ids.device).unsqueeze(0)
            txt_x = txt_x + self.pos_embeddings(pos_ids)
        if self.config.dropout_rate > 0: txt_x = self.embed_dropout(txt_x)

        img_x: Optional[torch.Tensor] = None
        N_img = 0 
        if self.config.is_multimodal and hasattr(self, 'image_embedder'):
            if image_features is None:
                 if self.config.multimodal_fusion_type != "sequential_cross_attn" or self.config.num_cross_attention_layers > 0: 
                     warnings.warn("Multimodal model running without image_features. Cross-attention may be skipped.", UserWarning, stacklevel=2)
            else:
                img_x = self.image_embedder(image_features)
                N_img = img_x.shape[1] if img_x.ndim == 3 else 0 
                if image_padding_mask is not None and img_x is not None and img_x.ndim ==3 and N_img > 0 and image_padding_mask.shape != (B, N_img):
                     raise ValueError(f"image_padding_mask shape ({image_padding_mask.shape}) mismatch img_embeds ({img_x.shape})")
        
        if self.config.is_multimodal and img_x is not None and N_txt > 0 and N_img > 0 and \
           self.config.multimodal_fusion_type == "sequential_cross_attn" and self.config.num_cross_attention_layers > 0 and \
           hasattr(self, 'text_to_image_cross_attns') and self.text_to_image_cross_attns is not None: 
            
            current_txt_x, current_img_x = txt_x, img_x
            for i in range(self.config.num_cross_attention_layers):
                normed_txt = self.ln_txt_seq_cross[i](current_txt_x)
                normed_img = self.ln_img_seq_cross[i](current_img_x) 
                
                txt_cross_out, _ = self.text_to_image_cross_attns[i](query=normed_txt, key=normed_img, value=normed_img, key_padding_mask=image_padding_mask, need_weights=False)
                current_txt_x = current_txt_x + txt_cross_out 
            txt_x = current_txt_x
        
        current_img_x_for_interleaved = img_x 
        for i, layer_module in enumerate(self.layers):
            block_args: Dict[str, Any] = {"key_padding_mask": key_padding_mask, "is_causal": is_causal}
            cross_attn_txt_img_mod, cross_attn_img_txt_mod = None, None
            if self.config.is_multimodal and self.config.multimodal_fusion_type == "interleaved_cross_attn" \
               and current_img_x_for_interleaved is not None and \
               self.interleaved_cross_attn_txt_img is not None and self.interleaved_cross_attn_img_txt is not None:
                cross_attn_txt_img_mod = self.interleaved_cross_attn_txt_img[i]
                cross_attn_img_txt_mod = self.interleaved_cross_attn_img_txt[i]
                block_args.update({
                    "image_pad_mask_for_cross_attn": image_padding_mask,
                    "cross_attention_module_txt_img": cross_attn_txt_img_mod,
                    "cross_attention_module_img_txt": cross_attn_img_txt_mod
                })

            if self.config.use_activation_checkpointing and self.training:
                # This wrapper needs to be robust to changes in layer_module's signature
                # or the number/type of arguments it might need for different attention types.
                # Current layer_module signature is (x, image_embeds_for_cross_attn, **block_args)
                def _checkpoint_block_runner_wrapper(current_txt_x_arg, current_img_x_arg_ckpt, **kwargs_for_block_ckpt):
                    return layer_module(current_txt_x_arg, image_embeds_for_cross_attn=current_img_x_arg_ckpt, **kwargs_for_block_ckpt)

                txt_x, current_img_x_for_interleaved = activation_checkpoint_fn(
                    _checkpoint_block_runner_wrapper, 
                    txt_x,                            
                    current_img_x_for_interleaved,    
                    use_reentrant=self.config.activation_checkpointing_reentrant, 
                    preserve_rng_state=True, # Important for reproducibility with dropout/RFF redraw
                    **block_args 
                )
            else:
                txt_x, current_img_x_for_interleaved = layer_module(txt_x, image_embeds_for_cross_attn=current_img_x_for_interleaved, **block_args)
        
        if N_txt > 0: txt_x = self.final_norm(txt_x) 
        logits = self.head(txt_x)
        if logits.dtype != expected_dtype and not torch.is_autocast_enabled(): logits = logits.to(expected_dtype)
        return logits

    def get_sparsity_loss(self) -> torch.Tensor: 
        if self.config.sparsity_regularization_coeff <= 0:
            try: 
                device = next(self.parameters(), torch.tensor(0.0)).device
            except StopIteration:
                device = torch.device("cpu") 
            return torch.tensor(0.0, device=device)

        l1_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for name, module in self.named_modules():
            is_target_type = any(target_type.lower() in str(type(module)).lower() for target_type in self.config.sparsity_target_modules)
            if is_target_type and hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                if not isinstance(module, (nn.LayerNorm, nn.GRU)): 
                     l1_loss = l1_loss + torch.norm(module.weight, p=1)
        return self.config.sparsity_regularization_coeff * l1_loss

# ... (The rest of the file: get_lr, log_parameter_stats, test functions, __main__)
# Ensure __main__ and test functions use the updated config and model.
# For example, in __main__, the instantiation of MarkovianTransformer will now use the refactored components.
# The test for activation checkpointing might need scrutiny, as the way `preserve_rng_state` interacts with
# the potentially more complex internal state of the refactored attention modules (especially if RFF redraw
# is involved within a checkpointed segment for `hybrid_split_heads`) could be tricky.
# However, `use_reentrant=False` for checkpointing is generally more robust with complex kwargs.
# The test_markovian_bias_fallback will also need to ensure it targets the correct attn_markov module,
# which might be nested if `hybrid_split_heads` is default.
# The parameter scaling logic in MarkovianTransformer.__init__ has been updated to reflect the new structure.

def get_lr(it: int, config: MarkovianTransformerConfig) -> float:
    lr_i,lr_f=config.learning_rate,config.learning_rate*config.min_lr_ratio
    w_iters,d_phase_dur=config.warmup_iters,config.lr_decay_iters
    if w_iters>0 and it<w_iters: return lr_i*(it+1)/w_iters
    if d_phase_dur==0: return lr_i 
    it_d = it-w_iters
    if it_d>=d_phase_dur: return lr_f
    d_ratio = it_d/d_phase_dur
    if not(0<=d_ratio<=1):
        warnings.warn(f"LR decay ratio ({d_ratio}) out of bounds for iter {it}. Clamping.",UserWarning)
        d_ratio=max(0.0,min(1.0,d_ratio))
    coeff = 0.5*(1.0+math.cos(math.pi*d_ratio))
    return lr_f+coeff*(lr_i-lr_f)

def log_parameter_stats(model:nn.Module,model_name:str="Model",log_non_trainable=False,log_grads=True):
    print(f"\n--- Param Stats: {model_name} ---")
    param_infos = []
    total_params, trainable_params = 0, 0
    max_name_len = 0
    for name, param in model.named_parameters():
        max_name_len = max(max_name_len, len(name))
        param_infos.append((name, param))
    header_format = f"  {{:<{max_name_len}}} {{:<30}} | {{:<55}} "
    if log_grads: header_format += "| {}"
    print(header_format.format("Param Name","Status(Shape,Count)","Value(min,max,mean,std,absmean,nan,inf)",
                               "Grad(min,max,mean,std,absmean,nan,inf)" if log_grads else ""))
    separator_line = f"  {'-'*max_name_len} {'-'*30} | {'-'*55} "
    if log_grads: separator_line += f"| {'-'*55}"
    print(separator_line)
    for name, param in param_infos:
        numel = param.numel(); total_params += numel
        status_str = f"({list(param.shape)},{numel})"
        value_stats_str = "N/A (Empty Tensor)"
        if numel > 0:
            try:
                p_cpu_f32 = param.data.detach().cpu().to(torch.float32)
                finite_p = p_cpu_f32[torch.isfinite(p_cpu_f32)]
                v_min,v_max,v_mean,v_std,v_absmean = (float('nan'),)*5
                if finite_p.numel()>0: v_min,v_max,v_mean,v_std,v_absmean = finite_p.min().item(),finite_p.max().item(),finite_p.mean().item(),finite_p.std().item(),finite_p.abs().mean().item()
                nan_c,inf_c = torch.isnan(p_cpu_f32).sum().item(),torch.isinf(p_cpu_f32).sum().item()
                value_stats_str=(f"min={v_min:+.1e},max={v_max:+.1e},mu={v_mean:+.1e},sig={v_std:.1e},amu={v_absmean:.1e},nan={nan_c},inf={inf_c}")
            except Exception as e: value_stats_str=f"Error value stats: {e}"
        if param.requires_grad:
            trainable_params += numel
            grad_stats_str = "N/A (No Grad/Not Required)"
            if log_grads and param.grad is not None:
                if param.grad.numel()>0:
                    try:
                        g_cpu_f32 = param.grad.detach().cpu().to(torch.float32)
                        finite_g = g_cpu_f32[torch.isfinite(g_cpu_f32)]
                        g_min,g_max,g_mean,g_std,g_absmean = (float('nan'),)*5
                        if finite_g.numel()>0: g_min,g_max,g_mean,g_std,g_absmean = finite_g.min().item(),finite_g.max().item(),finite_g.mean().item(),finite_g.std().item(),finite_g.abs().mean().item()
                        nan_g_c,inf_g_c = torch.isnan(g_cpu_f32).sum().item(),torch.isinf(g_cpu_f32).sum().item()
                        grad_stats_str=(f"min={g_min:+.1e},max={g_max:+.1e},mu={g_mean:+.1e},sig={g_std:.1e},amu={g_absmean:.1e},nan={nan_g_c},inf={inf_g_c}")
                    except Exception as e: grad_stats_str=f"Error grad stats: {e}"
                else: grad_stats_str = "Empty Grad Tensor"
            fmt_args = [name, "TRAIN " + status_str, value_stats_str]
            if log_grads: fmt_args.append(grad_stats_str)
            print(header_format.format(*fmt_args))
        elif log_non_trainable:
            fmt_args = [name, "NON-TRAIN " + status_str, value_stats_str]
            if log_grads: fmt_args.append("Grad: N/A")
            print(header_format.format(*fmt_args))
    print(f"\n  Total params: {total_params/1e6:.3f}M, Trainable: {trainable_params/1e6:.3f}M")
    try: device_info = next(model.parameters()).device if total_params > 0 else 'N/A (No params)'
    except StopIteration: device_info = 'N/A (No params by next())'
    print(f"  Device: {device_info}\n--- End Stats ---")

def assert_allclose_or_pytest(a, b, msg_prefix="", **kwargs):
    is_close = torch.allclose(a, b, **kwargs)
    max_diff_val = float('nan')
    if a.numel() > 0 and b.numel() > 0 and a.shape == b.shape:
        try: max_diff_val = (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()
        except Exception: pass
    message = f"{msg_prefix} Max diff: {max_diff_val:.6g}"
    if PYTEST_AVAILABLE and hasattr(pytest, 'assume'): pytest.assume(is_close, message)
    else: assert is_close, message

def test_activation_checkpointing(base_config: MarkovianTransformerConfig, device_to_test_on):
    print("\n--- Test: Activation Checkpointing ---")
    cfg_ckpt = copy.deepcopy(base_config)
    cfg_ckpt.use_activation_checkpointing = True
    cfg_ckpt.activation_checkpointing_reentrant = False 
    cfg_ckpt.compile_model = False 
    if cfg_ckpt.dropout_rate == 0.0: cfg_ckpt.dropout_rate = 0.1; print(f"INFO: CKPT test forcing dropout to {cfg_ckpt.dropout_rate}.")
    
    cfg_no_ckpt = copy.deepcopy(cfg_ckpt)
    cfg_no_ckpt.use_activation_checkpointing = False
    
    if cfg_ckpt.num_layers == 0: print("SKIP CKPT: num_layers=0."); pytest.skip("num_layers=0") if PYTEST_AVAILABLE else None; return
    
    is_interleaved_mm = cfg_ckpt.is_multimodal and cfg_ckpt.multimodal_fusion_type == "interleaved_cross_attn"
    if is_interleaved_mm and cfg_ckpt.use_activation_checkpointing: 
        print("SKIP CKPT: Interleaved MM checkpointing wrapper is complex to test generically here.")
        pytest.skip("Interleaved MM checkpointing with complex args.") if PYTEST_AVAILABLE else None
        return

    try:
        torch.manual_seed(123); m_ckpt = MarkovianTransformer(cfg_ckpt).to(device_to_test_on)
        torch.manual_seed(123); m_no_ckpt = MarkovianTransformer(cfg_no_ckpt).to(device_to_test_on)
        B, N_s = 2, min(cfg_ckpt.max_seq_len // 2, 16) if cfg_ckpt.max_seq_len > 0 else 0
        ids = torch.randint(0, cfg_ckpt.vocab_size, (B, N_s), device=device_to_test_on, dtype=torch.long) if N_s > 0 else torch.empty(B,0,dtype=torch.long, device=device_to_test_on)
        tgts = torch.randint(0, cfg_ckpt.num_output_classes, (B, N_s), device=device_to_test_on, dtype=torch.long) if N_s > 0 else torch.empty(B,0,dtype=torch.long, device=device_to_test_on)
        
        img_f_test, img_pad_mask_test = None, None
        if cfg_ckpt.is_multimodal:
            num_img_patches_test = min(8, cfg_ckpt.num_image_patches) if cfg_ckpt.num_image_patches > 0 else 0
            if num_img_patches_test > 0:
                img_f_test = torch.randn(B, num_img_patches_test, cfg_ckpt.image_feature_dim, device=device_to_test_on)
                img_pad_mask_test = torch.zeros(B, num_img_patches_test, dtype=torch.bool, device=device_to_test_on) 

        m_ckpt.eval(); m_no_ckpt.eval()
        with torch.no_grad():
            l_ckpt_e = m_ckpt(ids.clone(), image_features=img_f_test.clone() if img_f_test is not None else None, image_padding_mask=img_pad_mask_test)
            l_no_ckpt_e = m_no_ckpt(ids.clone(), image_features=img_f_test.clone() if img_f_test is not None else None, image_padding_mask=img_pad_mask_test)
        
        if N_s > 0 : 
            assert_allclose_or_pytest(l_ckpt_e, l_no_ckpt_e, atol=1e-5, rtol=1e-4, msg_prefix="CKPT Eval Fwd")
        print(f"CKPT Eval Fwd: PASSED (shape {l_ckpt_e.shape})")

        m_ckpt.train(); m_no_ckpt.train()
        seed_fwd_bwd = 456
        torch.manual_seed(seed_fwd_bwd); opt_c = torch.optim.SGD(filter(lambda p:p.requires_grad, m_ckpt.parameters()), lr=1e-1); opt_c.zero_grad()
        l_c_tr = m_ckpt(ids.clone(), image_features=img_f_test.clone() if img_f_test is not None else None, image_padding_mask=img_pad_mask_test)
        
        loss_c = torch.tensor(0.0, device=device_to_test_on, requires_grad=True)
        if N_s > 0 and l_c_tr.numel() > 0 and (tgts.view(-1) != -100).any(): 
            loss_c = F.cross_entropy(l_c_tr.view(-1, cfg_ckpt.num_output_classes), tgts.view(-1), ignore_index=-100)
        if loss_c.requires_grad and loss_c.isfinite(): loss_c.backward()
        gr_c = {n: p.grad.clone() for n, p in m_ckpt.named_parameters() if p.grad is not None and p.requires_grad}

        torch.manual_seed(seed_fwd_bwd); opt_nc = torch.optim.SGD(filter(lambda p:p.requires_grad, m_no_ckpt.parameters()), lr=1e-1); opt_nc.zero_grad()
        l_nc_tr = m_no_ckpt(ids.clone(), image_features=img_f_test.clone() if img_f_test is not None else None, image_padding_mask=img_pad_mask_test)
        
        loss_nc = torch.tensor(0.0, device=device_to_test_on, requires_grad=True)
        if N_s > 0 and l_nc_tr.numel() > 0 and (tgts.view(-1) != -100).any():
            loss_nc = F.cross_entropy(l_nc_tr.view(-1, cfg_no_ckpt.num_output_classes), tgts.view(-1), ignore_index=-100)
        if loss_nc.requires_grad and loss_nc.isfinite(): loss_nc.backward()
        gr_nc = {n: p.grad.clone() for n, p in m_no_ckpt.named_parameters() if p.grad is not None and p.requires_grad}

        if N_s > 0 and l_c_tr.numel() > 0 and l_nc_tr.numel() > 0 :
            assert_allclose_or_pytest(l_c_tr, l_nc_tr, atol=1e-5, rtol=1e-4, msg_prefix="CKPT Train Fwd (logits)")
            print(f"CKPT Train Fwd (logits): PASSED (shape {l_c_tr.shape})")
            
            grads_ok = True
            if not any(p.requires_grad for p in m_ckpt.parameters()): print("WARN: CKPT no trainable params."); pytest.skip("No trainable params.") if PYTEST_AVAILABLE else None; return
            
            if not gr_c and not gr_nc and any(p.requires_grad for p in m_ckpt.parameters()):
                print("WARN: No grads produced by either model despite trainable params. Loss or data issue?")
            elif len(gr_c) != len(gr_nc): grads_ok = False; print(f"FAIL: Grad count mismatch. Ckpt:{len(gr_c)}, NoCkpt:{len(gr_nc)}")
            elif not gr_c and any(p.requires_grad for p in m_ckpt.parameters()): grads_ok = False; print(f"FAIL: No grads in ckpt, but trainable params exist.")
            else:
                for n in gr_c:
                    if n not in gr_nc: grads_ok=False; print(f"FAIL: Grad {n} missing in NoCkpt."); break
                    if not torch.allclose(gr_c[n], gr_nc[n], atol=1e-5, rtol=1e-3):
                        max_d=(gr_c[n]-gr_nc[n]).abs().max().item(); grads_ok=False; print(f"FAIL: Grad mismatch '{n}'. MaxD: {max_d:.4g}"); break
                for n_nc in gr_nc: 
                    if n_nc not in gr_c: grads_ok=False; print(f"FAIL: Grad {n_nc} missing in Ckpt."); break
            
            if grads_ok: print(f"CKPT Bwd (gradients): PASSED.")
            else: print(f"CKPT Bwd (gradients): FAILED."); assert grads_ok, "CKPT Grads differ."
        else: print("CKPT Test: SKIPPED fwd/bwd compare (empty logits or no valid targets).")
    except Exception as e: print(f"CKPT Test: FAILED exception - {e}"); traceback.print_exc(limit=3); pytest.fail(f"CKPT test exception: {e}") if PYTEST_AVAILABLE else None

def test_weight_tying(base_config, device_to_test_on):
    print("\n--- Test: Weight Tying ---")
    cfg = copy.deepcopy(base_config)
    cfg.tie_output_to_embedding_weights=True
    cfg.num_output_classes=cfg.vocab_size 
    cfg.compile_model=False

    try:
        m=MarkovianTransformer(cfg).to(device_to_test_on)
        expected_to_be_tied = (cfg.vocab_size==cfg.num_output_classes and
                               cfg.embed_size==m.token_embeddings.embedding_dim)
        test_passed_ok=False

        if expected_to_be_tied:
            if m.head.weight is m.token_embeddings.weight:
                print("Tying: PASSED - Head and token_embedding weights share memory, as expected.")
                test_passed_ok=True
            else:
                print("Tying: FAILED - Head and token_embedding weights DO NOT share memory, but were expected to.")
                test_passed_ok=False

            if test_passed_ok and any(p.requires_grad for p in m.parameters()):
                opt=torch.optim.SGD(filter(lambda p:p.requires_grad,m.parameters()),lr=1e-3)
                m.train()
                opt.zero_grad()
                
                N_s_tying = max(1, cfg.max_seq_len // 4 if cfg.max_seq_len > 0 else 4)
                ids=torch.randint(0,cfg.vocab_size,(2,N_s_tying),device=device_to_test_on,dtype=torch.long) if cfg.vocab_size > 0 else torch.empty(2,0,dtype=torch.long, device=device_to_test_on)
                if ids.numel() == 0 and cfg.vocab_size > 0: 
                    print("Tying: SKIP gradient flow test - input_ids are empty for tying test."); return

                loss_val = m(ids)
                if loss_val.numel() == 0: 
                     print("Tying: SKIP gradient flow test - output logits are empty."); return
                
                loss=loss_val.sum()
                if loss.requires_grad and loss.isfinite(): loss.backward()

                emb_grad_exists = m.token_embeddings.weight.grad is not None
                emb_grad_non_zero = emb_grad_exists and m.token_embeddings.weight.grad.abs().sum().item() > EPSILON_CLAMP

                if emb_grad_non_zero: print(f"Tying: Gradient flow to token_embeddings.weight confirmed (grad sum: {m.token_embeddings.weight.grad.abs().sum().item():.3g}).")
                elif emb_grad_exists: print(f"Tying: WARN - Gradient for token_embeddings.weight exists but is zero (sum: {m.token_embeddings.weight.grad.abs().sum().item():.3g}).")
                else: 
                    if any(p.requires_grad for p in m.parameters()): print("Tying: FAILED - No gradient flowed to token_embeddings.weight."); test_passed_ok=False
                    else: print("Tying: SKIP grad flow check - no trainable parameters.")
            elif test_passed_ok and not any(p.requires_grad for p in m.parameters()):
                print("Tying: SKIP gradient flow test - no trainable parameters in the model.")
        else: 
            if m.head.weight is not m.token_embeddings.weight:
                print("Tying: PASSED - Head and token_embedding weights DO NOT share memory, as expected (conditions for tying not met).")
                test_passed_ok=True
            else: print("Tying: FAILED - Head and token_embedding weights SHARE memory, but were NOT expected to."); test_passed_ok=False

        if PYTEST_AVAILABLE and hasattr(pytest,'assume'): pytest.assume(test_passed_ok)
        elif not test_passed_ok: raise AssertionError("Weight Tying Test Failed based on logged messages.")
    except Exception as e:
        print(f"Tying Test: FAILED due to an exception - {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        if PYTEST_AVAILABLE: pytest.fail(f"Tying test raised an exception: {e}")

def test_gru_influence(base_config, device_to_test_on):
    print("\n--- Test: GRU Influence ---")
    if base_config.num_layers==0:
        print("SKIP GRU test: num_layers=0, GRU is within transformer blocks.")
        if PYTEST_AVAILABLE: pytest.skip("GRU is in blocks, num_layers=0")
        return

    try:
        cfg_with_gru=copy.deepcopy(base_config); cfg_with_gru.use_gru_layer=True; cfg_with_gru.compile_model=False
        cfg_without_gru=copy.deepcopy(base_config); cfg_without_gru.use_gru_layer=False; cfg_without_gru.compile_model=False

        torch.manual_seed(0)
        m_with_gru=MarkovianTransformer(cfg_with_gru).to(device_to_test_on)
        torch.manual_seed(0) 
        m_without_gru=MarkovianTransformer(cfg_without_gru).to(device_to_test_on)

        N_s_gru = max(1, cfg_with_gru.max_seq_len // 4 if cfg_with_gru.max_seq_len > 0 else 4)
        ids=torch.randint(0,cfg_with_gru.vocab_size,(2,N_s_gru),device=device_to_test_on,dtype=torch.long)

        m_with_gru.eval(); m_without_gru.eval()
        with torch.no_grad():
            logits_with_gru=m_with_gru(ids.clone())
            logits_without_gru=m_without_gru(ids.clone())

        if logits_with_gru.numel() == 0 or logits_without_gru.numel() == 0:
             print("GRU Influence: SKIPPED comparison - output logits are empty.")
             return

        outputs_differ = not torch.allclose(logits_with_gru, logits_without_gru, atol=1e-6)

        if outputs_differ:
            max_abs_diff = (logits_with_gru - logits_without_gru).abs().max().item()
            print(f"GRU Influence: PASSED - Outputs differ when GRU layer is toggled (max absolute difference: {max_abs_diff:.4g}).")
        else:
            print("GRU Influence: FAILED/No Effect - Outputs are identical with and without the GRU layer.")

        if PYTEST_AVAILABLE and hasattr(pytest,'assume'):
            pytest.assume(outputs_differ, "GRU layer toggle did not change model output.")
        elif not outputs_differ:
            raise AssertionError("GRU layer should influence output, but outputs remained identical.")

    except Exception as e:
        print(f"GRU Influence Test: FAILED due to an exception - {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        if PYTEST_AVAILABLE: pytest.fail(f"GRU influence test raised an exception: {e}")

def test_markovian_bias_fallback(base_config, device_to_test_on):
    print("\n--- Test: Markovian Bias Fallback Logic (Missing Backward Transition) ---")
    cfg = copy.deepcopy(base_config)
    cfg.use_bidirectional_markov=True 
    cfg.compile_model=False
    cfg.num_layers=max(1,cfg.num_layers) 
    # Ensure layer 0 uses markovian attention for this test
    cfg.default_attention_type = "markovian" 
    if cfg.attention_type_per_layer is not None and len(cfg.attention_type_per_layer) >=1 :
        cfg.attention_type_per_layer[0] = "markovian"
    elif cfg.attention_type_per_layer is None and cfg.num_layers >=1:
        cfg.attention_type_per_layer = ["markovian"] * cfg.num_layers


    if not cfg.layer_max_orders or len(cfg.layer_max_orders) != cfg.num_layers:
        cfg.layer_max_orders=[max(1,cfg.max_order if cfg.max_order > 0 else 1)]*cfg.num_layers
    else: 
        cfg.layer_max_orders=[max(1,lmo if lmo > 0 else 1) for lmo in cfg.layer_max_orders]
    if cfg.num_layers > 0 and cfg.layer_max_orders[0] == 0: cfg.layer_max_orders[0] = 1

    attn_layer_0 = None
    original_transition_backward_param = None
    model_for_test = MarkovianTransformer(cfg).to(device_to_test_on) 

    try:
        if not model_for_test.layers or not isinstance(model_for_test.layers[0], MarkovianTransformerBlock) :
            print("SKIP Fallback Test: No suitable layers or block type.")
            if PYTEST_AVAILABLE: pytest.skip("No layers or wrong block type for fallback test")
            return

        block0 = model_for_test.layers[0] # type: ignore
        # Access the correct attention module based on the block's structure
        if block0.att_type == "markovian":
            attn_layer_0 = block0.attn_markov
        elif block0.att_type == "hybrid_split_heads" or block0.att_type == "hybrid_parallel_sum":
             attn_layer_0 = block0.attn_markov # Assume it exists for hybrid
        else: # e.g. qi_rff
             print(f"SKIP Fallback Test: Layer 0 attention type is {block0.att_type}, not Markovian-based.")
             if PYTEST_AVAILABLE: pytest.skip("Layer 0 not Markovian-based for fallback test.")
             return
        
        if not attn_layer_0 or not isinstance(attn_layer_0, LearnableMarkovianAttention) or attn_layer_0.heads == 0:
            print(f"SKIP Fallback Test: Layer 0 does not have an active LearnableMarkovianAttention module. Found: {type(attn_layer_0)}")
            if PYTEST_AVAILABLE: pytest.skip("No active LearnableMarkovianAttention in layer 0 for fallback test.")
            return
        
        can_run_test = (attn_layer_0.layer_max_order > 0 and
                        attn_layer_0.trans_config_use_bidir and 
                        hasattr(attn_layer_0,'transition_backward') and
                        isinstance(attn_layer_0.transition_backward, nn.Parameter))

        if not can_run_test:
            print(f"SKIP Fallback Test: Conditions not met. LMO={attn_layer_0.layer_max_order}, "
                  f"trans_config_use_bidir={attn_layer_0.trans_config_use_bidir}, "
                  f"has_tb_param={isinstance(getattr(attn_layer_0,'transition_backward',None),nn.Parameter)}")
            if PYTEST_AVAILABLE: pytest.skip("Layer 0 conditions not met for fallback test.")
            return

        original_transition_backward_param = attn_layer_0.transition_backward
        delattr(attn_layer_0, 'transition_backward')

        N_s_fallback = max(1, cfg.max_seq_len // 4 if cfg.max_seq_len > 0 else 4)
        ids=torch.randint(0,cfg.vocab_size,(2,N_s_fallback),device=device_to_test_on,dtype=torch.long)
        
        expected_warning_regex_str = r"Bidir Markov, but 'transition_backward' missing\. Uni-dir bias\."
        
        with pytest.warns(UserWarning, match=expected_warning_regex_str) as recorded_warnings:
            model_for_test.eval()
            _ = model_for_test(ids)

        warning_issued = len(recorded_warnings) > 0

        if warning_issued:
            print("Fallback Test: PASSED - Expected UserWarning regarding missing 'transition_backward' was captured.")
        else:
            print(f"Fallback Test: FAILED - Expected UserWarning (regex: '{expected_warning_regex_str}') was NOT captured.")
            all_warnings_during_op = warnings.catch_warnings(record=True) 
            with all_warnings_during_op as w_list_debug:
                warnings.simplefilter("always") 
                _ = model_for_test(ids.clone()) 
                print(f"  DEBUG: All UserWarnings emitted during re-run: {[str(warn_msg.message) for warn_msg in w_list_debug if issubclass(warn_msg.category, UserWarning)]}")

        if PYTEST_AVAILABLE and hasattr(pytest,'assume'):
            pytest.assume(warning_issued, "Fallback warning for missing transition_backward not issued.")
        elif not warning_issued:
            raise AssertionError(f"Fallback warning (regex: '{expected_warning_regex_str}') not issued.")

    except Exception as e:
        print(f"Fallback Test: FAILED due to an exception - {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        if PYTEST_AVAILABLE: pytest.fail(f"Fallback test raised an exception: {e}")
    finally:
        if attn_layer_0 and original_transition_backward_param is not None and \
           hasattr(attn_layer_0,'transition_forward') and \
           not hasattr(attn_layer_0, 'transition_backward'):
            setattr(attn_layer_0, 'transition_backward', original_transition_backward_param)
            print("INFO: Restored 'transition_backward' parameter in test_markovian_bias_fallback.")


if __name__ == '__main__':
    # --- Pytest Fallback Setup ---
    if not PYTEST_AVAILABLE:
        print("INFO: Pytest not available. Using fallback for warns/raises/skip/fail.")
        # This setup is done globally at the top of the file now.
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Global: Using device: {device}, PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: True, Version: {torch.version.cuda}, Devices: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0: print(f"Active CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else: print(f"CUDA available: False")
    print("-" * 50)

    main_config_params = dict(
        vocab_size=100, num_output_classes=100, embed_size=64,
        num_layers=2, heads=4, max_order=2, layer_max_orders=None,
        dropout_rate=0.0 if FORCE_DETERMINISTIC_TESTING else 0.1,
        max_seq_len=32,
        
        default_attention_type = "hybrid_split_heads", # Test the new hybrid_split_heads
        attention_type_per_layer = ["markovian", "hybrid_split_heads"], # Example per-layer
        hybrid_attention_markov_head_ratio = 0.5, 

        use_adapters=True,                           
        qi_attention_rff_dim_ratio=0.5,              
        qi_attention_redraw_rff_on_train=False, # Keep False for easier grad checking if needed
        is_multimodal=True,                          
        image_feature_dim=32,                        
        num_image_patches=8,                         
        num_cross_attention_layers=1,                
        multimodal_fusion_type="interleaved_cross_attn", 
        sparsity_regularization_coeff=1e-5,          
        order_gate_variant="deep_mlp",               
        
        token_specific_order_gate=True, head_specific_transitions=True,
        use_bidirectional_markov=True, constrain_transitions=True,
        transition_softplus_arg_offset=1e-3, ffn_variant="swiglu",
        tie_output_to_embedding_weights=False, 
        use_gru_layer=True, gru_hidden_size=None, 
        
        compile_model=False, # Keep False for easier debugging of new features
        activation_checkpointing_reentrant=False, 
        use_activation_checkpointing=False, # Set to True to test checkpointing with new hybrid

        label_smoothing=0.0, learning_rate=3e-4,
        weight_decay=0.01, grad_clip=1.0,
        warmup_iters=2, lr_decay_iters=10, min_lr_ratio=0.1,
    )

    main_config = MarkovianTransformerConfig(**main_config_params) 
    # layer_max_orders automatically handled by __post_init__ if None based on max_order and num_layers
    # Or, if attention_type_per_layer is set, ensure layer_max_orders is also set if its length matters.
    # The __post_init__ correctly sets self.layer_max_orders = [self.max_order] * self.num_layers if it's None.


    torch.manual_seed(42); torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
    
    cfg_main_advanced = copy.deepcopy(main_config)

    print(f"--- Main Model Instance with Advanced Features Enabled (Refactored Hybrid) ---")
    print(cfg_main_advanced)
    
    model_advanced = MarkovianTransformer(cfg_main_advanced).to(device)
    log_parameter_stats(model_advanced, f"Initial Main Advanced Model ({cfg_main_advanced.num_layers}L, Refactored Hybrid)", log_grads=False)

    B, N_s_main = 2, min(cfg_main_advanced.max_seq_len // 2, 16) if cfg_main_advanced.max_seq_len > 0 else 0
    
    ids_main_run = torch.randint(0,cfg_main_advanced.vocab_size,(B,N_s_main),device=device,dtype=torch.long) if N_s_main > 0 else torch.empty(B,0,dtype=torch.long, device=device)
    img_feats_main_run, img_pad_mask_main_run = None, None

    if cfg_main_advanced.is_multimodal:
        num_patches_main_run = cfg_main_advanced.num_image_patches if N_s_main == 0 and cfg_main_advanced.num_image_patches > 0 else min(N_s_main if N_s_main > 0 else float('inf'), cfg_main_advanced.num_image_patches)
        num_patches_main_run = max(0, num_patches_main_run)
        if num_patches_main_run > 0:
            img_feats_main_run = torch.randn(B, num_patches_main_run, cfg_main_advanced.image_feature_dim, device=device)
            img_pad_mask_main_run = torch.zeros(B, img_feats_main_run.shape[1], dtype=torch.bool, device=device)
            if img_feats_main_run.shape[1] > 1: img_pad_mask_main_run[:, -img_feats_main_run.shape[1]//2:] = True
        else: # num_patches_main_run is 0
            img_feats_main_run = torch.empty(B, 0, cfg_main_advanced.image_feature_dim, device=device) # Ensure correct empty tensor
            img_pad_mask_main_run = torch.empty(B, 0, dtype=torch.bool, device=device)


    print(f"\n--- Main Advanced Model Test (Forward Pass, Refactored Hybrid) ---")
    try:
        model_advanced.eval()
        with torch.no_grad():
            logits_adv = model_advanced(ids_main_run.clone(), 
                                     image_features=img_feats_main_run.clone() if img_feats_main_run is not None else None,
                                     image_padding_mask=img_pad_mask_main_run.clone() if img_pad_mask_main_run is not None else None)
        print(f"Logits (advanced config, refactored) shape: {logits_adv.shape}"); 
        assert logits_adv.shape==(B,N_s_main,cfg_main_advanced.num_output_classes)
        print(f"Main Advanced Model Test (Refactored Hybrid): PASSED")
    except Exception as e:
        print(f"Main Advanced Model Test (Refactored Hybrid): FAILED - {e}"); traceback.print_exc(limit=3) # Increased limit
        if PYTEST_AVAILABLE: pytest.fail("Main advanced model forward pass (refactored hybrid) failed.")

    if N_s_main > 0 and any(p.requires_grad for p in model_advanced.parameters()):
        print(f"\n--- Example Training Iterations (Main Advanced Model, Refactored Hybrid) ---")
        model_advanced.train()
        optimizer_adv = torch.optim.AdamW(filter(lambda p:p.requires_grad, model_advanced.parameters()), 
                                      lr=cfg_main_advanced.learning_rate, weight_decay=cfg_main_advanced.weight_decay)
        current_iter_count = 0
        NUM_TRAIN_ITERS = 3 
        print(f"Starting training for {NUM_TRAIN_ITERS} iterations...")
        for iter_num in range(NUM_TRAIN_ITERS):
            current_lr = get_lr(current_iter_count, cfg_main_advanced)
            for param_group in optimizer_adv.param_groups: param_group['lr'] = current_lr
            optimizer_adv.zero_grad(set_to_none=True)

            train_ids_adv = ids_main_run 
            train_img_feats_adv = img_feats_main_run
            train_img_pad_mask_adv = img_pad_mask_main_run
            
            train_targets_adv = train_ids_adv.clone()
            ignore_idx = -100
            train_key_padding_mask_adv = torch.zeros_like(train_ids_adv, dtype=torch.bool) if N_s_main > 0 else None 
            if N_s_main > 0 and B > 0 and train_key_padding_mask_adv is not None: 
                 train_targets_adv[0, N_s_main//2:] = ignore_idx 
                 train_key_padding_mask_adv[0, N_s_main//2:] = True


            logits = model_advanced(train_ids_adv, 
                                key_padding_mask=train_key_padding_mask_adv, 
                                is_causal=True, 
                                image_features=train_img_feats_adv,
                                image_padding_mask=train_img_pad_mask_adv)
            
            loss_main_val = torch.tensor(0.0, device=device, dtype=logits.dtype)
            if N_s_main > 0 and logits.numel() > 0 and (train_targets_adv.view(-1) != ignore_idx).any():
                 loss_main_val = F.cross_entropy(logits.reshape(-1, cfg_main_advanced.num_output_classes), 
                                                 train_targets_adv.reshape(-1), 
                                                 ignore_index=ignore_idx,
                                                 label_smoothing=cfg_main_advanced.label_smoothing)
            
            loss_sparsity_val = model_advanced.get_sparsity_loss()
            total_loss_val = loss_main_val + loss_sparsity_val
            
            if total_loss_val.requires_grad and total_loss_val.isfinite():
                total_loss_val.backward()
                if cfg_main_advanced.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_advanced.parameters(), cfg_main_advanced.grad_clip)
                optimizer_adv.step()
            else:
                print(f"WARN Iter {current_iter_count}: Skipping backward/step. Loss requires_grad: {total_loss_val.requires_grad}, Loss is_finite: {total_loss_val.isfinite()}")

            
            print(f"Iter {current_iter_count}: LR={current_lr:.1e}, Loss={total_loss_val.item():.3f} (Main: {loss_main_val.item():.3f}, Sparsity: {loss_sparsity_val.item():.3e})")
            current_iter_count+=1
        
        log_parameter_stats(model_advanced,f"Main Advanced Model After Training (Refactored Hybrid)",log_grads=True)
    else:
        print("SKIP training loop example for advanced model (refactored): N_s_main is 0 or no trainable parameters.")


    print("\n\n--- Running Isolated Feature Test Configurations (Refactored Hybrid) ---")
    original_main_config_params_iso = dict( 
        vocab_size=100, num_output_classes=100, embed_size=64,
        num_layers=2, heads=4, max_order=2, layer_max_orders=None, # Will be auto-filled
        dropout_rate=0.0 if FORCE_DETERMINISTIC_TESTING else 0.1, max_seq_len=32,
        default_attention_type="markovian", 
        use_adapters=False, is_multimodal=False, 
        image_feature_dim=32, num_image_patches=8, num_cross_attention_layers=1, 
        sparsity_regularization_coeff=0.0, order_gate_variant="mlp", compile_model=False,
        activation_checkpointing_reentrant=False 
    )
    original_main_config = MarkovianTransformerConfig(**original_main_config_params_iso)

    cfg_run_qi_isolated = copy.deepcopy(original_main_config); cfg_run_qi_isolated.default_attention_type = "qi_rff"
    cfg_run_mm_seq_isolated = copy.deepcopy(original_main_config); cfg_run_mm_seq_isolated.is_multimodal = True; cfg_run_mm_seq_isolated.multimodal_fusion_type = "sequential_cross_attn"
    cfg_run_mm_int_isolated = copy.deepcopy(original_main_config); cfg_run_mm_int_isolated.is_multimodal = True; cfg_run_mm_int_isolated.multimodal_fusion_type = "interleaved_cross_attn"
    cfg_run_hybrid_split_isolated = copy.deepcopy(original_main_config); cfg_run_hybrid_split_isolated.default_attention_type = "hybrid_split_heads"
    cfg_run_hybrid_parallel_isolated = copy.deepcopy(original_main_config); cfg_run_hybrid_parallel_isolated.default_attention_type = "hybrid_parallel_sum"
    cfg_run_per_layer_attn_isolated = copy.deepcopy(original_main_config)
    if cfg_run_per_layer_attn_isolated.num_layers >= 2:
        cfg_run_per_layer_attn_isolated.attention_type_per_layer = ["markovian"] * (cfg_run_per_layer_attn_isolated.num_layers -1) + ["qi_rff"]
    elif cfg_run_per_layer_attn_isolated.num_layers == 1:
        cfg_run_per_layer_attn_isolated.attention_type_per_layer = ["qi_rff"] # All layers (1 layer) is qi_rff

    cfg_run_adapters_isolated = copy.deepcopy(original_main_config); cfg_run_adapters_isolated.use_adapters = True
    cfg_run_sparsity_isolated = copy.deepcopy(original_main_config); cfg_run_sparsity_isolated.sparsity_regularization_coeff = 1e-4
    
    test_configs_isolated = {
        "baseline_isolated": original_main_config, 
        "qi_rff_attention_isolated": cfg_run_qi_isolated,
        "multimodal_sequential_isolated": cfg_run_mm_seq_isolated,
        "multimodal_interleaved_isolated": cfg_run_mm_int_isolated,
        "hybrid_split_heads_isolated": cfg_run_hybrid_split_isolated, # This will now use the refactored logic
        "hybrid_parallel_sum_isolated": cfg_run_hybrid_parallel_isolated,
        "per_layer_attention_isolated": cfg_run_per_layer_attn_isolated,
        "with_adapters_isolated": cfg_run_adapters_isolated,
        "with_sparsity_isolated": cfg_run_sparsity_isolated,
    }

    for name, cfg_run_iso in test_configs_isolated.items():
        print(f"\n--- Testing Isolated Configuration: {name} (Refactored Hybrid) ---")
        # layer_max_orders are auto-filled by config's __post_init__

        model_iso = MarkovianTransformer(cfg_run_iso).to(device)
        ids_iso = torch.randint(0,cfg_run_iso.vocab_size,(B,N_s_main),device=device,dtype=torch.long) if N_s_main > 0 else torch.empty(B,0,dtype=torch.long, device=device)
        img_feats_iso, img_pad_mask_iso = None, None
        if cfg_run_iso.is_multimodal:
            num_patches_iso = cfg_run_iso.num_image_patches if N_s_main == 0 and cfg_run_iso.num_image_patches > 0 else min(N_s_main if N_s_main > 0 else float('inf'), cfg_run_iso.num_image_patches)
            num_patches_iso = max(0, num_patches_iso) # Ensure non-negative
            if num_patches_iso > 0:
                img_feats_iso = torch.randn(B, num_patches_iso, cfg_run_iso.image_feature_dim, device=device)
                img_pad_mask_iso = torch.zeros(B, img_feats_iso.shape[1], dtype=torch.bool, device=device)
                if img_feats_iso.shape[1] > 1 : img_pad_mask_iso[:, -img_feats_iso.shape[1]//2:] = True
            else:
                img_feats_iso = torch.empty(B, 0, cfg_run_iso.image_feature_dim, device=device)
                img_pad_mask_iso = torch.empty(B, 0, dtype=torch.bool, device=device)
        try:
            model_iso.eval()
            with torch.no_grad():
                l_reg_iso = model_iso(ids_iso.clone(), image_features=img_feats_iso.clone() if img_feats_iso is not None else None, image_padding_mask=img_pad_mask_iso.clone() if img_pad_mask_iso is not None else None)
            print(f"Logits (regular, {name}) shape: {l_reg_iso.shape}"); assert l_reg_iso.shape==(B,N_s_main,cfg_run_iso.num_output_classes)
            
            ids_empty_iso = torch.empty(B,0,dtype=torch.long, device=device)
            img_feats_empty_N_iso, img_pad_mask_empty_N_iso = None, None
            if cfg_run_iso.is_multimodal:
                num_img_patches_empty_text_iso = cfg_run_iso.num_image_patches
                if num_img_patches_empty_text_iso > 0:
                     img_feats_empty_N_iso = torch.randn(B, num_img_patches_empty_text_iso, cfg_run_iso.image_feature_dim, device=device)
                     img_pad_mask_empty_N_iso = torch.zeros(B, num_img_patches_empty_text_iso, dtype=torch.bool, device=device)
                else:
                     img_feats_empty_N_iso = torch.empty(B,0,cfg_run_iso.image_feature_dim,device=device)
                     img_pad_mask_empty_N_iso = torch.empty(B,0,dtype=torch.bool, device=device)
            with torch.no_grad():
                l_empty_iso = model_iso(ids_empty_iso.clone(), image_features=img_feats_empty_N_iso, image_padding_mask=img_pad_mask_empty_N_iso)
            print(f"Logits (empty_seq, {name}) shape: {l_empty_iso.shape}"); assert l_empty_iso.shape==(B,0,cfg_run_iso.num_output_classes)
            print(f"Forward pass for '{name}': PASSED")

            if any(p.requires_grad for p in model_iso.parameters()):
                model_iso.train()
                optimizer_iso = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_iso.parameters()), lr=1e-4)
                optimizer_iso.zero_grad()
                train_targets_iso = ids_iso.clone()
                ignore_idx = -100 
                if N_s_main > 0 and B > 0: train_targets_iso[0, N_s_main//2:] = ignore_idx 
                logits_train_iso = model_iso(ids_iso, image_features=img_feats_iso, image_padding_mask=img_pad_mask_iso, is_causal=True)
                loss_main_val_iso = torch.tensor(0.0, device=device, dtype=logits_train_iso.dtype)
                if N_s_main > 0 and logits_train_iso.numel() > 0 and (train_targets_iso.view(-1) != ignore_idx).any():
                     loss_main_val_iso = F.cross_entropy(logits_train_iso.reshape(-1, cfg_run_iso.num_output_classes), train_targets_iso.reshape(-1), ignore_index=ignore_idx)
                loss_sparsity_val_iso = model_iso.get_sparsity_loss()
                total_loss_val_iso = loss_main_val_iso + loss_sparsity_val_iso
                if total_loss_val_iso.requires_grad and total_loss_val_iso.isfinite() : total_loss_val_iso.backward()
                optimizer_iso.step()
                print(f"Training step for '{name}': PASSED (Loss: {total_loss_val_iso.item():.3f})")
            else:
                print(f"Training step for '{name}': SKIPPED (no trainable parameters)")
        except Exception as e:
            print(f"Test for '{name}': FAILED - {type(e).__name__}: {e}")
            traceback.print_exc(limit=3) # Increased limit
            if PYTEST_AVAILABLE: pytest.fail(f"Test {name} (refactored hybrid) failed: {e}")
    

    print("\n--- Running Specific Unit Tests (Refactored Hybrid, using simplified base_test_cfg) ---")
    base_test_cfg_dict = dict(
        vocab_size=50, num_output_classes=50, embed_size=32, 
        num_layers=1, heads=2, max_order=1, layer_max_orders=None, 
        dropout_rate=0.0, 
        max_seq_len=16,
        default_attention_type="markovian", # Base for unit tests unless overridden
        use_adapters=False, is_multimodal=False, 
        sparsity_regularization_coeff=0.0, order_gate_variant="mlp", compile_model=False,
        image_feature_dim=16, num_image_patches=4, num_cross_attention_layers=1,
        qi_attention_rff_dim_ratio=0.5, 
        hybrid_attention_markov_head_ratio=0.5, # For tests that might use hybrid
        activation_checkpointing_reentrant=False, 
    )
    base_test_cfg = MarkovianTransformerConfig(**base_test_cfg_dict) 
    
    test_activation_checkpointing(copy.deepcopy(base_test_cfg), device)
    
    cfg_tying = copy.deepcopy(base_test_cfg)
    test_weight_tying(cfg_tying, device)
    
    test_gru_influence(copy.deepcopy(base_test_cfg), device)
    test_markovian_bias_fallback(copy.deepcopy(base_test_cfg), device)

    print("-"*100+"\nExample run finished (Refactored Hybrid).")