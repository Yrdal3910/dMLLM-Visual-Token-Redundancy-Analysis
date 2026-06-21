"""
Fast dLLM Cache Hook for LLaDA Model

This module contains hooks and utilities for implementing fast distributed LLM caching
in the LLaDA model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Sequence, Union
import numpy as np

import pdb
import math
from collections import deque
FAST_SAMPLING = False

class FastDLLMGenerationHook:
    """
    Hook class for implementing fast dLLM caching functionality in LLaDA model.
    This class handles both attention-level caching and generation-level optimizations.
    """
    
    def __init__(self, model):
        self.model = model
        self.original_methods = {}
        self.is_registered = False
    
    def register_hooks(self):
        """Register fast dLLM hooks to the model."""
        if self.is_registered:
            return
            
        # Store original methods
        self.original_methods['generate'] = self.model.generate  # 新增：保存原始的generate方法
        
        # Store original attention forwards
        for layer_idx, layer in enumerate(self.model.model.layers):
            self.original_methods[f'attention_{layer_idx}'] = layer.self_attn.forward
            # Replace attention forward with fast cache version
            layer.self_attn.forward = self._create_fast_attention_forward(layer.self_attn, layer_idx)
        
        self.model.generate = self._fast_generate  # 新增：替换generate方法
        
        self.is_registered = True
    
    def unregister_hooks(self):
        """Unregister fast dLLM hooks from the model."""
        if not self.is_registered:
            return
            
        # Restore original methods
        self.model.generate = self.original_methods['generate']  # 新增：恢复原始的generate方法
        
        # Restore original attention forwards
        for layer_idx, layer in enumerate(self.model.model.layers):
            layer.self_attn.forward = self.original_methods[f'attention_{layer_idx}']
        
        self.is_registered = False
    
    def _create_fast_attention_forward(self, attention_layer, layer_idx, system_prompt_len: Optional[torch.LongTensor] = None, visual_seq_len: Optional[torch.LongTensor] = None):
        """Create fast attention forward method with caching support."""
        def fast_attention_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            system_prompt_len: Optional[torch.LongTensor] = None,
            visual_seq_len: Optional[torch.LongTensor] = None,
            fast_dllm_cache: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            
            # If not using fast cache or output_attentions is needed, use original method
            if output_attentions:
                return self.original_methods[f'attention_{layer_idx}'](
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs
                )
            
            bsz, q_len, _ = hidden_states.size()

            query_states = attention_layer.q_proj(hidden_states)
            key_states = attention_layer.k_proj(hidden_states)
            value_states = attention_layer.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attention_layer.num_heads, attention_layer.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attention_layer.num_key_value_heads, attention_layer.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attention_layer.num_key_value_heads, attention_layer.head_dim).transpose(1, 2)

            # Apply rotary position embedding with fast cache consideration
            cache_offset = 0
            if fast_dllm_cache and len(fast_dllm_cache) > layer_idx:
                cache_offset = fast_dllm_cache[layer_idx][0].shape[-2]
            
            cos, sin = attention_layer.rotary_emb(value_states, position_ids + cache_offset if cache_offset > 0 else position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Handle past key values
            past_key_value = getattr(attention_layer, "past_key_value", past_key_value)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
            
            # Fast dLLM cache logic
            if fast_dllm_cache is not None:
                if len(fast_dllm_cache) <= layer_idx:
                    fast_dllm_cache.append((key_states, value_states))
                else:
                    past_key, past_value = fast_dllm_cache[layer_idx]
                    key_states = torch.cat([past_key, key_states], dim=-2)
                    value_states = torch.cat([past_value, value_states], dim=-2)

            # Repeat key-value pairs for multi-head attention
            key_states = self._repeat_kv(key_states, attention_layer.num_key_value_groups)
            value_states = self._repeat_kv(value_states, attention_layer.num_key_value_groups)

            # Apply causal mask
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            # Ensure contiguous tensors for CUDA
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # Scaled dot-product attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                is_causal=False,
                dropout_p=attention_layer.attention_dropout if attention_layer.training else 0.0,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, attention_layer.hidden_size)
            attn_output = attention_layer.o_proj(attn_output)

            return attn_output, None, past_key_value
        
        return fast_attention_forward
    
    @torch.no_grad()
    def _fast_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        system_prompt_len: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        #自己改的
        image_token_length = torch.tensor(0, dtype=torch.int)
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, image_token_length) = self.model.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.model.get_model().embed_tokens(inputs)
        output = self._fast_generate_with_embeds(inputs_embeds=inputs_embeds, system_prompt_len=system_prompt_len, image_token_length=image_token_length, **kwargs)
        return output
    
    @torch.no_grad()
    def _fast_generate_with_embeds(
        self, 
        inputs_embeds, 
        steps=128, 
        gen_length=128, 
        block_length=128, 
        temperature=0.0,
        cfg_scale=0.0, 
        remasking='low_confidence', 
        mask_id=126336, 
        tokenizer=None, 
        stopping_criteria=None, 
        generation_suffix=None, 
        threshold=None, 
        prefix_refresh_interval=32, 
        system_prompt_len: Optional[torch.Tensor] = None, 
        image_token_length: Optional[torch.Tensor] = None, 
        **kwargs
    ):
        """
        Fast generation with embeddings using dLLM cache optimization.
        This method incorporates all fast dLLM related optimizations.
        """
        # Use mixed precision for faster computation
        with torch.cuda.amp.autocast(enabled=True):
            # Handle generation suffix
            suffix_embeds = None
            suffix_token_ids = None
            suffix_len = 0
            if generation_suffix is not None and tokenizer is not None and len(generation_suffix) > 0:
                suffix_token_ids = tokenizer.encode(generation_suffix, add_special_tokens=False)
                suffix_token_ids = torch.tensor(suffix_token_ids, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
                suffix_embeds = self.model.model.embed_tokens(suffix_token_ids)
                suffix_len = suffix_embeds.shape[1]

            # Create input in embedding space
            total_length = inputs_embeds.shape[1] + gen_length + suffix_len
            masked_embed = self.model.model.embed_tokens(torch.tensor([mask_id]).to(inputs_embeds.device))
            x_embeds = masked_embed.repeat(1, total_length, 1).to(inputs_embeds.device)
            x_embeds[:, :inputs_embeds.shape[1]] = inputs_embeds.clone()
            if suffix_embeds is not None:
                x_embeds[:, -suffix_len:] = suffix_embeds

            # Create tracking tensor for token IDs
            x = torch.full((1, total_length), mask_id, dtype=torch.long, device=inputs_embeds.device)
            if suffix_token_ids is not None:
                x[:, -suffix_len:] = suffix_token_ids

            # Prompt index tracking
            prompt_index = torch.zeros((1, total_length), dtype=torch.bool, device=inputs_embeds.device)
            prompt_index[:, :inputs_embeds.shape[1]] = 1

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            assert steps % num_blocks == 0
            steps = steps // num_blocks

            # Initialize stop tracking
            stop_position = inputs_embeds.shape[1] + gen_length
            found_stop_seq = False
            stop_tokens = []
            
            if stopping_criteria is not None:
                assert tokenizer is not None, "tokenizer is required when stopping_criteria is not None"
                for stop_str in stopping_criteria:
                    tokens = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_tokens.append(tokens)

            # Process each block
            if FAST_SAMPLING:
                # -------- 基于新逻辑：忽略 block_length，但兼容 fast_dllm_cache --------
                device = inputs_embeds.device
                B = 1
                S_total = x.shape[1]
                assert x.shape[0] == 1, "Currently batch_size is restricted to 1"

                # 还原被“分块逻辑”修改过的 steps，并完全忽略 block_length
                num_blocks_restored = max(1, gen_length // block_length)
                total_steps = steps * num_blocks_restored  # 恢复调用方传入的原始 steps

                # ---- 超参 ----
                lambda_decay = 0.25
                var_steps = 2
                var_thresh = 1.0
                conf_thresh = 0.85

                # ---- 生成段起止 ----
                gen_start = inputs_embeds.shape[1]
                gen_end   = gen_start + gen_length

                # ---- 每步名义解码上限（用于保守解码 & 进度对齐）----
                assert gen_length % total_steps == 0, "gen_length should be divisible by total steps"
                avg_per_step = math.ceil(gen_length / total_steps)
                k_pre = avg_per_step

                # ---- 状态 ----
                decoded_total = 0
                finalized_mask = torch.zeros_like(x, dtype=torch.bool, device=device)  # 只统计生成段的“已落锤”
                p_hist = deque(maxlen=var_steps)  # 存最近 var_steps 帧的 x0_p

                # 生成段内位置偏移 i_j（0-based），用于 e^{-λ i}
                i_offset = torch.zeros((B, S_total), device=device, dtype=torch.float32)
                offs = torch.arange(0, gen_length, device=device, dtype=torch.float32)
                i_offset[:, gen_start:gen_end] = offs

                # 与原缓存策略对齐：刷新时跑全长并将 cache 切到 gen_start；否则仅从 gen_start 增量前推
                fast_dllm_cache = []

                actual_steps = 0
                for t in range(1, total_steps + 1):
                    actual_steps = t

                    # 判定是否处于“增量前推”模式（非CFG且非刷新步）
                    tail_mode = (cfg_scale <= 0.0) and ((t - 1) % prefix_refresh_interval != 0)
                    pos_start = gen_start if tail_mode else 0  # 局部切片在全长里的起点

                    # 1) 前向：CFG 分支跑全长；非 CFG 分支用 fast_dllm_cache（刷新=全长，增量=尾部）
                    if cfg_scale > 0.0:
                        un_embeds = x_embeds.clone()
                        un_mask = prompt_index.unsqueeze(-1).expand_as(x_embeds)
                        un_embeds[un_mask] = masked_embed.repeat(x_embeds.shape[0], x_embeds.shape[1], 1)[un_mask]
                        combined_embeds = torch.cat([x_embeds, un_embeds], dim=0)

                        outputs = self.model.model(
                            inputs_embeds=combined_embeds,
                            system_prompt_len=system_prompt_len,
                            image_token_length=image_token_length
                        )
                        logits_raw_all = self.model.lm_head(outputs[0]).float()  # (2, S_total, V)
                        logits_main, logits_un = torch.chunk(logits_raw_all, 2, dim=0)
                        logits_raw = logits_un + (cfg_scale + 1.0) * (logits_main - logits_un)  # (1, S_total, V)
                    else:
                        if (t - 1) % prefix_refresh_interval == 0:
                            fast_dllm_cache = []
                            outputs = self.model.model(
                                inputs_embeds=x_embeds,
                                fast_dllm_cache=fast_dllm_cache,
                                system_prompt_len=system_prompt_len,
                                image_token_length=image_token_length
                            )
                            # 切 cache 到 gen_start（忽略 block）
                            fast_dllm_cache = self._create_cache_slice(fast_dllm_cache, gen_start)
                        else:
                            outputs = self.model.model(
                                inputs_embeds=x_embeds[:, gen_start:],
                                fast_dllm_cache=fast_dllm_cache,
                                system_prompt_len=system_prompt_len,
                                image_token_length=image_token_length
                            )
                        logits_raw = self.model.lm_head(outputs[0]).float()  # (1, S_slice, V)

                    # ----- 统一局部/全长视角 -----
                    S_total = x.shape[1]
                    S_slice = logits_raw.shape[1]               # 本次前向覆盖的长度
                    mask_index_full = torch.all(torch.abs(x_embeds - masked_embed) < 1e-5, dim=2)  # (1, S_total)
                    mask_index_local = mask_index_full[:, pos_start:pos_start + S_slice]           # (1, S_slice)

                    # 禁用特殊 token：只在局部切片上做
                    forbidden_tokens = [126081, 126080, 126346, 126347]
                    for token_id in forbidden_tokens:
                        logits_raw[:, :, token_id] = torch.where(
                            mask_index_local, torch.tensor(-float('inf'), device=logits_raw.device), logits_raw[:, :, token_id]
                        )

                    # 采样 & 置信度（局部）
                    logits_with_noise = self._add_gumbel_noise(logits_raw, temperature=temperature)  # (1, S_slice, V)
                    x0_local = torch.argmax(logits_with_noise, dim=-1)                               # (1, S_slice)

                    p_local = torch.softmax(logits_raw.to(torch.float64), dim=-1)                    # (1, S_slice, V)
                    x0_p_local = torch.gather(p_local, dim=-1, index=x0_local.unsqueeze(-1)).squeeze(-1).to(torch.float32)  # (1,S_slice)

                    # 将局部量拼回全长（便于统一做“方差/衰减/进度对齐”）
                    x0_p_full = torch.full((1, S_total), -float('inf'), device=x.device, dtype=torch.float32)
                    x0_p_full[:, pos_start:pos_start + S_slice] = x0_p_local

                    top1_raw_local = torch.gather(logits_raw, dim=-1, index=x0_local.unsqueeze(-1)).squeeze(-1).to(torch.float32)  # (1,S_slice)
                    top1_raw_full = torch.full((1, S_total), -float('inf'), device=x.device, dtype=torch.float32)
                    top1_raw_full[:, pos_start:pos_start + S_slice] = top1_raw_local

                    # 2) allow_mask：构造“全长”允许写回的掩码
                    allow_mask_full = torch.zeros_like(mask_index_full)
                    # 仅生成段
                    gen_mask = torch.zeros_like(allow_mask_full)
                    gen_mask[:, gen_start:gen_end] = True
                    allow_mask_full = mask_index_full & gen_mask
                    # suffix 禁止覆盖
                    if suffix_len > 0:
                        allow_mask_full[:, -suffix_len:] = False
                    # stop 后禁止
                    if found_stop_seq:
                        allow_mask_full[:, stop_position:] = False
                    # 未覆盖到的局部区域也不允许（把本次没算 logits 的位置禁掉）
                    covered_mask = torch.zeros_like(allow_mask_full)
                    covered_mask[:, pos_start:pos_start + S_slice] = True
                    allow_mask_full = allow_mask_full & covered_mask

                    if not allow_mask_full.any():
                        break

                    # 3) 更新置信度历史（全长）
                    p_hist.append(x0_p_full.detach())

                    # 4) 衰减打分：用“全长”做，避免形状不一致
                    score_adj = top1_raw_full * torch.exp(-lambda_decay * i_offset)       # (1, S_total)
                    score_adj_masked = torch.where(allow_mask_full, score_adj, torch.tensor(-float('inf'), device=x.device))

                    # 5) 选择待解码集合（全长 to_decode_mask_full）
                    to_decode_mask_full = torch.zeros_like(allow_mask_full)
                    if t <= var_steps:
                        remain = int(allow_mask_full.sum().item())
                        take_k = min(k_pre, remain)
                        if take_k > 0:
                            _, topk_idx = torch.topk(score_adj_masked, k=take_k, dim=1)
                            to_decode_mask_full.scatter_(1, topk_idx, True)
                    else:
                        if len(p_hist) == var_steps:
                            stack_p = torch.stack(list(p_hist), dim=0)  # (var_steps, 1, S_total)
                            var_map = stack_p.var(dim=0, unbiased=False).to(torch.float32)  # (1, S_total)
                            stable = (var_map < var_thresh) & (x0_p_full >= conf_thresh) & allow_mask_full
                            to_decode_mask_full = stable.clone()

                    # 6) 进度对齐（全长）
                    required = min(gen_length, avg_per_step * t)
                    decoded_total = int(finalized_mask[:, gen_start:gen_end].sum().item())
                    current_pick = int(to_decode_mask_full.sum().item())
                    need = required - decoded_total - current_pick
                    if need > 0:
                        remain_mask = allow_mask_full & (~to_decode_mask_full)
                        if remain_mask.any():
                            score_adj_remain = torch.where(remain_mask, score_adj, torch.tensor(-float('inf'), device=x.device))
                            take_k = min(need, int(remain_mask.sum().item()))
                            if take_k > 0:
                                _, topk_idx = torch.topk(score_adj_remain, k=take_k, dim=1)
                                to_decode_mask_full.scatter_(1, topk_idx, True)

                    # 7) 写回：只对“局部覆盖且被选中”的位置写回
                    if to_decode_mask_full.any():
                        # 先构造全长的 x0_full / x0_embeds_full，避免 shape 不一致
                        x0_full = x.clone()
                        # 仅在局部切片内把候选 token 放进去（非 mask 位保持原 x）
                        x0_local_applied = torch.where(
                            mask_index_local, x0_local, x[:, pos_start:pos_start + S_slice]
                        )
                        x0_full[:, pos_start:pos_start + S_slice] = x0_local_applied

                        # 准备全长的候选嵌入（仅替换局部切片的 mask 位）
                        x0_local_embeds = self.model.model.embed_tokens(x0_local)  # (1, S_slice, d)
                        local_filled = torch.where(
                            mask_index_local.unsqueeze(-1).expand(-1, -1, x_embeds.size(-1)),
                            x0_local_embeds,
                            x_embeds[:, pos_start:pos_start + S_slice]
                        )
                        x0_embeds_full = x_embeds.clone()
                        x0_embeds_full[:, pos_start:pos_start + S_slice] = local_filled

                        # 依据全长 to_decode_mask_full 落锤
                        x_embeds[to_decode_mask_full] = x0_embeds_full[to_decode_mask_full]
                        x[to_decode_mask_full] = x0_full[to_decode_mask_full]
                        finalized_mask |= to_decode_mask_full

                    # 8) stop_tokens 检查（保持与原代码一致，基于生成段全长）
                    if stopping_criteria is not None:
                        generated_part = x[0, gen_start:gen_end]
                        earliest_pos = None
                        for stop_seq in stop_tokens:
                            seq = stop_seq if isinstance(stop_seq, list) else [stop_seq]
                            L = len(seq)
                            if L == 0 or L > generated_part.size(0):
                                continue
                            for start_idx in range(generated_part.size(0) - L + 1):
                                if torch.all(generated_part[start_idx:start_idx + L] ==
                                            torch.tensor(seq, device=x.device, dtype=generated_part.dtype)):
                                    pos_abs = gen_start + start_idx
                                    if earliest_pos is None or pos_abs < earliest_pos:
                                        earliest_pos = pos_abs
                                    break
                        if earliest_pos is not None:
                            if (not found_stop_seq) or (earliest_pos < stop_position):
                                found_stop_seq = True
                                stop_position = earliest_pos

                    # 若发现 stop，则若 stop 之前不再有 mask，提前结束
                    if found_stop_seq:
                        mask_index_now = torch.all(torch.abs(x_embeds - masked_embed) < 1e-5, dim=2)
                        pre_stop_masks = mask_index_now[0, gen_start:stop_position]
                        if not pre_stop_masks.any():
                            break

                print(f"actual_steps with fast-sampling and fast-dllm is: {actual_steps}")
                
            else:
                for num_block in range(num_blocks):
                    block_start = inputs_embeds.shape[1] + num_block * block_length
                    block_end = inputs_embeds.shape[1] + (num_block + 1) * block_length

                    # Skip if stop found before current block
                    if found_stop_seq and stop_position <= block_start:
                        break
                    
                    block_embeds = x_embeds[:, block_start:block_end]
                    block_mask_index = torch.all(torch.abs(block_embeds - masked_embed) < 1e-5, dim=2)
                    num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps)
                    
                    
                    i = 0

                    while True:
                        if threshold is None and i >= steps:
                            break
                        
                        # Check mask state
                        mask_index = torch.all(torch.abs(x_embeds - masked_embed) < 1e-5, dim=2)
                        
                        if found_stop_seq:
                            pre_stop_masks = mask_index[0, inputs_embeds.shape[1]:stop_position]
                            if not pre_stop_masks.any():
                                break
                        
                        current_block_masks = mask_index[0, block_start:block_end]
                        if not current_block_masks.any():
                            break
                        
                        # Handle CFG
                        if cfg_scale > 0.0:
                            un_embeds = x_embeds.clone()
                            un_mask = prompt_index.unsqueeze(-1).expand_as(x_embeds)
                            un_embeds[un_mask] = masked_embed.repeat(x_embeds.shape[0], x_embeds.shape[1], 1)[un_mask]
                            combined_embeds = torch.cat([x_embeds, un_embeds], dim=0)
                            
                            outputs = self.model.model(
                                inputs_embeds=combined_embeds,
                                fast_dllm_cache=fast_dllm_cache, 
                                system_prompt_len=system_prompt_len, 
                                image_token_length=image_token_length,
                            )
                            logits = self.model.lm_head(outputs[0]).float()
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            if i % prefix_refresh_interval == 0:
                                fast_dllm_cache = []
                                outputs = self.model.model(
                                    inputs_embeds=x_embeds,
                                    fast_dllm_cache=fast_dllm_cache, 
                                    system_prompt_len=system_prompt_len, 
                                    image_token_length=image_token_length,
                                )
                                # Slice cache to block start
                                fast_dllm_cache = self._create_cache_slice(fast_dllm_cache, block_start)
                            else:
                                # Incremental forward pass
                                outputs = self.model.model(
                                    inputs_embeds=x_embeds[:, block_start:],
                                    fast_dllm_cache=fast_dllm_cache, 
                                    system_prompt_len=system_prompt_len, 
                                    image_token_length=image_token_length,
                                )
                            logits = self.model.lm_head(outputs[0]).float()

                        # Filter forbidden tokens
                        forbidden_tokens = [126081, 126080, 126346, 126347]
                        if i % prefix_refresh_interval == 0:
                            for token_id in forbidden_tokens:
                                logits[:, :, token_id] = torch.where(mask_index, -float('inf'), logits[:, :, token_id])
                        else:
                            for token_id in forbidden_tokens:
                                logits[:, :, token_id] = torch.where(mask_index[:, block_start:], -float('inf'), logits[:, :, token_id])

                        # Get transfer indices and update
                        if i % prefix_refresh_interval == 0:
                            x0, transfer_index = self._get_transfer_index(
                                logits, temperature, remasking, mask_index, x,
                                num_transfer_tokens[:, i] if threshold is None else None,
                                found_stop_seq, stop_position, block_end, suffix_len, threshold
                            )
                            x0_embeds = self.model.model.embed_tokens(x0)
                            x0_embeds = torch.where(mask_index.unsqueeze(-1).expand_as(x_embeds), x0_embeds, x_embeds)
                            x_embeds[transfer_index] = x0_embeds[transfer_index]
                            x[transfer_index] = x0[transfer_index]
                        else:
                            x0, transfer_index = self._get_transfer_index(
                                logits, temperature, remasking, mask_index[:, block_start:], x[:, block_start:],
                                num_transfer_tokens[:, i] if threshold is None else None,
                                found_stop_seq, stop_position - block_start, block_end - block_start, suffix_len, threshold
                            )
                            x0_embeds = self.model.model.embed_tokens(x0)
                            x0_embeds = torch.where(mask_index[:, block_start:].unsqueeze(-1).expand_as(x_embeds[:, block_start:]), 
                                                x0_embeds, x_embeds[:, block_start:])
                            x_embeds[:, block_start:][transfer_index] = x0_embeds[transfer_index]
                            x[:, block_start:][transfer_index] = x0[transfer_index]

                        # Check for stop words
                        if stopping_criteria is not None:
                            generated_part = x[0, inputs_embeds.shape[1]:inputs_embeds.shape[1] + gen_length]
                            for stop_seq in stop_tokens:
                                if not isinstance(stop_seq, list):
                                    stop_seq = [stop_seq]
                                for start_idx in range(generated_part.size(0) - len(stop_seq) + 1):
                                    if torch.all(generated_part[start_idx:start_idx + len(stop_seq)] == torch.tensor(stop_seq, device=x.device)):
                                        current_position = inputs_embeds.shape[1] + start_idx
                                        if not found_stop_seq or current_position < stop_position:
                                            stop_position = current_position
                                            found_stop_seq = True
                                        break
                                if found_stop_seq:
                                    break
                        i += 1

                    if threshold is not None:
                        print(f'Number of steps: {i}')

            # Return results
            if found_stop_seq:
                if suffix_len > 0:
                    return torch.cat([x[:, inputs_embeds.shape[1]:stop_position], x[:, -suffix_len:]], dim=1)
                else:
                    return x[:, inputs_embeds.shape[1]:stop_position]
            else:
                if suffix_len > 0:
                    return torch.cat([x[:, inputs_embeds.shape[1]:inputs_embeds.shape[1] + gen_length], x[:, -suffix_len:]], dim=1)
                else:
                    return x[:, inputs_embeds.shape[1]:inputs_embeds.shape[1] + gen_length]

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Apply rotary position embedding to query and key tensors."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (FastDLLMGenerationHook._rotate_half(q) * sin)
        k_embed = (k * cos) + (FastDLLMGenerationHook._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key-value pairs for multi-head attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _get_transfer_index(self, logits, temperature, remasking, mask_index, x, num_transfer_tokens, 
                          found_stop_seq, stop_position, block_end, suffix_len, threshold=None):
        """Get transfer indices for token updates during generation."""
        logits_with_noise = self._add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == 'low_confidence':
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)
        
        # Handle stop sequences and block boundaries
        if found_stop_seq:
            x0_p[:, stop_position:] = -np.inf
        else:
            x0_p[:, block_end:] = -np.inf

        # Prevent overwriting suffix
        if suffix_len > 0:
            x0_p[:, -suffix_len:] = -np.inf
        
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        
        # pdb.set_trace()
        if threshold is not None:
            num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

        for j in range(confidence.shape[0]):
            if threshold is None:
                top_i = num_transfer_tokens[j]
            else:
                ns = list(range(1, num_transfer_tokens[j] + 1))
                es = [threshold / (n + 1) for n in ns]
                threshs = [1 - e for e in es]
                threshs[0] = -1  # at least one token is transferred
                
                sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1, descending=True)[0]
                assert len(sorted_confidence) == len(threshs)
                
                for top_i in range(len(threshs)):
                    if sorted_confidence[top_i] < threshs[top_i]:
                        break

                if top_i == 0 or top_i == len(threshs) - 1:
                    top_i += 1

            _, select_index = torch.topk(confidence[j], k=top_i)
            transfer_index[j, select_index] = True

        return x0, transfer_index

    @staticmethod
    def _add_gumbel_noise(logits, temperature):
        """Add Gumbel noise for categorical sampling."""
        if temperature == 0:
            return logits
        
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(mask_index, steps):
        """Precompute the number of tokens to transition at each step."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = base.expand(-1, steps).clone()

        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)

    def _create_cache_slice(self, fast_dllm_cache, block_start):
        """Create a sliced version of fast_dllm_cache for block processing."""
        new_past_key_values = []
        for i in range(len(fast_dllm_cache)):
            new_past_key_values.append([])
            for j in range(len(fast_dllm_cache[i])):
                new_past_key_values[i].append(fast_dllm_cache[i][j][:, :, :block_start])
        return new_past_key_values


def register_fast_dllm_hook(model):
    """
    Register fast dLLM cache hooks to the model.
    
    Args:
        model: The LLaDA model to register hooks to
        
    Returns:
        FastDLLMGenerationHook: The hook instance for management
    """
    hook = FastDLLMGenerationHook(model)
    hook.register_hooks()
    return hook


def unregister_fast_dllm_hook(hook):
    """
    Unregister fast dLLM cache hooks from the model.
    
    Args:
        hook: The FastDLLMGenerationHook instance to unregister
    """
    hook.unregister_hooks() 