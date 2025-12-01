#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy
from llava.model.utils.packing import set_seqlens_in_batch
from llava.train.sequence_parallel.globals import get_pg_manager
from llava.utils.logging import logger

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel, position_transfer, reparam


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


# FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        images: Optional[torch.FloatTensor] = None,
        media_config: Optional[List] = None,
        variables: Optional[List[dict]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        packing: bool = True,
        force_packing: bool = False,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if images is not None:
            if media is not None:
                raise ValueError("Both 'media' and 'images' are provided. Please provide only one.")
            logger.warning("The 'images' argument is deprecated. Please use 'media' instead.")
            media = {"image": images}

        if media_config is None:
            media_config = defaultdict(dict)

        if inputs_embeds is None:
            # Check if we have LAPE data (variables field) to decide processing path
            has_lape_data = variables is not None and len(variables) > 0 and any(
                v is not None and isinstance(v, dict) and len(v) > 0 for v in variables
            )
            
            if has_lape_data:
                # LAPE processing path: use prepare_inputs_labels_for_multimodal_video
                print("[LAPE] Using prepare_inputs_labels_for_multimodal_video path (has variables)")
                # prepare_inputs returns: input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
                _, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal_video(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    media.get("image") if media else None,
                    modalities=["image"],  # LAPE主要处理图像
                    image_sizes=None,
                    variables=variables,
                )
            else:
                # Original VILA processing path: use _embed
                print("[LAPE] Using _embed path (no variables)")
                inputs_embeds, labels, attention_mask = self._embed(
                    input_ids,
                    media,
                    media_config,
                    labels,
                    attention_mask,
                    variables=variables,
                )

        if force_packing or (packing and self.training and not dpo_forward):
            if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
            set_seqlens_in_batch(seqlens_in_batch)

            (inputs_embeds, attention_mask, position_ids, labels) = self.repack_multimodal_data(
                inputs_embeds, attention_mask, position_ids, labels
            )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            **{k: v for k, v in kwargs.items() if k not in ["variables"]},
        )

        # LAPE soft-label supervision for OUTPUT tokens (align with llava_qwen)
        # Trigger when: training, labels provided, variables provided for all instances, and LAPE is initialized
        if (
            self.training
            and labels is not None
            and variables is not None
            and all(v is not None for v in variables)
            and getattr(self, "has_init_specific_embeddings", False)
        ):
            hidden_states = outputs.logits.new_tensor(0)  # placeholder to get device/dtype
            # Re-run backbone to fetch last hidden states (logits alone are insufficient for soft targets over extended heads)
            base_out = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                labels=None,
                output_hidden_states=True,
                use_cache=False,
            )
            if hasattr(base_out, "hidden_states") and base_out.hidden_states is not None:
                hidden_states = base_out.hidden_states[-1]
            else:
                hidden_states = None

            if hidden_states is not None:
                # Build extended output weight: [base_vocab; temporal_out; spatial_h_out; spatial_w_out]
                base_w = self.get_lm_head().weight  # [V, H]
                temp_w = reparam(self.temporal_output_embeddings.weight, self.temporal_reparam_mat)
                sh_w = reparam(self.spatial_height_output_embeddings.weight, self.spatial_height_reparam_mat)
                sw_w = reparam(self.spatial_width_output_embeddings.weight, self.spatial_width_reparam_mat)
                ext_w = torch.cat([
                    base_w.to(hidden_states.device, hidden_states.dtype),
                    temp_w.to(hidden_states.device, hidden_states.dtype),
                    sh_w.to(hidden_states.device, hidden_states.dtype),
                    sw_w.to(hidden_states.device, hidden_states.dtype),
                ], dim=0)

                logits_ext = torch.nn.functional.linear(hidden_states, ext_w)  # [B, T, V+extra]
                shift_logits = logits_ext[..., :-1, :].contiguous().view(-1, logits_ext.shape[-1])

                B = labels.shape[0]
                V = base_w.shape[0]
                T_out = getattr(self, "num_temporal_tokens", getattr(self.config, "num_temporal_tokens", 100))
                S_out = getattr(self, "num_spatial_tokens", getattr(self.config, "num_spatial_tokens", 100))
                vc = self.vision_config

                total_loss = 0.0
                total_count = 0

                for b in range(B):
                    cur_labels = labels[b]
                    t_idx = torch.where(cur_labels == vc.temporal_output_token_id)[0]
                    sh_idx = torch.where(cur_labels == vc.spatial_height_output_token_id)[0]
                    sw_idx = torch.where(cur_labels == vc.spatial_width_output_token_id)[0]

                    cur_vars = variables[b]
                    t_vals = cur_vars.get("temporal_output_locations", []) if cur_vars is not None else []
                    sh_vals = cur_vars.get("spatial_height_output_locations", []) if cur_vars is not None else []
                    sw_vals = cur_vars.get("spatial_width_output_locations", []) if cur_vars is not None else []

                    cur_logits = logits_ext[b:b+1]
                    cur_shift_logits = cur_logits[..., :-1, :].contiguous().view(-1, logits_ext.shape[-1])

                    cur_labels_shift = cur_labels[1:].contiguous()
                    N = cur_labels_shift.shape[0]
                    target = torch.zeros(N, V + T_out + 2 * S_out, device=cur_shift_logits.device, dtype=cur_shift_logits.dtype)
                    valid_mask = (cur_labels_shift != -100)
                    idxs = torch.where(valid_mask)[0]
                    if idxs.numel() > 0:
                        target[idxs, cur_labels_shift[idxs]] = 1.0

                    for i, pos in enumerate(t_idx):
                        if pos.item() == 0 or (pos.item() - 1) >= N:
                            continue
                        if i >= len(t_vals):
                            continue
                        floor_p, ceil_p, ratio = position_transfer(float(t_vals[i]), T_out)
                        si = pos.item() - 1
                        target[si, vc.temporal_output_token_id] = 0.0
                        target[si, V + floor_p] = 1.0 - ratio
                        if floor_p != ceil_p:
                            target[si, V + ceil_p] = ratio

                    for i, pos in enumerate(sh_idx):
                        if pos.item() == 0 or (pos.item() - 1) >= N:
                            continue
                        if i >= len(sh_vals):
                            continue
                        floor_p, ceil_p, ratio = position_transfer(float(sh_vals[i]), S_out)
                        si = pos.item() - 1
                        target[si, vc.spatial_height_output_token_id] = 0.0
                        base = V + T_out
                        target[si, base + floor_p] = 1.0 - ratio
                        if floor_p != ceil_p:
                            target[si, base + ceil_p] = ratio

                    for i, pos in enumerate(sw_idx):
                        if pos.item() == 0 or (pos.item() - 1) >= N:
                            continue
                        if i >= len(sw_vals):
                            continue
                        floor_p, ceil_p, ratio = position_transfer(float(sw_vals[i]), S_out)
                        si = pos.item() - 1
                        target[si, vc.spatial_width_output_token_id] = 0.0
                        base = V + T_out + S_out
                        target[si, base + floor_p] = 1.0 - ratio
                        if floor_p != ceil_p:
                            target[si, base + ceil_p] = ratio

                    logp = torch.nn.functional.log_softmax(cur_shift_logits, dim=-1)
                    loss_vec = -(target * logp).sum(dim=-1)
                    loss_vec = loss_vec[valid_mask[: loss_vec.shape[0]].to(loss_vec.device)]
                    if loss_vec.numel() > 0:
                        total_loss = total_loss + loss_vec.mean()
                        total_count += 1

                if total_count > 0:
                    outputs.loss = total_loss / total_count

        # Loss rescale for SP
        if get_pg_manager() is not None:
            loss_weight = calculate_loss_weight(labels)
            outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, labels

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        variables: Optional[list] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, CausalLMOutputWithPast]:
        """
        A compatibility wrapper for generation functions, similar to llava_qwen.
        It accepts `images` and `variables` and prepares `inputs_embeds` for the main generation loop.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported when using `generate` with `images`.")

        # If images are provided, we need to prepare the multimodal embeddings
        if images is not None:
            # The _embed method handles the conversion from input_ids + images to inputs_embeds
            inputs_embeds, labels, attention_mask = self._embed(
                input_ids=input_ids,
                media={"image": images},
                media_config=[{"image_sizes": image_sizes}] if image_sizes is not None else [{}],
                labels=None,  # No labels during inference
                attention_mask=attention_mask,
                variables=variables,
            )
            # We need to nullify input_ids as we are passing inputs_embeds
            input_ids = None
        else:
            # If no images, just get the token embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Call the original generate method from the parent class (PreTrainedModel)
        return super().generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
