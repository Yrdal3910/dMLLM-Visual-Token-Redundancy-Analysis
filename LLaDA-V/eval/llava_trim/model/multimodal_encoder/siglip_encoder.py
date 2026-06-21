"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoTokenizer, PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print


# Fraction of visual tokens discarded by TRIM. The remaining discarded
# features are represented by one appended mean token.
PRUNING_RATIO = 0.75


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

# class SigLipVisionTower(nn.Module):
#     def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
#         super().__init__()

#         self.is_loaded = False

#         self.config = SigLipVisionConfig()

#         self.vision_tower_name = vision_tower

#         self.image_processor = SigLipImageProcessor()

#         if not delay_load:
#             rank0_print(f"Loading vision tower: {vision_tower}")
#             self.load_model()
#         elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
#             # TODO: better detector is needed.
#             rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
#             self.load_model()
#         elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
#             rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
#             self.load_model()
#         else:
#             self.cfg_only = self.config

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
#             return

#         self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

#         del self.vision_tower.vision_model.encoder.layers[-1:]
#         self.vision_tower.vision_model.head = nn.Identity()
#         self.vision_tower.requires_grad_(False)

#         self.is_loaded = True

#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
#                 image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
#                 assert image_features.shape[-2] == 729
#                 image_features.append(image_feature)
#         else:
#             image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#             image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
#             assert image_features.shape[-2] == 729

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         for p in self.vision_tower.parameters():
#             return p.dtype

#     @property
#     def device(self):
#         for p in self.vision_tower.parameters():
#             return p.device

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def num_patches(self):
#         return (self.config.image_size // self.config.patch_size) ** 2

#     @property
#     def num_patches_per_side(self):
#         return self.config.image_size // self.config.patch_size
#         # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

#     @property
#     def image_size(self):
#         return self.config.image_size

class SigLipVisionTower(nn.Module):
    """
    SigLIP vision tower with configurable feature selection and optional
    text-guided TRIM token reduction.
    """

    def __init__(self, vision_tower, vision_tower_cfg=None, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = getattr(vision_tower_cfg, "mm_vision_select_layer", -1)
        self.select_feature = getattr(vision_tower_cfg, "mm_vision_select_feature", "patch")
        self.token_reduce_func = getattr(vision_tower_cfg, "mm_vision_token_reduce_func", None)
        self.text_side_path = (
            getattr(vision_tower_cfg, "text_side_path", None)
            or getattr(vision_tower_cfg, "mm_text_tower", None)
            or self.vision_tower_name
        )

        self.config = SigLipVisionConfig()
        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False) or (
            hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts
        ):
            rank0_print("Checkpoint indicates vision tower weights are present; loading.")
            self.load_model()
        else:
            self.cfg_only = self.config

    @property
    def _trim_enabled(self):
        return isinstance(self.token_reduce_func, str) and "TRIM" in self.token_reduce_func.upper()

    def _init_text_side_mandatory(self, device_map=None):
        """
        Load the tokenizer and text encoder required by TRIM.

        The text-side path falls back to the vision checkpoint, which should
        therefore point to a complete SigLIP checkpoint.
        """
        path = self.text_side_path
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the SigLIP tokenizer from {path}. "
                "Ensure the checkpoint contains its tokenizer files."
            ) from exc

        self.text_tower = None
        errors = []
        try:
            whole_model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=device_map,
                trust_remote_code=True,
            )
            self.text_tower = getattr(whole_model, "text_model", None)
            if self.text_tower is None and hasattr(whole_model, "get_text_model"):
                self.text_tower = whole_model.get_text_model()
        except Exception as exc:
            errors.append(f"full checkpoint: {exc}")

        if self.text_tower is None:
            try:
                self.text_tower = AutoModel.from_pretrained(
                    path,
                    subfolder="text_model",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    device_map=device_map,
                )
            except Exception as exc:
                errors.append(f"text_model subfolder: {exc}")

        if self.text_tower is None:
            details = "\n- ".join(errors)
            raise RuntimeError(
                f"Failed to load the SigLIP text encoder from {path}. "
                "Use a complete SigLIP checkpoint containing text-model weights."
                + (f"\n- {details}" if details else "")
            )

        self.text_tower.requires_grad_(False)
        self.text_tower.eval()

    def _encode_texts_chunked(self, texts, max_len=None):
        """
        Encode text in non-overlapping token windows and average the windows
        for each sample.
        """
        if isinstance(texts, str):
            texts = [texts]

        if max_len is None:
            cfg = getattr(self.text_tower, "config", None)
            max_len = getattr(cfg, "max_position_embeddings", 64)

        reserve = self.text_tokenizer.num_special_tokens_to_add(pair=False)
        chunk_body_len = max(1, max_len - reserve)
        text_device = next(self.text_tower.parameters()).device

        all_embeds = []
        for text in texts:
            ids = self.text_tokenizer(text, add_special_tokens=False)["input_ids"]
            chunks = [ids[i:i + chunk_body_len] for i in range(0, len(ids), chunk_body_len)] or [[]]

            chunk_embeds = []
            for chunk in chunks:
                chunk_with_specials = self.text_tokenizer.build_inputs_with_special_tokens(chunk)
                input_ids = torch.tensor([chunk_with_specials], dtype=torch.long, device=text_device)
                attention_mask = torch.ones_like(input_ids)

                pad_id = self.text_tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.text_tokenizer.eos_token_id
                if pad_id is None:
                    pad_id = 0
                if input_ids.shape[1] < max_len:
                    pad_len = max_len - input_ids.shape[1]
                    input_ids = F.pad(input_ids, (0, pad_len), value=pad_id)
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)
                elif input_ids.shape[1] > max_len:
                    input_ids = input_ids[:, :max_len]
                    attention_mask = attention_mask[:, :max_len]

                with torch.no_grad():
                    output = self.text_tower(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                    )

                if getattr(output, "text_embeds", None) is not None:
                    chunk_embeds.append(output.text_embeds)
                elif getattr(output, "pooler_output", None) is not None:
                    chunk_embeds.append(output.pooler_output)
                else:
                    pooled = output.last_hidden_state[:, 0]
                    if hasattr(self.text_tower, "text_projection"):
                        chunk_embeds.append(self.text_tower.text_projection(pooled))
                    else:
                        chunk_embeds.append(pooled)

            sample_embed = torch.stack(chunk_embeds, dim=0).mean(dim=0)
            all_embeds.append(sample_embed)

        return torch.cat(all_embeds, dim=0)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.config = self.vision_tower.config
        image_size = self.config.image_size
        self.image_processor = SigLipImageProcessor(
            image_mean=self.config.image_mean,
            size=(image_size, image_size),
            crop_size={"height": image_size, "width": image_size},
        )

        if hasattr(self.vision_tower, "vision_model") and hasattr(self.vision_tower.vision_model, "encoder"):
            layers = getattr(self.vision_tower.vision_model.encoder, "layers", None)
            if layers and len(layers) > 0:
                del self.vision_tower.vision_model.encoder.layers[-1:]

        if hasattr(self.vision_tower, "vision_model") and hasattr(self.vision_tower.vision_model, "head"):
            self.vision_tower.vision_model.head = nn.Identity()

        self.vision_tower.requires_grad_(False)

        self.post_layernorm = getattr(self.vision_tower.vision_model, "post_layernorm", None)
        if self.post_layernorm is None:
            self.post_layernorm = nn.Identity()

        self.visual_projection = nn.Identity()
        if self._trim_enabled:
            self._init_text_side_mandatory(device_map=device_map)
            vis_dim = self.config.hidden_size
            txt_cfg = getattr(self.text_tower, "config", None)
            txt_dim = getattr(txt_cfg, "projection_size", None) or getattr(txt_cfg, "hidden_size", None)
            if txt_dim is not None and vis_dim != txt_dim:
                raise RuntimeError(
                    f"TRIM requires aligned SigLIP image/text dimensions, got {vis_dim} and {txt_dim}. "
                    "A learned projection from the complete SigLIP model is required."
                )

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        all_image_features = image_forward_outs.hidden_states[self.select_layer]
        expected_patches = self.num_patches
        sequence_length = all_image_features.shape[1]
        has_cls = sequence_length == expected_patches + 1

        if self.select_feature == "patch":
            if has_cls:
                image_features = all_image_features[:, 1:]
            else:
                image_features = all_image_features
        elif self.select_feature == "cls_patch":
            image_features = all_image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        return image_features, all_image_features

    def token_reduction(self, image_features, all_image_features, text_features=None):
        """
        Apply TRIM and append one mean token representing discarded patches.
        """
        if not self.token_reduce_func:
            batch_size, token_count, _ = image_features.shape
            return image_features, [token_count] * batch_size
        if not self._trim_enabled:
            raise ValueError(f"Unknown token reduction function: {self.token_reduce_func}")
        if text_features is None:
            raise RuntimeError("TRIM requires text features. Pass texts to the vision tower forward method.")

        batch_size, tokens_number, _ = image_features.shape
        if tokens_number <= 1:
            return image_features, [tokens_number] * batch_size

        if not 0 < PRUNING_RATIO < 1:
            raise ValueError(
                f"PRUNING_RATIO must be between 0 and 1, got {PRUNING_RATIO}."
            )

        with torch.amp.autocast(device_type=image_features.device.type, enabled=False):
            if isinstance(self.post_layernorm, nn.LayerNorm) and self.post_layernorm.weight is not None:
                ln_dtype = self.post_layernorm.weight.dtype
            else:
                ln_dtype = torch.float32

            normed = self.post_layernorm(image_features.to(dtype=ln_dtype))
            if hasattr(self.visual_projection, "weight"):
                proj_dtype = self.visual_projection.weight.dtype
            else:
                proj_dtype = normed.dtype

            proj_image_features = self.visual_projection(normed.to(dtype=proj_dtype))
            text_in = text_features.to(
                device=proj_image_features.device,
                dtype=proj_image_features.dtype,
            )
            similarities = torch.matmul(proj_image_features, text_in.unsqueeze(2)).squeeze(2)

        similarities = -similarities
        similarities = F.softmax(similarities, dim=-1)

        num_tokens_to_keep = int(tokens_number * (1.0 - PRUNING_RATIO))
        num_tokens_to_keep = max(1, min(tokens_number - 1, num_tokens_to_keep))

        _, topk_indices = torch.topk(
            similarities,
            num_tokens_to_keep,
            dim=1,
            largest=True,
            sorted=False,
        )

        selected_image_features = torch.zeros_like(image_features)
        batch_indices = torch.arange(batch_size, device=image_features.device).unsqueeze(1)
        token_mask = torch.zeros(batch_size, tokens_number, device=image_features.device, dtype=torch.bool)
        token_mask[batch_indices, topk_indices] = True

        selected_image_features[:, :num_tokens_to_keep] = image_features[token_mask].view(batch_size, num_tokens_to_keep, -1)

        remaining_mask = torch.logical_not(token_mask)
        remaining_mean = (
            image_features * remaining_mask.unsqueeze(-1)
        ).sum(dim=1) / remaining_mask.sum(dim=1, keepdim=True)
        selected_image_features[:, num_tokens_to_keep] = remaining_mean

        actual_dims = [num_tokens_to_keep + 1] * batch_size
        return selected_image_features, actual_dims

    def _normalize_texts(self, texts, batch_size: int) -> list[str]:
        """
        Normalize supported text inputs to a list matching the image batch.

        Supported inputs:
        - str
        - list[str]
        - dict or list[dict] using common text keys
        - bytes or bytearray entries
        """

        def pick_from_dict(d):
            for k in ("text", "question", "prompt", "content", "context"):
                if k in d and d[k] is not None:
                    return str(d[k])
            return " ".join(str(v) for v in d.values() if v is not None)

        if texts is None:
            raise ValueError("TRIM requires text input, but texts is None.")

        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, dict):
            texts = [pick_from_dict(texts)]
        elif isinstance(texts, (list, tuple)):
            normalized = []
            for text in texts:
                if text is None:
                    normalized.append("")
                elif isinstance(text, str):
                    normalized.append(text)
                elif isinstance(text, dict):
                    normalized.append(pick_from_dict(text))
                elif isinstance(text, (bytes, bytearray)):
                    normalized.append(text.decode("utf-8", errors="ignore"))
                else:
                    raise TypeError(
                        f"Unsupported text element type: {type(text)}. Pass strings or dictionaries."
                    )
            texts = normalized
        else:
            raise TypeError(f"Unsupported texts type: {type(texts)}.")

        texts = [text.strip() for text in texts]

        if len(texts) == 1 and batch_size > 1:
            texts = texts * batch_size
        if len(texts) != batch_size:
            raise ValueError(
                f"Text count ({len(texts)}) does not match image batch size ({batch_size})."
            )

        return texts

    @torch.no_grad()
    def forward(self, images, texts=None):
        """
        Return image features and the valid token count for each sample.
        """
        text_features = None
        if self._trim_enabled:
            if isinstance(images, list):
                batch_size = len(images)
            else:
                batch_size = images.shape[0]
            texts = self._normalize_texts(texts, batch_size)
            text_features = self._encode_texts_chunked(texts)

        if isinstance(images, list):
            feats_list = []
            all_feats_list = []
            for image in images:
                outs = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True
                )
                image_features, all_image_features = self.feature_select(outs)
                tower_dtype = next(self.vision_tower.parameters()).dtype
                image_features = image_features.to(tower_dtype)
                all_image_features = all_image_features.to(tower_dtype)
                feats_list.append(image_features)
                all_feats_list.append(all_image_features)

            image_features = torch.cat(feats_list, dim=0)
            all_image_features = torch.cat(all_feats_list, dim=0)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True
            )
            image_features, all_image_features = self.feature_select(image_forward_outs)
            image_features = image_features.to(images.dtype)
            all_image_features = all_image_features.to(images.dtype)

        image_features, actual_dims = self.token_reduction(image_features, all_image_features, text_features)

        return image_features, actual_dims

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
