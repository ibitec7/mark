# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from argparse import ArgumentParser
from collections import defaultdict
import torch
from omegaconf.omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_mamba_model import MegatronMambaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision

from nemo.utils import logging

from omegaconf import open_dict

from megatron.core.models.mamba.mamba_model import MambaModel
from nemo.collections.llm import BaseMambaConfig130M
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

ckpt_path = "./models/pytorch_model.bin"
pyt_checkpoint = torch.load(ckpt_path)

dtype = pyt_checkpoint["lm_head.weight"].dtype
device = pyt_checkpoint["lm_head.weight"].device

pyt_checkpoint["lm_head.weight"] = torch.randn(30528, 768, device=device, dtype=dtype)
pyt_checkpoint["backbone.embedding.weight"] = torch.randn(30528, 768, device=device, dtype=dtype)

new_ckpt_path = os.path.join(os.path.dirname(ckpt_path), f"wrapped_{os.path.basename(ckpt_path)}")

torch.save({"model": pyt_checkpoint}, new_ckpt_path)


'''
Example

CUDA_VISIBLE_DEVICES="0" python /opt/NeMo/scripts/checkpoint_converters/convert_mamba2_pyt_to_nemo.py \
                                --input_name_or_path <path to the source pytorch model> \
                                --output_path <path to target .nemo model> \
                                --mamba_ssm_ngroups 8 \
                                --precision bf16 \
                                --tokenizer_model_dir <path to tokenizer.model, only set for 8b models, otherwise defaults to None>
'''

# Initialize distributed environment
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["LOCAL_RANK"] = "0"

def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=f"{os.path.dirname(__file__)}/../../examples/nlp/language_modeling/conf/megatron_mamba_config.yaml",
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument("--mamba_ssm_ngroups", type=int, default=8, help="ngroups for Mamba model")
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["bf16", "32"], help="Precision for checkpoint weights saved"
    )
    parser.add_argument(
        "--tokenizer_model_dir", type=str, default=None, help="Path to the tokenizer.model, required for 8b models"
    )
    args = parser.parse_args()
    return args


def convert(args):

    checkpoint_weights = torch.load(args.input_name_or_path, map_location='cpu')['model']
    new_state_dict = {}

    config = BaseMambaConfig130M()

    config2 = TransformerConfig(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        ffn_hidden_size=config.ffn_hidden_size,
        use_cpu_initialization=True,
        mamba_num_groups=1,
    )

    if 'backbone' in list(checkpoint_weights.keys())[0]:
        if 'model' in list(checkpoint_weights.keys())[0]:
            checkpoint_weights = {key.replace('model.', '', 1): value for key, value in checkpoint_weights.items()}

            # Codestral Mamba Model Tokenizer Settings
            tokenizer_library = 'megatron'
            tokenizer_type = 'GPTSentencePieceTokenizer'
            tokenizer_model = args.tokenizer_model_dir

        else:

            # Tri Dao and Albert Gu Mamba Model Tokenizer Settings
            tokenizer_library = 'huggingface'
            tokenizer_type = 'EleutherAI/gpt-neox-20b'
            tokenizer_model = None

        layer_keys = [key for key in checkpoint_weights.keys() if re.match(r'backbone\.layers\.\d+\.', key)]
        layer_numbers = set(int(re.search(r'backbone\.layers\.(\d+)\.', key).group(1)) for key in layer_keys)
        num_layers = max(layer_numbers) + 1

        direct_mappings = {
            'embedding.word_embeddings.weight': 'backbone.embedding.weight',
            'decoder.final_norm.weight': 'backbone.norm_f.weight',
            'output_layer.weight': 'lm_head.weight',
        }

        for new_key, old_key in direct_mappings.items():
            new_state_dict[new_key] = checkpoint_weights[old_key]

        layer_attributes = [
            'mixer.A_log',
            'mixer.D',
            'mixer.conv1d.weight',
            'mixer.conv1d.bias',
            'mixer.in_proj.weight',
            'mixer.dt_bias',
            'mixer.out_proj.weight',
            'mixer.norm.weight',
            'norm.weight',
        ]

        for i in range(num_layers):
            for attr in layer_attributes:
                if attr == 'norm.weight':
                    new_key = f'decoder.layers.{i}.mixer.in_proj.layer_norm_weight'
                    old_key = f'backbone.layers.{i}.norm.weight'
                else:
                    new_key = f'decoder.layers.{i}.{attr}'
                    old_key = f'backbone.layers.{i}.{attr}'
                new_state_dict[new_key] = checkpoint_weights[old_key]

    else:

        layer_keys = [key for key in checkpoint_weights.keys() if re.match(r'decoder\.layers\.\d+\.', key)]
        layer_numbers = set(int(re.search(r'decoder\.layers\.(\d+)\.', key).group(1)) for key in layer_keys)
        num_layers = max(layer_numbers) + 1

        for key, value in checkpoint_weights.items():
            if '.norm.weight' in key and 'mixer' not in key:
                key = key[:-11] + 'mixer.in_proj.layer_norm_weight'
            new_state_dict[key] = value

        # NVIDIA Mamba Model Tokenizer Settings
        tokenizer_library = 'megatron'
        tokenizer_type = 'GPTSentencePieceTokenizer'
        tokenizer_model = args.tokenizer_model_dir

    layers = defaultdict(list)

    for key in new_state_dict.keys():
        match = re.match(r'decoder\.layers\.(\d+)\.(\w+)', key)
        if match:
            index, layer_type = match.groups()
            layers[index].append(layer_type)

    layer_pattern = ''
    for i in range(max(map(int, layers.keys())) + 1):
        index_str = str(i)
        layer_types = layers.get(index_str, [])
        if 'mixer' in layer_types:
            layer_pattern += 'M'
        elif 'self_attention' in layer_types:
            layer_pattern += '*'
        elif 'mlp' in layer_types:
            layer_pattern += '-'
        else:
            raise AssertionError("Layer not found. Each layer must be eiher MLP, Mamba, or Attention")

    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.trainer["precision"] = args.precision
    nemo_config.model.vocab_size, nemo_config.model.hidden_size = new_state_dict[
        'embedding.word_embeddings.weight'
    ].shape
    nemo_config.model.num_layers = num_layers
    nemo_config.model.hybrid_override_pattern = layer_pattern
    nemo_config.model.mamba_ssm_ngroups = args.mamba_ssm_ngroups
    nemo_config.model.tokenizer.library = tokenizer_library
    nemo_config.model.tokenizer.type = tokenizer_type
    nemo_config.model.tokenizer.model = tokenizer_model

    if "-" in layer_pattern:
        nemo_config.model.ffn_hidden_size = new_state_dict[
            f'decoder.layers.{layer_pattern.index("-")}.mlp.linear_fc1.weight'
        ].shape[0]
    else:
        nemo_config.model.ffn_hidden_size = nemo_config.model.hidden_size

    nemo_config.model.use_cpu_initialization = True

    logging.info(f"Loading Mamba2 Pytorch checkpoint : `{args.input_name_or_path}`")

    # model = MambaModel(
    #     config=config2,
    #     mamba_stack_spec=mamba_stack_spec,
    #     vocab_size=30528,
    #     max_sequence_length=4096
    # )

    print("Megatron model initialized, now loading state dict...")

    trainer = MegatronLMPPTrainerBuilder(nemo_config).create_trainer()


    nemo_model_from_pyt = MegatronMambaModel(cfg=nemo_config, trainer=trainer)
    # nemo_model_from_pyt.model = model

    # for k, v in nemo_model_from_pyt.state_dict().items():
    #     if "_extra" in k:
    #         new_state_dict[k] = v

    # model.load_state_dict(new_state_dict, strict=False)

    # OmegaConf.save(config=nemo_config, f="./configs/megatron_config.yaml")

    # torch.save({
    #     "model": model.state_dict(),
    #     "config": {
    #         "num_layers": 24,
    #         "hidden_size": 768,
    #         "mamba_num_ssm_groups": 1,
    #         "vocab_size": 30528,
    #         "max_sequence_length": 4096,
    #     }
    # }, "./models/guider.pt")

    nemo_model_from_pyt.load_state_dict(new_state_dict, strict=False)
    dtype = torch_dtype_from_precision(args.precision)
    nemo_model_from_pyt = nemo_model_from_pyt.to(dtype=dtype)
    nemo_model_from_pyt.save_to(args.output_path)
    logging.info(f'Mamba2 NeMo model saved to: ./models/guider.pt')


if __name__ == '__main__':
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(1, 1)

    model_parallel_cuda_manual_seed(59)

    torch.cuda.random.manual_seed_all(59)

    args = get_args()
    convert(args)

    print("Conversion completed successfully.")

    torch.distributed.destroy_process_group()
