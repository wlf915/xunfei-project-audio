# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup

target_speaker_embedding = None

def set_reproducible_seed(seed=42):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Reproducibility seed set to {seed}")


def train():
    set_reproducible_seed(42)
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir="/root/autodl-tmp/qwen3tts/logs/tensorboard/exp01_repro_seed42",
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    accumulation_steps = accelerator.gradient_accumulation_steps
    update_steps_per_epoch = math.ceil(len(train_dataloader) / accumulation_steps)
    total_update_steps = num_epochs * update_steps_per_epoch
    warmup_steps = math.ceil(total_update_steps * args.warmup_ratio)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    accelerator.print(
        f"Scheduler: cosine | warmup_ratio={args.warmup_ratio} | "
        f"micro_batches_per_epoch={len(train_dataloader)} | "
        f"update_steps_per_epoch={update_steps_per_epoch} | "
        f"total_update_steps={total_update_steps} | "
        f"warmup_steps={warmup_steps}"
    )

    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad()

            accelerator.print(
                f"Epoch {epoch} | Step {step} | Loss: {loss.item():.6f} | "
                f"LR: {lr_scheduler.get_last_lr()[0]:.10e}"
            )

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            del state_dict
            torch.cuda.empty_cache()

            infer_script = os.environ.get("EPOCH_INFER_SCRIPT")
            test_jsonl = os.environ.get("EPOCH_TEST_JSONL")
            result_root = os.environ.get("EPOCH_RESULT_ROOT")

            if not infer_script or not test_jsonl or not result_root:
                raise RuntimeError(
                    "EPOCH_INFER_SCRIPT, EPOCH_TEST_JSONL and "
                    "EPOCH_RESULT_ROOT must all be set"
                )

            epoch_label = f"epoch{epoch}"
            compare_root_path = Path(result_root) / "compare_audio"
            test_epoch_path = Path(result_root) / "test20" / epoch_label

            existing_compare_files = list(
                compare_root_path.glob(f"{epoch_label}_*.wav")
            )
            if existing_compare_files:
                raise FileExistsError(
                    f"Compare files already exist for {epoch_label}: "
                    f"{existing_compare_files}"
                )
            if test_epoch_path.exists():
                raise FileExistsError(
                    f"Test directory already exists: {test_epoch_path}"
                )

            command = [
                sys.executable,
                infer_script,
                "--checkpoint",
                output_dir,
                "--test-jsonl",
                test_jsonl,
                "--output-root",
                result_root,
                "--epoch-label",
                epoch_label,
                "--speaker-name",
                args.speaker_name,
                "--seed",
                "42",
            ]

            accelerator.print(
                f"Starting inference for epoch {epoch}: "
                f"compare_root={compare_root_path}, "
                f"test={test_epoch_path}"
            )
            subprocess.run(command, check=True)

            compare_files = list(
                compare_root_path.glob(f"{epoch_label}_*.wav")
            )
            test_files = list(test_epoch_path.glob("test_*.wav"))
            wav_files = compare_files + test_files

            accelerator.print(
                f"Epoch {epoch} verification: "
                f"all_wav={len(wav_files)}, "
                f"compare={len(compare_files)}, "
                f"test={len(test_files)}"
            )

            if len(wav_files) != 23:
                raise RuntimeError(
                    f"Epoch {epoch}: expected 23 wav files, found {len(wav_files)}"
                )
            if len(compare_files) != 3:
                raise RuntimeError(
                    f"Epoch {epoch}: expected 3 compare files, "
                    f"found {len(compare_files)}"
                )
            if len(test_files) != 20:
                raise RuntimeError(
                    f"Epoch {epoch}: expected 20 test files, "
                    f"found {len(test_files)}"
                )

            shutil.rmtree(output_dir)
            accelerator.print(
                f"Epoch {epoch} inference verified; checkpoint deleted: {output_dir}"
            )
            model.train()

if __name__ == "__main__":
    train()
