from contextlib import nullcontext
import os
import torch
import torch.nn.functional as F
import numpy as np

from cosyvoice.utils.train_utils import (
    init_distributed, init_dataset_and_dataloader, check_modify_and_save_config,
    wrap_cuda_model, init_optimizer_and_scheduler, init_summarywriter, save_model,
    cosyvoice_join, batch_backward, update_parameter_and_lr, log_per_step, log_per_save
)


def calculate_dpo_loss(policy_chosen_logps, reference_chosen_logps, policy_rejected_logps, reference_rejected_logps, beta=0.1, label_smoothing=0.0):
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

    dpo_losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    return dpo_losses.mean()


def batch_forward(model, batch, info_dict, ref_model=None, other_args=None):
    device = int(os.environ.get('LOCAL_RANK', 0))

    dtype = info_dict["dtype"]
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if info_dict['train_engine'] == 'torch_ddp':
        autocast = nullcontext()
    else:
        autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)

    with torch.no_grad(), nullcontext():
        ref_dict = ref_model(batch, device)

    with autocast:
        policy_dict = model(batch, device)

        length = other_args[0]
        use_lambda = other_args[1]
        lambda_type = other_args[2]

        policy_logps_list = [policy_dict['logps']] + [policy_dict[f'logps{i}'] for i in range(2, length + 1)]
        reference_logps_list = [ref_dict['logps']] + [ref_dict[f'logps{i}'] for i in range(2, length + 1)]

        # Formula 2: psi_{c,l}(i) = 1 - (i-1)/(K+2), i=1,...,K+2, where K+2 = length
        psi_values = [1.0 - (i - 1) / length for i in range(1, length + 1)]

        if use_lambda:
            # Formula 7: G(i) = 2^{psi(i)} - 1
            G = [(2 ** psi - 1) for psi in psi_values]

        delta_list = []
        for i in range(length):
            for j in range(i + 1, length):
                if use_lambda and lambda_type in ["index", "score"]:
                    # Formula 8: lambda_{i,j} = |G(i)-G(j)| * |1/D(i) - 1/D(j)|
                    # where D(i) = 1/log(1+i), so 1/D(i) = log(1+i), i is 1-indexed
                    delta = abs((G[i] - G[j]) * (np.log(2 + i) - np.log(2 + j)))
                    delta_list.append(delta)
                else:
                    delta_list.append(1.0)

        mean_delta = sum(delta_list) / len(delta_list)
        normalized_deltas = [x / mean_delta for x in delta_list]

        lipo_loss = 0.0
        index = 0
        for i in range(length):
            for j in range(i + 1, length):
                if use_lambda and lambda_type in ["index", "score"]:
                    lipo_loss += normalized_deltas[index] * calculate_dpo_loss(
                        policy_logps_list[i], reference_logps_list[i],
                        policy_logps_list[j], reference_logps_list[j])
                    index += 1
                else:
                    lipo_loss += calculate_dpo_loss(
                        policy_logps_list[i], reference_logps_list[i],
                        policy_logps_list[j], reference_logps_list[j])

        alpha = 1.0
        policy_dict['sft_loss'] = policy_dict['loss']
        policy_dict['lipo_loss'] = lipo_loss
        policy_dict['loss'] = alpha * policy_dict['sft_loss'] + lipo_loss

        del policy_dict['logps']
        for i in range(2, length + 1):
            del policy_dict[f'logps{i}']
        info_dict['loss_dict'] = policy_dict
    return info_dict
