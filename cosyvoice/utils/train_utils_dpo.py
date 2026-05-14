from contextlib import nullcontext
import os
import torch
import torch.nn.functional as F

from cosyvoice.utils.train_utils import (
    init_distributed, init_dataset_and_dataloader, check_modify_and_save_config,
    wrap_cuda_model, init_optimizer_and_scheduler, init_summarywriter, save_model,
    cosyvoice_join, batch_backward, update_parameter_and_lr, log_per_step, log_per_save
)


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
        policy_chosen_logps = policy_dict['logps']
        policy_rejected_logps = policy_dict['rejected_logps']

        reference_chosen_logps = ref_dict['logps']
        reference_rejected_logps = ref_dict['rejected_logps']

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        label_smoothing = 0.0
        beta = 0.1
        alpha = 1

        dpo_losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        dpo_loss = dpo_losses.mean()

        chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        policy_dict['reward acc'] = (chosen_reward > rejected_rewards).float().mean()

        policy_dict['sft_loss'] = policy_dict['loss']
        policy_dict['dpo_loss'] = dpo_loss
        total_loss = alpha * policy_dict['sft_loss'] + dpo_loss
        policy_dict['loss'] = total_loss
        policy_dict['chosen_reward'] = chosen_reward.mean().detach()
        policy_dict['rejected_rewards'] = rejected_rewards.mean().detach()

        del policy_dict['logps']
        del policy_dict['rejected_logps']
        info_dict['loss_dict'] = policy_dict
    return info_dict
