from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.utils.common import th_accuracy
from cosyvoice.llm.llm import TransformerLM as BaseTransformerLM, get_batch_logps


class TransformerLM(BaseTransformerLM):

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        text_token = batch['text_token'].to(device)
        text_token_len = batch['text_token_len'].to(device)

        # Determine how many candidates are in the batch
        num_candidates = 1
        while f'speech_token{num_candidates + 1}' in batch:
            num_candidates += 1

        # Load all speech tokens and embeddings
        suffixes = [''] + [str(i) for i in range(2, num_candidates + 1)]
        speech_tokens = []
        speech_token_lens = []
        embeddings = []
        for suffix in suffixes:
            speech_tokens.append(batch[f'speech_token{suffix}'].to(device))
            speech_token_lens.append(batch[f'speech_token_len{suffix}'].to(device))
            embeddings.append(batch[f'embedding{suffix}'].to(device))

        # Prepare lm_targets for all candidates
        lm_targets = []
        for idx in range(num_candidates):
            lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_tokens[idx][i, :speech_token_lens[idx][i]].tolist() + [self.speech_token_size]) for i in range(text_token.size(0))]
            lm_targets.append(pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device))

        # Encode text
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # Project all embeddings
        for idx in range(num_candidates):
            embeddings[idx] = F.normalize(embeddings[idx], dim=1)
            embeddings[idx] = self.spk_embed_affine_layer(embeddings[idx])
            embeddings[idx] = embeddings[idx].unsqueeze(1)

        # Special tokens
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # Embed all speech tokens
        for idx in range(num_candidates):
            speech_tokens[idx] = self.speech_embedding(speech_tokens[idx])

        # Run LLM forward for all candidates and compute logps
        result = {}
        for idx in range(num_candidates):
            lm_input, lm_input_len = self.pad_unpad_sequence(
                sos_eos_emb, embeddings[idx], text_token, text_token_len,
                task_id_emb, speech_tokens[idx], speech_token_lens[idx])
            lm_output, _ = self.llm(lm_input, lm_input_len.to(device))
            logits = self.llm_decoder(lm_output)
            logps, _ = get_batch_logps(logits, lm_targets[idx])

            if idx == 0:
                loss = self.criterion_ce(logits, lm_targets[0])
                acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_targets[0], ignore_label=IGNORE_ID)
                result['loss'] = loss
                result['acc'] = acc
                result['logps'] = logps
            else:
                result[f'logps{idx + 1}'] = logps

        return result
