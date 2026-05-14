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
        speech_token = batch['speech_token'].to(device)
        speech_token_len = batch['speech_token_len'].to(device)
        rejected_speech_token = batch['rejected_speech_token'].to(device)
        rejected_speech_token_len = batch['rejected_speech_token_len'].to(device)
        embedding = batch['embedding'].to(device)
        rejected_embedding = batch['rejected_embedding'].to(device)

        # Prepare targets
        lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + speech_token[i, :speech_token_len[i]].tolist() + [self.speech_token_size]) for i in range(text_token.size(0))]
        lm_target = pad_sequence(lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        rejected_lm_target = [torch.tensor([IGNORE_ID] * (2 + text_token_len[i]) + rejected_speech_token[i, :rejected_speech_token_len[i]].tolist() + [self.speech_token_size]) for i in range(text_token.size(0))]
        rejected_lm_target = pad_sequence(rejected_lm_target, batch_first=True, padding_value=IGNORE_ID).to(device)

        # Encode text
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # Embedding projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)
        embedding = embedding.unsqueeze(1)

        rejected_embedding = F.normalize(rejected_embedding, dim=1)
        rejected_embedding = self.spk_embed_affine_layer(rejected_embedding)
        rejected_embedding = rejected_embedding.unsqueeze(1)

        # Special tokens
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # Embed speech tokens
        speech_token = self.speech_embedding(speech_token)
        rejected_speech_token = self.speech_embedding(rejected_speech_token)

        # Forward chosen
        lm_input, lm_input_len = self.pad_unpad_sequence(sos_eos_emb, embedding, text_token, text_token_len, task_id_emb, speech_token, speech_token_len)
        lm_output, _ = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        logps, _ = get_batch_logps(logits, lm_target)

        # Forward rejected
        rejected_lm_input, rejected_lm_input_len = self.pad_unpad_sequence(sos_eos_emb, rejected_embedding, text_token, text_token_len, task_id_emb, rejected_speech_token, rejected_speech_token_len)
        rejected_lm_output, _ = self.llm(rejected_lm_input, rejected_lm_input_len.to(device))
        rejected_logits = self.llm_decoder(rejected_lm_output)
        rejected_logps, _ = get_batch_logps(rejected_logits, rejected_lm_target)

        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(logits.view(-1, self.speech_token_size + 1), lm_target, ignore_label=IGNORE_ID)
        return {'loss': loss, 'acc': acc, 'logps': logps, 'rejected_logps': rejected_logps}
