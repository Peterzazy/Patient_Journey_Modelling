import torch
import torch.nn as nn
from Models.PJ.event_transformer_icd import *
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward_sequential(self, note_embed, event_embeds):
        enriched = note_embed.clone()
        gate_list = []
        for e in event_embeds:
            g = torch.sigmoid(self.gate_linear(torch.cat([note_embed, e], dim=-1)))
            enriched = enriched + g * e
            gate_list.append(g)
        return enriched, gate_list

    def forward(self, note_embed, event_embeds):
        """
        note_embed: [hidden]
        event_embeds: [num_events, hidden]
        """
        # Expand note_embed to [num_events, hidden]
        note_expanded = note_embed.unsqueeze(0).expand(event_embeds.size(0), -1)

        # Concatenate all pairs: [num_events, hidden * 2]
        concat = torch.cat([note_expanded, event_embeds], dim=-1)

        # Compute gates in batch: [num_events, hidden]
        gates = torch.sigmoid(self.gate_linear(concat))

        # Weighted contributions: [num_events, hidden]
        gated_events = gates * event_embeds

        # Sum all contributions: [hidden]
        enrichment = gated_events.sum(dim=0)

        # Final enriched embedding
        enriched = note_embed + enrichment

        return enriched, gates


class GatedFusionTokens(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward_sequential(self, token_embeds, event_embeds):
        """
        token_embeds: [num_tokens, hidden]
        event_embeds: [num_events, hidden]
        Returns:
            enriched_tokens: [num_tokens, hidden]
            gates: List of [num_tokens, hidden] (one per event)
        """
        enriched_tokens = token_embeds.clone()
        all_gates = []
        for e in event_embeds:
            # For each event, compute gate for each token
            repeated_event = e.unsqueeze(0).expand(token_embeds.size(0), -1)  # [num_tokens, hidden]
            g = torch.sigmoid(self.gate_linear(torch.cat([token_embeds, repeated_event], dim=-1)))
            enriched_tokens = enriched_tokens + g * repeated_event
            all_gates.append(g)
        return enriched_tokens, all_gates

    def forward(self, note_embed, event_embeds):
        """
        note_embed: [num_tokens, hidden]
        event_embeds: [num_events, hidden]
        """
        num_tokens = note_embed.size(0)
        num_events = event_embeds.size(0)

        # Expand note_embed to [num_tokens, num_events, hidden]
        note_expanded = note_embed.unsqueeze(1).expand(-1, num_events, -1)

        # Expand event_embeds to [num_tokens, num_events, hidden]
        event_expanded = event_embeds.unsqueeze(0).expand(num_tokens, -1, -1)

        # Concatenate along last dim: [num_tokens, num_events, hidden*2]
        concat = torch.cat([note_expanded, event_expanded], dim=-1)

        # Compute gates: [num_tokens, num_events, hidden]
        gates = torch.sigmoid(self.gate_linear(concat))

        # Multiply gates * events: [num_tokens, num_events, hidden]
        gated_events = gates * event_expanded

        # Sum over events: [num_tokens, hidden]
        enrichment = gated_events.sum(dim=1)

        enriched = note_embed + enrichment

        return enriched, gates

class MaskedGatedFusion(nn.Module):
    def __init__(self, hidden_size, fusion="gating"):
        super().__init__()

        self.fusion = fusion
        if fusion == "gating":
            self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)
            nn.init.xavier_uniform_(self.gate_linear.weight)
            if self.gate_linear.bias is not None:
                nn.init.zeros_(self.gate_linear.bias)

        elif fusion == "attention":
            self.query_proj = nn.Linear(hidden_size, hidden_size)  # projection for queries
            nn.init.xavier_uniform_(self.query_proj.weight)
            if self.query_proj.bias is not None:
                nn.init.zeros_(self.query_proj.bias)
        else:
            self.multiheadattn = nn.MultiheadAttention(
                hidden_size, num_heads=1, batch_first=True
            )
            for name, param in self.multiheadattn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        self.layernorm = nn.LayerNorm(hidden_size)

    def forward_gate(self, text_embeds, struct_embeds, mask):
        """
        text_embeds: [num_texts, hidden]
        struct_embeds: [num_struct, hidden]
        mask: [num_texts, num_struct]
        """
        num_texts = text_embeds.size(0)
        num_struct = struct_embeds.size(0)

        # Expand text embeddings to [num_texts, num_struct, hidden]
        text_exp = text_embeds.unsqueeze(1).expand(-1, num_struct, -1)
        struct_exp = struct_embeds.unsqueeze(0).expand(num_texts, -1, -1)

        # Concatenate: [num_texts, num_struct, hidden*2]
        concat = torch.cat([text_exp, struct_exp], dim=-1)

        # Gates: [num_texts, num_struct, hidden]
        gates = torch.relu(self.gate_linear(concat))

        # Masked gates: zero out gates for invalid events
        gates = gates * mask.unsqueeze(-1)  # broadcasting mask

        # Weighted contributions
        weighted = gates * struct_exp

        # Sum over events: [num_texts, hidden]
        enrichment = weighted.sum(dim=1)

        enriched = text_embeds + enrichment

        enriched = self.layernorm(enriched)
        return enriched, gates

    def forward_attention(self, text_embeds, struct_embeds, mask):
        queries = self.query_proj(text_embeds)  # [num_texts, hidden]

        # Dot-product attention
        scores = torch.matmul(queries, struct_embeds.T)  # [num_texts, num_struct]
        scores = scores.masked_fill(mask == False, -1e3)  # masking

        attn_weights = F.softmax(scores, dim=1)  # [num_texts, num_struct]

        # attn_weights = torch.sigmoid(scores)

        # Zero out weights where no valid struct entries (safely)
        no_valid = (mask.sum(dim=1) == 0)  # [num_texts]
        attn_weights = attn_weights * (~no_valid).float().unsqueeze(1)  # out-of-place masking

        enrichment = torch.matmul(attn_weights, struct_embeds)  # [num_texts, hidden]
        # enriched = self.layernorm(enrichment+text_embeds)  # optional: add residual

        enriched = enrichment
        return enriched, attn_weights

    def forward_mult_head(self, text_embeds, struct_embeds, mask):
        """
            text_embeds:   [num_texts, hidden]
            struct_embeds: [num_struct, hidden]
            mask:          [num_texts, num_struct] - boolean mask (True = valid)
            """
        num_texts, hidden = text_embeds.size()
        num_struct = struct_embeds.size(0)

        # [1] Prepare query/key/value
        query = text_embeds.unsqueeze(1)  # [T, 1, H]
        key = struct_embeds.unsqueeze(0).expand(num_texts, -1, -1)  # [T, S, H]
        value = key  # [T, S, H]

        # [2] Compute key_padding_mask and fix rows with all-False masks
        key_padding_mask = ~mask.bool()  # PyTorch expects True = pad
        no_valid_struct = mask.sum(dim=1) == 0  # [T]
        key_padding_mask[no_valid_struct] = False  # Allow full attention to avoid NaNs

        # [3] Apply multi-head attention
        attn_output, attn_weights = self.multiheadattn(
            query=query,  # [T, 1, H]
            key=key,  # [T, S, H]
            value=value,  # [T, S, H]
            key_padding_mask=key_padding_mask,
            need_weights=True
        )

        # [4] Output is [T, 1, H] → squeeze sequence length
        enrichment = attn_output.squeeze(1)  # [T, H]
        attn_weights = attn_weights.squeeze(1)  # [T, S]

        # [5] Residual + layer norm
        enriched = self.layernorm(enrichment+text_embeds)

        return enriched, attn_weights.squeeze(1)


    def forward(self, text_embeds, struct_embeds, mask):
        if self.fusion == "attention":
            enriched, gates = self.forward_attention(text_embeds, struct_embeds, mask)

        elif self.fusion == "gating":
            enriched, gates = self.forward_gate(text_embeds, struct_embeds, mask)
        else:
            enriched, gates = self.forward_mult_head(text_embeds, struct_embeds, mask)

        return enriched, gates


class Model(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        for key in config:
            setattr(self, key, config[key])
        self.seq_len = 512
        self.hidden_size = 768
        self.device = device

        #self.fusion_layer = GatedFusionTokens(self.hidden_size) if self.use_all_tokens else GatedFusion(
            # self.hidden_size)
        self.fusion_layer = MaskedGatedFusion(self.hidden_size)

        self._initialize_event_embedder()
        self.transformer = AutoModel.from_pretrained(self.base_checkpoint)
        if self.use_multihead_attention:
            self.label_attn = TemporalMultiHeadLabelAttentionClassifier(
                self.hidden_size,
                self.seq_len,
                self.num_labels,
                self.num_heads_labattn,
                device=device,
                all_tokens=self.use_all_tokens,
                reduce_computation=self.reduce_computation,
            )
        self.document_regressor = HierARDocumentTransformer(
            self.hidden_size, self.num_layers, self.num_attention_heads
        )

        self._initialize_embeddings()
        self.event_norm = nn.LayerNorm(self.hidden_size)

    def _initialize_event_embedder(self):
        self.lab_embedder = FastLabEventEmbedder(num_items=self.lab_param["num_items"],
                                                 num_flags=self.lab_param["num_flags"])
        self.microbio_embedder = FastMicrobioEventEmbedder(num_specimens=self.microbio_params["num_specimens"],
                                                           num_organisms=self.microbio_params["num_organisms"],
                                                           num_antibiotics=self.microbio_params["num_antibiotics"],
                                                           num_interpretations=self.microbio_params[
                                                               "num_interpretations"])
        self.drug_embedder = FastDrugEventEmbedder(num_gsn=self.drug_param["num_gsn"],
                                                   num_routes=self.drug_param["num_routes"])

    def _initialize_embeddings(self):
        self.pelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversepelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.delookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversedelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.celookup = nn.parameter.Parameter(
            torch.normal(0, 0.1, size=(15, 1, self.hidden_size), dtype=torch.float),
            requires_grad=True,
        )

    def event_representation_layer_sequential(self, event_types, seq_tokenized, input_ids, attention_mask):
        full_embeddings = []
        i = 0
        for idx, (etype, evalue) in enumerate(zip(event_types, seq_tokenized)):
            if etype == "Lab":
                item_id, flag = evalue
                embed = self.lab_embedder(
                    torch.tensor([item_id], device=self.device, dtype=torch.long),
                    torch.tensor([flag], device=self.device, dtype=torch.long)
                )[0]
                full_embeddings.append(embed)

            elif etype == "Microbio":
                spec, org, ab, sens = evalue
                embed = self.microbio_embedder(
                    torch.tensor([spec], device=self.device, dtype=torch.long),
                    torch.tensor([org], device=self.device, dtype=torch.long),
                    torch.tensor([ab], device=self.device, dtype=torch.long),
                    torch.tensor([sens], device=self.device, dtype=torch.long)
                )[0]
                full_embeddings.append(embed)

            elif etype == "Drug":
                gsn, route = evalue
                embed = self.drug_embedder(
                    torch.tensor([gsn], device=self.device, dtype=torch.long),
                    torch.tensor([route], device=self.device, dtype=torch.long)
                )[0]
                full_embeddings.append(embed)

            elif etype == "Text":
                # output = self.transformer(evalue[0].unsqueeze(0), evalue[1].unsqueeze(0)).last_hidden_state
                output = self.transformer(input_ids[i].unsqueeze(0), attention_mask[i].unsqueeze(0)).last_hidden_state
                output = output if self.use_all_tokens else output[:, 0, :]
                full_embeddings.append(output.squeeze(0))
                i += 1
            else:
                raise ValueError(f"Unknown event type {etype}")

        return full_embeddings

    def event_representation_layer(self, event_types, seq_tokenized, input_ids, attention_mask):

        full_embeddings = []

        note_outputs = self.transformer(input_ids, attention_mask).last_hidden_state[:, 0, :]
        txt_indices = [i for i in range(len(event_types)) if event_types[i] == "Text"]

        lab_indices = [i for i in range(len(event_types)) if event_types[i] == "Lab"]
        lab_values = [seq_tokenized[i] for i in range(len(event_types)) if event_types[i] == "Lab"]
        microbio_indices = [i for i in range(len(event_types)) if event_types[i] == "Microbio"]
        microbio_values = [seq_tokenized[i] for i in range(len(event_types)) if event_types[i] == "Microbio"]
        drug_indices = [i for i in range(len(event_types)) if event_types[i] == "Drug"]
        drug_values = [seq_tokenized[i] for i in range(len(event_types)) if event_types[i] == "Drug"]

        lab_embeds, microbio_embed, drug_embed = None, None, None
        if lab_values:
            item_ids = torch.tensor([x[0] for x in lab_values], device=self.device)
            flags = torch.tensor([x[1] for x in lab_values], device=self.device)
            lab_embeds = self.lab_embedder(item_ids, flags)

        if microbio_values:
            spec = torch.tensor([x[0] for x in microbio_values], device=self.device)
            org = torch.tensor([x[1] for x in microbio_values], device=self.device)
            ab = torch.tensor([x[2] for x in microbio_values], device=self.device)
            sens = torch.tensor([x[3] for x in microbio_values], device=self.device)
            microbio_embed = self.microbio_embedder(spec, org, ab, sens)

        if drug_values:
            gsn = torch.tensor([x[0] for x in drug_values], device=self.device)
            route = torch.tensor([x[1] for x in drug_values], device=self.device)
            drug_embed = self.drug_embedder(gsn, route)

        i_text, i_lab, i_micro_bio, i_drug = 0, 0, 0, 0
        for i in range(len(event_types)):
            if event_types[i] == "Lab":
                full_embeddings.append(lab_embeds[i_lab])
                i_lab += 1
            elif event_types[i] == "Microbio":
                full_embeddings.append(microbio_embed[i_micro_bio])
                i_micro_bio += 1
            elif event_types[i] == "Drug":
                full_embeddings.append(drug_embed[i_drug])
                i_drug += 1
            elif event_types[i] == "Text":
                full_embeddings.append(note_outputs[i_text])
                i_text += 1
        return full_embeddings



    def mask_fusion(self,  event_types, full_embeddings):

        full_embeddings = self.event_norm(torch.stack(full_embeddings))

        text_indices = [i for i in range(len(event_types)) if event_types[i] == "Text"]
        structured_event_indices = [i for i in range(len(event_types)) if event_types[i] != "Text"]

        if len(structured_event_indices) == 0:
            return full_embeddings, []

        structured_embeds = torch.stack([full_embeddings[i] for i in structured_event_indices])
        text_embeds = torch.stack([full_embeddings[i] for i in text_indices])
        # For each text idx, find prior structured indices
        mask_list = []
        for text_idx in text_indices:
            mask_row = []
            for struct_idx in structured_event_indices:
                mask_row.append(1 if struct_idx < text_idx else 0)
            mask_list.append(mask_row)

        mask = torch.tensor(mask_list, device=text_embeds.device, dtype=torch.bool)  # [Num_texts, Num_struct]
        enriched_texts, gate_lists = self.fusion_layer(text_embeds, structured_embeds, mask)

        return enriched_texts, gate_lists

    def fusion_sequential(self, event_types, full_embeddings):
        # Final shape
        enriched_texts = []
        gate_lists = []
        structured_event_indices = [i for i, etype in enumerate(event_types) if etype != "Text"]

        if self.use_all_tokens:
            for idx, (etype, embedding) in enumerate(zip(event_types, full_embeddings)):
                if etype == "Text":
                    # Select prior structured events
                    prior_indices = [i for i in structured_event_indices if i < idx]
                    past_events = [full_embeddings[j] for j in prior_indices]
                    if len(past_events) > 0:
                        past_events = torch.stack(past_events)

                        # embedding shape: [num_tokens, hidden]
                        enriched_tokens, gate_ls = self.fusion_layer(embedding, past_events)

                        # Take enriched CLS token
                        enriched_cls = enriched_tokens[0]
                        enriched_texts.append(enriched_cls)
                        gate_lists.append(gate_ls)
                    else:
                        enriched_texts.append(embedding)

        else:

            for idx, (etype, embedding) in enumerate(zip(event_types, full_embeddings)):
                if etype == "Text":
                    # Select prior structured events
                    prior_indices = [i for i in structured_event_indices if i < idx]
                    past_events = [full_embeddings[j] for j in prior_indices]
                    if len(past_events) > 0:
                        past_events = torch.stack(past_events)
                        enriched, gate_ls = self.fusion_layer(embedding, past_events)
                        enriched_texts.append(enriched)
                        gate_lists.append(gate_ls)
                    else:
                        enriched_texts.append(embedding)

        return enriched_texts, gate_lists


    def forward(self, event_types, seq_tokenized, input_ids,
                attention_mask,
                seq_ids,
                category_ids,
                cutoffs,
                note_end_chunk_ids=None,
                token_type_ids=None,
                is_evaluation=False,
                pos_encodings=None,
                rev_encodings=None
                ):

        # full_embeddings = self.event_representation_layer_sequential(event_types, seq_tokenized, input_ids, attention_mask)

        full_embeddings = self.event_representation_layer(event_types, seq_tokenized, input_ids,
                                                                     attention_mask)
        # full_embeddings = self.event_norm(torch.stack(full_embeddings))
        # full_embeddings = list(full_embeddings)

        # enriched_texts, gate_lists = self.fusion_sequential(event_types, full_embeddings)

        if (pos_encodings is not None) and (rev_encodings is not None):
            full_embeddings = torch.stack(full_embeddings) + pos_encodings + rev_encodings
            full_embeddings = list(full_embeddings)


        enriched_texts, gate_lists = self.mask_fusion(event_types, full_embeddings)

        # sequence_output = torch.stack(enriched_texts).unsqueeze(1)

        sequence_output = enriched_texts.unsqueeze(1)
        max_seq_id = seq_ids[-1].item()
        reverse_seq_ids = max_seq_id - seq_ids

        chunk_count = input_ids.size()[0]
        reverse_pos_ids = (chunk_count - torch.arange(chunk_count) - 1).to(self.device)

        if self.use_positional_embeddings:
            sequence_output += self.pelookup[: sequence_output.size()[0], :, :]
        if self.use_reverse_positional_embeddings:
            sequence_output += torch.index_select(
                self.reversepelookup, dim=0, index=reverse_pos_ids
            )

        if self.use_document_embeddings:
            sequence_output += torch.index_select(self.delookup, dim=0, index=seq_ids)
        if self.use_reverse_document_embeddings:
            sequence_output += torch.index_select(
                self.reversedelookup, dim=0, index=reverse_seq_ids
            )

        if self.use_category_embeddings:
            sequence_output += torch.index_select(
                self.celookup, dim=0, index=category_ids
            )
        if self.use_all_tokens:
            # before: sequence_output shape [batchsize, seqlen, hiddensize] = [# chunks, 512, hidden size]
            # after: sequence_output shape [#chunks*512, 1, hidden size]
            sequence_output_all = sequence_output.view(-1, 1, self.hidden_size)
            sequence_output_all = sequence_output_all[:, 0, :]
            sequence_output = sequence_output[:, [0], :]

        else:
            sequence_output = sequence_output[:, [0], :]

        """sequence_output = sequence_output[
            :, 0, :
        ]  """  # remove the singleton to get something of shape [#chunks, hidden_size] or [#chunks*512, hidden_size]

        # if not baseline, add document autoregressor
        if not self.is_baseline:
            # document regressor returns document embeddings and predicted categories
            sequence_output = self.document_regressor(
                sequence_output.view(-1, 1, self.hidden_size)
            )
        # make aux predictions
        if self.aux_task in ("next_document_embedding", "last_document_embedding"):
            if self.apply_transformation:
                aux_predictions = self.document_predictor(sequence_output)
            else:
                aux_predictions = sequence_output
        elif self.aux_task == "next_document_category":
            aux_predictions = self.category_predictor(sequence_output)
        elif self.aux_task == "none":
            aux_predictions = None
        # apply label attention at document-level

        if is_evaluation == False:
            if self.use_all_tokens:
                scores = self.label_attn(sequence_output_all, cutoffs=cutoffs)
            else:
                scores = self.label_attn(sequence_output, cutoffs=cutoffs)
            return scores, sequence_output, gate_lists

        else:
            if self.use_all_tokens:
                return sequence_output_all
            else:
                return sequence_output, gate_lists


class Model_LAHST(nn.Module):
    """Model for ICD-9 code temporal predictions.

    Code based on HTDC (Ng et al, 2022).

    Our contributions:
    - Hierarchical autoregressive transformer
    - Auxiliary tasks, including:
        - next document embedding predictor
        (which can also be used for last emb. pred.)
        - next document category predictor
    """

    def __init__(self, config, device):
        super().__init__()
        for key in config:
            setattr(self, key, config[key])

        self.seq_len = 512
        self.hidden_size = 768
        self.device = device
        self._initialize_embeddings()

        # base transformer
        self.transformer = AutoModel.from_pretrained(self.base_checkpoint)

        # LWAN
        if self.use_multihead_attention:
            self.label_attn = TemporalMultiHeadLabelAttentionClassifier(
                self.hidden_size,
                self.seq_len,
                self.num_labels,
                self.num_heads_labattn,
                device=device,
                all_tokens=self.use_all_tokens,
                reduce_computation=self.reduce_computation,
            )
            # self.label_attn = TemporalLabelAttentionClassifier(
            #     self.hidden_size,
            #     self.seq_len,
            #     self.num_labels,
            #     self.num_heads_labattn,
            #     device=device,
            #     all_tokens=self.use_all_tokens,
            # )
        else:
            pass
        # hierarchical AR transformer
        if not self.is_baseline:
            self.document_regressor = HierARDocumentTransformer(
                self.hidden_size, self.num_layers, self.num_attention_heads
            )

        elif self.aux_task != "none":
            raise ValueError(
                "auxiliary_task must be next_document_embedding or next_document_category or none"
            )

    def _initialize_embeddings(self):
        self.pelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversepelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.delookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversedelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.max_chunks, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.celookup = nn.parameter.Parameter(
            torch.normal(0, 0.1, size=(15, 1, self.hidden_size), dtype=torch.float),
            requires_grad=True,
        )

    def forward(
            self,
            event_types,
            seq_tokenized,
            input_ids,
            attention_mask,
            seq_ids,
            category_ids,
            cutoffs,
            note_end_chunk_ids=None,
            token_type_ids=None,
            is_evaluation=False,
            return_attn_weights=False,
            **kwargs
    ):
        max_seq_id = seq_ids[-1].item()
        reverse_seq_ids = max_seq_id - seq_ids

        chunk_count = input_ids.size()[0]
        reverse_pos_ids = (chunk_count - torch.arange(chunk_count) - 1).to(self.device)

        sequence_output = self.transformer(input_ids, attention_mask).last_hidden_state

        if self.use_positional_embeddings:
            sequence_output += self.pelookup[: sequence_output.size()[0], :, :]
        if self.use_reverse_positional_embeddings:
            sequence_output += torch.index_select(
                self.reversepelookup, dim=0, index=reverse_pos_ids
            )

        if self.use_document_embeddings:
            sequence_output += torch.index_select(self.delookup, dim=0, index=seq_ids)
        if self.use_reverse_document_embeddings:
            sequence_output += torch.index_select(
                self.reversedelookup, dim=0, index=reverse_seq_ids
            )

        if self.use_category_embeddings:
            sequence_output += torch.index_select(
                self.celookup, dim=0, index=category_ids
            )
        if self.use_all_tokens:
            # before: sequence_output shape [batchsize, seqlen, hiddensize] = [# chunks, 512, hidden size]
            # after: sequence_output shape [#chunks*512, 1, hidden size]
            sequence_output_all = sequence_output.view(-1, 1, self.hidden_size)
            sequence_output_all = sequence_output_all[:, 0, :]
            sequence_output = sequence_output[:, [0], :]

        else:
            sequence_output = sequence_output[:, [0], :]  # keep cls

        sequence_output = sequence_output[
                          :, 0, :
                          ]  # remove the singleton to get something of shape [#chunks, hidden_size] or [#chunks*512, hidden_size]

        # if not baseline, add document autoregressor
        if not self.is_baseline:
            # document regressor returns document embeddings and predicted categories
            sequence_output = self.document_regressor(
                sequence_output.view(-1, 1, self.hidden_size)
            )
        # make aux predictions
        if self.aux_task in ("next_document_embedding", "last_document_embedding"):
            if self.apply_transformation:
                aux_predictions = self.document_predictor(sequence_output)
            else:
                aux_predictions = sequence_output
        elif self.aux_task == "next_document_category":
            aux_predictions = self.category_predictor(sequence_output)
        elif self.aux_task == "none":
            aux_predictions = None
        # apply label attention at document-level

        if is_evaluation == False:
            if self.use_all_tokens:
                scores = self.label_attn(sequence_output_all, cutoffs=cutoffs)
            else:
                scores = self.label_attn(sequence_output, cutoffs=cutoffs)
            return scores, sequence_output, aux_predictions

        else:
            if self.use_all_tokens:
                return sequence_output_all
            else:
                return sequence_output, 0


class Model_Event(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        for key in config:
            setattr(self, key, config[key])
        self.seq_len = 512
        self.hidden_size = 768
        self.device = device


        #self.fusion_layer = GatedFusionTokens(self.hidden_size) if self.use_all_tokens else GatedFusion(
            # self.hidden_size)
        self.fusion_layer = MaskedGatedFusion(self.hidden_size)

        self._initialize_event_embedder()
        if self.use_multihead_attention:
            self.label_attn = TemporalMultiHeadLabelAttentionClassifier(
                self.hidden_size,
                self.seq_len,
                self.num_labels,
                self.num_heads_labattn,
                device=device,
                all_tokens=self.use_all_tokens,
                reduce_computation=self.reduce_computation,
            )
        self.document_regressor = HierARDocumentTransformer(
            self.hidden_size, self.num_layers, self.num_attention_heads
        )

        self._initialize_embeddings()

    def _initialize_embeddings(self):
        self.pelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.seq_len, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversepelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.seq_len, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.delookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.seq_len, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.reversedelookup = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.seq_len, 1, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.celookup = nn.parameter.Parameter(
            torch.normal(0, 0.1, size=(3, 1, self.hidden_size), dtype=torch.float),
            requires_grad=True,
        )

    def _initialize_event_embedder(self):
        self.lab_embedder = FastLabEventEmbedder(num_items=self.lab_param["num_items"],
                                                 num_flags=self.lab_param["num_flags"])
        self.microbio_embedder = FastMicrobioEventEmbedder(num_specimens=self.microbio_params["num_specimens"],
                                                           num_organisms=self.microbio_params["num_organisms"],
                                                           num_antibiotics=self.microbio_params["num_antibiotics"],
                                                           num_interpretations=self.microbio_params[
                                                               "num_interpretations"])
        self.drug_embedder = FastDrugEventEmbedder(num_gsn=self.drug_param["num_gsn"],
                                                   num_routes=self.drug_param["num_routes"])


    def event_representation_layer(self, event_types, seq_tokenized):

        full_embeddings = []

        #note_outputs = self.transformer(input_ids, attention_mask).last_hidden_state[:, 0, :]
        txt_indices = [i for i in range(len(event_types)) if event_types[i] == "Text"]

        lab_indices = [i for i in range(len(event_types)) if event_types[i] == "Lab"]
        lab_values = [seq_tokenized[i] for i in range(len(event_types)) if event_types[i] == "Lab"]
        microbio_indices = [i for i in range(len(event_types)) if event_types[i] == "Microbio"]
        microbio_values = [seq_tokenized[i] for i in range(len(event_types)) if event_types[i] == "Microbio"]
        drug_indices = [i for i in range(len(event_types)) if event_types[i] == "Drug"]
        drug_values = [seq_tokenized[i] for i in range(len(event_types)) if event_types[i] == "Drug"]

        lab_embeds, microbio_embed, drug_embed = None, None, None
        if lab_values:
            item_ids = torch.tensor([x[0] for x in lab_values], device=self.device)
            flags = torch.tensor([x[1] for x in lab_values], device=self.device)
            lab_embeds = self.lab_embedder(item_ids, flags)

        if microbio_values:
            spec = torch.tensor([x[0] for x in microbio_values], device=self.device)
            org = torch.tensor([x[1] for x in microbio_values], device=self.device)
            ab = torch.tensor([x[2] for x in microbio_values], device=self.device)
            sens = torch.tensor([x[3] for x in microbio_values], device=self.device)
            microbio_embed = self.microbio_embedder(spec, org, ab, sens)

        if drug_values:
            gsn = torch.tensor([x[0] for x in drug_values], device=self.device)
            route = torch.tensor([x[1] for x in drug_values], device=self.device)
            drug_embed = self.drug_embedder(gsn, route)

        i_text, i_lab, i_micro_bio, i_drug = 0, 0, 0, 0
        for i in range(len(event_types)):
            if event_types[i] == "Lab":
                full_embeddings.append(lab_embeds[i_lab])
                i_lab += 1
            elif event_types[i] == "Microbio":
                full_embeddings.append(microbio_embed[i_micro_bio])
                i_micro_bio += 1
            elif event_types[i] == "Drug":
                full_embeddings.append(drug_embed[i_drug])
                i_drug += 1
        return full_embeddings

    def forward(self, event_types, seq_tokenized,
                seq_ids,
                category_ids,
                cutoffs,
                note_end_chunk_ids=None,
                token_type_ids=None,
                is_evaluation=False,
                pos_encodings=None,
                rev_encodings=None
                ):

        # full_embeddings = self.event_representation_layer_sequential(event_types, seq_tokenized, input_ids, attention_mask)

        full_embeddings = self.event_representation_layer(event_types, seq_tokenized)

        full_embeddings = torch.stack(full_embeddings)

        sequence_output = full_embeddings.unsqueeze(1)
        max_seq_id = seq_ids[-1].item()
        reverse_seq_ids = max_seq_id - seq_ids

        chunk_count = full_embeddings.size()[0]
        reverse_pos_ids = (chunk_count - torch.arange(chunk_count) - 1).to(self.device)

        if self.use_positional_embeddings:
            sequence_output += self.pelookup[: sequence_output.size()[0], :, :]
        if self.use_reverse_positional_embeddings:
            sequence_output += torch.index_select(
                self.reversepelookup, dim=0, index=reverse_pos_ids
            )

        if self.use_document_embeddings:
            sequence_output += torch.index_select(self.delookup, dim=0, index=seq_ids)
        if self.use_reverse_document_embeddings:
            sequence_output += torch.index_select(
                self.reversedelookup, dim=0, index=reverse_seq_ids
            )

        if self.use_category_embeddings:
            sequence_output += torch.index_select(
                self.celookup, dim=0, index=category_ids
            )
        if self.use_all_tokens:
            # before: sequence_output shape [batchsize, seqlen, hiddensize] = [# chunks, 512, hidden size]
            # after: sequence_output shape [#chunks*512, 1, hidden size]
            sequence_output_all = sequence_output.view(-1, 1, self.hidden_size)
            sequence_output_all = sequence_output_all[:, 0, :]
            sequence_output = sequence_output[:, [0], :]
