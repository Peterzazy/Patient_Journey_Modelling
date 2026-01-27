import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModel,
)

class TemporalMultiHeadLabelAttentionClassifier(nn.Module):
    """ Masked Multihead Label Attention Classifier.

    Performs masked multihead attention using label embeddings
    as queries and document encodings as keys and values.

    This class also applies linear projection and sigmoid
    to obtain the final probability of each label.
    """

    def __init__(
            self,
            hidden_size,
            seq_len,
            num_labels,
            num_heads,
            device,
            all_tokens=True,
            reduce_computation=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.device = device
        self.all_tokens = all_tokens
        self.reduce_computation = reduce_computation

        self.multiheadattn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True
        )

        self.label_queries = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.num_labels, self.hidden_size), dtype=torch.float
            ),
            requires_grad=True,
        )
        self.label_weights = nn.parameter.Parameter(
            torch.normal(
                0, 0.1, size=(self.hidden_size, self.num_labels), dtype=torch.float
            ),
            requires_grad=True,
        )

    def forward(self, encoding, all_tokens=True, cutoffs=None):
        # encoding: Tensor of size (Nc x T) x H
        # mask: Tensor of size Nn x (Nc x T) x H
        # temporal_encoding = Nn x (N x T) x hidden_size
        T = self.seq_len
        if not self.all_tokens:
            T = 1  # only use the [CLS]-token representation
        Nc = int(encoding.shape[0] / T)
        H = self.hidden_size
        Nl = self.num_labels

        # label query: shape L, H
        # encoding: hape NcxT, H
        # query shape:  Nn, L, H
        # key shape: Nn, Nc*T, H
        # values shape: Nn, Nc*T, H
        # key padding mask: Nn, Nc*T (true if ignore)
        # output: N, L, H
        mask = torch.ones(size=(Nc, Nc * T), dtype=torch.bool).to(device=self.device)
        for i in range(Nc):
            mask[i, : (i + 1) * T] = False

        # only mask out at 2d, 5d, 13d and no DS to reduce computation
        # get list of cutoff indices from cutoffs dictionary

        if self.reduce_computation:
            cutoff_indices = [cutoffs[key][0] for key in cutoffs]
            mask = mask[cutoff_indices, :]

        attn_output = self.multiheadattn.forward(
            query=self.label_queries.repeat(mask.shape[0], 1, 1),
            key=encoding.repeat(mask.shape[0], 1, 1),
            value=encoding.repeat(mask.shape[0], 1, 1),
            key_padding_mask=mask,
            need_weights=False,
        )[0]

        score = torch.sum(
            attn_output
            * self.label_weights.unsqueeze(0).view(
                1, self.num_labels, self.hidden_size
            ),
            dim=2,
        )
        return score


class HierARDocumentTransformer(nn.Module):
    """Hierarchical Autoregressive Transformer.

    This class includes the hierarchical autoregressive transformer,
    which runs over the document embeddings applying masked
    multihead attention to the previous document embeddings.
    """
    def __init__(self, hidden_size, num_layers=1, nhead=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=nhead
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )

    def forward(self, document_encodings):
        # flag is causal = True so that it cannot attend to future document embeddings
        mask = nn.Transformer.generate_square_subsequent_mask(
            sz=document_encodings.shape[0]
        )
        document_encodings = self.transformer_encoder(
            document_encodings, mask=mask
        ).squeeze(
            1
        )  # shape Nc x 1 x D

        return document_encodings


class FastLabEventEmbedder(nn.Module):
    def __init__(self, num_items, num_flags, hidden_size=768):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, hidden_size)
        """self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )"""
        self.flag_embed = nn.Embedding(num_flags, hidden_size)

    def forward(self, item_ids, flags, value_nums=None):
        item_vec = self.item_embed(item_ids)                  # [B, hidden]
        flag_vec = self.flag_embed(flags)                  # [B, hidden]
        # value_input = value_nums.unsqueeze(1)    # [B, 1]
        # value_vec = self.value_mlp(value_input)               # [B, hidden]
        return item_vec + flag_vec                # [B, hidden]


class FastMicrobioEventEmbedder(nn.Module):
    def __init__(self, num_specimens, num_organisms, num_antibiotics, num_interpretations, hidden_size=768):
        super().__init__()
        self.specimen_embed = nn.Embedding(num_specimens, hidden_size)
        self.organism_embed = nn.Embedding(num_organisms, hidden_size)
        self.antibiotic_embed = nn.Embedding(num_antibiotics, hidden_size)
        self.interpretation_embed = nn.Embedding(num_interpretations, hidden_size)

    def forward(self, specimen_ids, organism_ids, antibiotic_ids, interpretation_ids):
        specimen_vec = self.specimen_embed(specimen_ids)
        organism_vec = self.organism_embed(organism_ids)
        antibiotic_vec = self.antibiotic_embed(antibiotic_ids)
        interp_vec = self.interpretation_embed(interpretation_ids)
        return specimen_vec + organism_vec + antibiotic_vec + interp_vec  # [B, hidden]


class FastDrugEventEmbedder(nn.Module):
    def __init__(self, num_gsn, num_routes, hidden_size=768):
        super().__init__()
        self.gsn_embed = nn.Embedding(num_gsn, hidden_size)
        self.route_embed = nn.Embedding(num_routes, hidden_size)
        """self.dose_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )"""

    def forward(self, gsn_ids, route_ids, dose_values = None, unit_ids= None):
        gsn_vec = self.gsn_embed(gsn_ids)
        route_vec = self.route_embed(route_ids)
        # dose_input = dose_values.unsqueeze(0)
        # dose_vec = self.dose_mlp(dose_input)
        return gsn_vec + route_vec       # [B, hidden]


