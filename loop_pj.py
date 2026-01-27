import random
import torch.cuda.amp as amp
import torch
import numpy as np
import time as timing
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import wandb

def make_positional_event(timestamps, event_types, embedding_dim=8, learnable=False, device="cpu"):
    """
    Create forward and reverse positional encodings based on time deltas.

    Args:
        timestamps: list of string timestamps
        event_types: list of event types
        embedding_dim: dimension of positional encoding
        learnable: if True, use learnable embeddings; else use sinusoidal
        device: torch device
    Returns:
        pos_encodings: [seq_len, embedding_dim] forward encoding
        rev_encodings: [seq_len, embedding_dim] reverse encoding
    """
    # Step 1: Convert timestamps to deltas (in minutes)
    timestamps_dt = [pd.Timestamp(t) for t in timestamps]
    t_start = timestamps_dt[0]
    t_end = timestamps_dt[-1]

    deltas_forward = torch.tensor([(t - t_start).total_seconds() / 60 for t in timestamps_dt], dtype=torch.float32,
                                  device=device)
    deltas_reverse = torch.tensor([(t_end - t).total_seconds() / 60 for t in timestamps_dt], dtype=torch.float32,
                                  device=device)

    def sinusoidal_encoding(deltas):
        position = deltas.unsqueeze(1)  # shape [seq_len, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=device) *
                             -(np.log(10000.0) / embedding_dim))
        pe = torch.zeros((len(deltas), embedding_dim), device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    if learnable:
        max_len = len(timestamps)
        pos_embedding_layer = torch.nn.Embedding(max_len, embedding_dim).to(device)
        rev_embedding_layer = torch.nn.Embedding(max_len, embedding_dim).to(device)

        forward_pos_ids = torch.arange(len(timestamps), device=device)
        reverse_pos_ids = torch.arange(len(timestamps) - 1, -1, -1, device=device)

        pos_encodings = pos_embedding_layer(forward_pos_ids)
        rev_encodings = rev_embedding_layer(reverse_pos_ids)
    else:
        pos_encodings = sinusoidal_encoding(deltas_forward)
        rev_encodings = sinusoidal_encoding(deltas_reverse)

    # Step 2: Display

    display = False
    if display:
        for i in range(len(event_types)):
            forward = pos_encodings[i].detach().cpu().numpy()
            reverse = rev_encodings[i].detach().cpu().numpy()
            print(
                f"{i}: {event_types[i]:9} | {timestamps[i]} | +{deltas_forward[i]:6.1f} min | -{deltas_reverse[i]:6.1f} min")
            print(f"   FWD Enc = {forward}")
            print(f"   REV Enc = {reverse}")

    return pos_encodings, rev_encodings


def select_data(event_type_sequence, sequences, sequences_tokenized, timestamps, max_num_events=32):
    max_number = len(sequences)

    if max_number < max_num_events:
        return event_type_sequence, sequences, sequences_tokenized, timestamps

    else:

        indexes = sorted(random.sample(range(0, max_number), k=max_num_events))
        event_type_seq = [event_type_sequence[i] for i in indexes]

        while 'Text' not in event_type_seq:
            indexes = sorted(random.sample(range(0, max_number), k=max_num_events))
            event_type_seq = [event_type_sequence[i] for i in indexes]

        seq = [sequences[i] for i in indexes]
        seq_tokenized = [sequences_tokenized[i] for i in indexes]
        tstamp = [timestamps[i] for i in indexes]
        return event_type_seq, seq, seq_tokenized, tstamp


def get_cutoffs(t0, timestamps, sequences, mode="train"):

    if mode == "train" or mode == "eval":
        cutoffs = {"2d": [-1], "5d": [-1], "13d": [-1], "noDS": [-1], "all": [-1]}
        for i, (time, s) in enumerate(zip(timestamps, sequences)):
            if s[2] != 5:
                hour = pd.Timestamp(time) - pd.Timestamp(t0)
                hour = hour.total_seconds() / 3600

                if hour < 2 * 24:
                    cutoffs["2d"] = [i]
                if hour < 5 * 24:
                    cutoffs["5d"] = [i]
                if hour < 13 * 24:
                    cutoffs["13d"] = [i]
                cutoffs["noDS"] = [i]

    else:
        cutoffs = {"2d": -1, "5d": -1, "13d": -1, "noDS": -1, "all": -1}
        for i, (time, s) in enumerate(zip(timestamps, sequences)):
            if s[2] != 5:
                hour = pd.Timestamp(time) - pd.Timestamp(t0)
                hour = hour.total_seconds() / 3600

                if hour < 2 * 24:
                    cutoffs["2d"] = i
                if hour < 5 * 24:
                    cutoffs["5d"] = i
                if hour < 13 * 24:
                    cutoffs["13d"] = i
                cutoffs["noDS"] = i

    return cutoffs

def select_sequence(event_type_sequence, sequences, sequences_tokenized, timestamps, tokenizer, num_text_chunks=16, mode="train",
                    make_positional=False):
    # First, collect indices of all text chunks

    text_indices = [i for i, e in enumerate(event_type_sequence) if e == 'Text']

    txt = [sequences[i] for i in range(len(event_type_sequence)) if
           event_type_sequence[i] == 'Text']
    output = [tokenizer(doc[0],
                        truncation=True,
                        return_overflowing_tokens=True,
                        padding="max_length",
                        return_tensors="pt") for doc in txt]

    new_sequence_tokenized = []
    new_timestamps = []
    new_event_type_sequence = []

    ids = []
    for i in range(len(event_type_sequence)):
        # Here we convert the sequences to take into account the chunks
        if event_type_sequence[i] == 'Text':
            ids_ = []
            for j in range(len(sequences_tokenized[i][0])):
                new_sequence_tokenized.append([sequences_tokenized[i][0][j],  # input ids of chunk j
                                               sequences_tokenized[i][1][j],  # attn_mask chunk j
                                               sequences_tokenized[i][2]])  # category id
                new_timestamps.append(timestamps[i])
                new_event_type_sequence.append(event_type_sequence[i])
                ids_.append(len(new_sequence_tokenized)-1)
            ids.append(ids_)

        else:
            new_timestamps.append(timestamps[i])
            new_sequence_tokenized.append(sequences_tokenized[i])
            new_event_type_sequence.append(event_type_sequence[i])

    txt_chunk_idx = [i for i, e in enumerate(new_event_type_sequence) if e == 'Text']
    txt_chunk_idx = txt_chunk_idx[-181:]  # To match LAHST preprocessing
    if mode == "train":
        selected_chunks = txt_chunk_idx if (len(txt_chunk_idx) <= num_text_chunks) else sorted(random.sample(txt_chunk_idx, k=num_text_chunks))
        ids_selected = []
        for t in ids:
            _ = []
            for idx in t:
                if idx in selected_chunks:
                    _.append(idx)
            if len(_) > 0:
                ids_selected.append(_)

        event_type_seq, seq, seq_tokenized, tstamp = [], [], [], []
        i = 0

        while i <= selected_chunks[len(selected_chunks)-1]:

            if new_event_type_sequence[i] == 'Text':
                if i in selected_chunks:
                    event_type_seq.append(new_event_type_sequence[i])
                    seq_tokenized.append(new_sequence_tokenized[i])
                    tstamp.append(new_timestamps[i])
            else:
                event_type_seq.append(new_event_type_sequence[i])
                seq_tokenized.append(new_sequence_tokenized[i])
                tstamp.append(new_timestamps[i])
            i += 1

        final_chunks = [seq_tokenized[i] for i in range(len(event_type_seq)) if
                        event_type_seq[i] == 'Text']

        final_input_ids = torch.cat([chunk[0].unsqueeze(0) for chunk in final_chunks])  # this concatenates to (overall # chunks, 512)
        final_attention_mask = torch.cat([chunk[1].unsqueeze(0) for chunk in final_chunks])
        seq_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [[i] * len(ids_selected[i]) for i in range(len(ids_selected))]
                )
            )
        )  # Appartient au même document
        category_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [final_chunks[i][2]]
                        for i in range(len(final_chunks))
                    ]
                )
            )
        )
        seq_ids = torch.LongTensor(seq_ids)
        category_ids = torch.LongTensor(category_ids)

        seq_id_vals = torch.unique(seq_ids).tolist()
        seq_id_dict = {seq: idx for idx, seq in enumerate(seq_id_vals)}
        seq_ids = seq_ids.apply_(seq_id_dict.get)

        tstamp_temp = [tstamp[i] for i in range(len(event_type_seq)) if event_type_seq[i] == 'Text']
        #
        cutoffs = get_cutoffs(t0=timestamps[0], timestamps=tstamp_temp, sequences=final_chunks, mode="train")

        if make_positional:
            pos_encodings, rev_encodings = make_positional_event(tstamp, event_type_seq, embedding_dim=768)
            return (event_type_seq, sequences, seq_tokenized, tstamp, final_input_ids, final_attention_mask, seq_ids,
                    category_ids, cutoffs, pos_encodings, rev_encodings)

        else:
            return event_type_seq, sequences, seq_tokenized, tstamp, final_input_ids, final_attention_mask, seq_ids, category_ids, cutoffs, None, None
    else:
        # Mode inference, we take the 16 first chunks and fill with the previous events, then we take the next 16 chunks and
        # fill with the previous events (that are note notes) etc

        selected_chunks = txt_chunk_idx
        ids_selected = []
        for t in ids:
            _ = []
            for idx in t:
                if idx in selected_chunks:
                    _.append(idx)
            if len(_) > 0:
                ids_selected.append(_)

        final_input_ids = torch.cat(
            [new_sequence_tokenized[chunk][0].unsqueeze(0) for chunk in selected_chunks])  # this concatenates to (overall # chunks, 512)
        final_attention_mask = torch.cat([new_sequence_tokenized[chunk][1].unsqueeze(0) for chunk in selected_chunks])

        seq_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [[i] * len(ids[i]) for i in range(len(ids))]
                )
            )
        )  # Appartient au même document

        category_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [new_sequence_tokenized[chunk][2]]
                        for chunk in selected_chunks
                    ]
                )
            )
        )

        seq_ids = torch.LongTensor(seq_ids)
        category_ids = torch.LongTensor(category_ids)

        seq_id_vals = torch.unique(seq_ids).tolist()
        seq_id_dict = {seq: idx for idx, seq in enumerate(seq_id_vals)}
        seq_ids = seq_ids.apply_(seq_id_dict.get)

        event_type_seq, seq, seq_tokenized, tstamp = [], [], [], []

        for i in range(0, final_input_ids.shape[0], num_text_chunks):

            selected_temp = selected_chunks[i: i + num_text_chunks]
            event_type_seq_temp, seq_tokenized_temp, tstamp_temp = [], [], []

            t = 0
            while t <= selected_temp[len(selected_temp) - 1]:

                if new_event_type_sequence[t] == 'Text':
                    if t in selected_temp:
                        event_type_seq_temp.append(new_event_type_sequence[t])
                        seq_tokenized_temp.append(new_sequence_tokenized[t])
                        tstamp_temp.append(new_timestamps[t])
                else:
                    event_type_seq_temp.append(new_event_type_sequence[t])
                    seq_tokenized_temp.append(new_sequence_tokenized[t])
                    tstamp_temp.append(new_timestamps[t])
                t += 1
            event_type_seq.append(event_type_seq_temp)
            seq_tokenized.append(seq_tokenized_temp)
            tstamp.append(tstamp_temp)


        tstamp_temp = [new_timestamps[i] for i in range(len(new_timestamps)) if new_event_type_sequence[i] == 'Text']
        tstamp_temp = tstamp_temp[-181:]
        final_chunks = [new_sequence_tokenized[i] for i in range(len(new_timestamps)) if new_event_type_sequence[i] == 'Text']
        final_chunks = final_chunks[-181:]
        cutoffs = get_cutoffs(t0=timestamps[0], timestamps=tstamp_temp, sequences=final_chunks, mode="evalu")

        if make_positional:

            pos_encodings, rev_encodings = [], []

            t = 0
            for s in seq_tokenized:
                pos_encodings_, rev_encodings_ = make_positional_event(tstamp[t], event_type_seq, embedding_dim=768)
                pos_encodings.append(pos_encodings_)
                rev_encodings.append(rev_encodings_)
                t+=1
                # l = len(s)
                # pos_encodings.append(pos_encodings_[0:l])
                # rev_encodings.append(rev_encodings_[0:l])
                #l+=len(s)


            return (event_type_seq, sequences, seq_tokenized, tstamp, final_input_ids, final_attention_mask, seq_ids,
                    category_ids, cutoffs, pos_encodings, rev_encodings)

        else:
            return event_type_seq, sequences, seq_tokenized, tstamp, final_input_ids, final_attention_mask, seq_ids, category_ids, cutoffs, None, None


def train(model, dataloader, optimizer, lr_scheduler, config, device, mymetrics, tokenizer, scaler, grad_accumulation_steps):
    model.train()

    step = 0
    optimizer.zero_grad()
    start = timing.time()

    preds = {"hyps": [], "refs": [], "hyps_aux": [], "refs_aux": []}
    # add cls, aux, total keys to preds
    train_loss = {"loss_cls": [], "loss_aux": [], "loss_total": []}
    if config["evaluate_temporal"]:
        preds["hyps_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}
        preds["refs_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}

    hadm_ls_check = [182396, 144347, 177066, 109365]

    gates_weights_hadm = []
    for batch_idx, (hadm, event_type_sequence, sequences, sequences_tokenized, timestamps, labels) in enumerate(dataloader):
        # torch.cuda.reset_peak_memory_stats(device)
        event_type_seq, seq, seq_tokenized, tstamp, input_ids, attention_mask, seq_ids, category_ids, cutoffs, pos_encodings, rev_encodings \
            = select_sequence(event_type_sequence, sequences,  sequences_tokenized, timestamps, tokenizer, mode="train") # to change to train

        with torch.amp.autocast(device_type='cuda'):
            scores, output, gate_lists = model(event_types=event_type_seq, seq_tokenized=seq_tokenized, input_ids=input_ids.to(device, dtype=torch.long),
                attention_mask=attention_mask.to(device, dtype=torch.long),
                seq_ids=seq_ids.to(device, dtype=torch.long),
                category_ids=category_ids.to(device, dtype=torch.long),
                cutoffs=cutoffs, pos_encodings=pos_encodings.to(device, dtype=torch.long) if pos_encodings is not None else None,
                rev_encodings=rev_encodings.to(device, dtype=torch.long) if rev_encodings is not None else None)
            #
            # train with loss on last temporal point only
            loss_cls = F.binary_cross_entropy_with_logits(
                scores[-1, :][None, :],
                torch.tensor(labels).to(device, dtype=torch.float)[None, :],
            )
            loss_aux = torch.tensor(0)

            loss = loss_cls + loss_aux

        scaler.scale(loss).backward()

        train_loss["loss_cls"].append(loss_cls.detach().cpu().numpy())
        train_loss["loss_aux"].append(loss_aux.detach().cpu().numpy())
        train_loss["loss_total"].append(loss.detach().cpu().numpy())
        # convert to probabilities
        probs = F.sigmoid(scores)
        # print(f"probs shape: {probs.shape}")
        # print(f"cutoffs: {cutoffs}")
        preds["hyps"].append(probs[-1, :].detach().cpu().numpy())
        labels = torch.tensor(labels).to(device, dtype=torch.float)
        preds["refs"].append(labels.detach().cpu().numpy())

        if config["evaluate_temporal"]:
            cutoff_times = ["2d", "5d", "13d", "noDS"]
            for n, time in enumerate(cutoff_times):
                if cutoffs[time][0] != -1:
                    if config["reduce_computation"]:
                        preds["hyps_temp"][time].append(
                            probs[n, :].detach().cpu().numpy()
                        )
                    else:
                        preds["hyps_temp"][time].append(
                            probs[cutoffs[time][0], :]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    preds["refs_temp"][time].append(
                        labels.detach().cpu().numpy()
                    )
        #

        if ((batch_idx + 1) % grad_accumulation_steps == 0) or (
                batch_idx + 1 == len(dataloader)
        ):
            # Log gradients & weights
            """for name, param in model.named_parameters():
                if param.requires_grad:
                    wandb.log({
                        f"grad_norm/{name}": param.grad.norm().item() if param.grad is not None else 0.0,
                        f"weight_norm/{name}": param.data.norm().item()
                    }, step=batch_idx + 1)"""

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()
            # break

        if hadm in hadm_ls_check:
            if gate_lists is not None and gate_lists.dim() == 2:  # [num_chunks, num_events]
                temp = []
                for idx in [3, 10]:
                    if idx < gate_lists.size(0):
                        temp.append(gate_lists[idx].detach().cpu().numpy())
                        """weights = gate_lists[idx].detach().cpu().numpy()  # [num_events]

                        # W&B heatmap-friendly format (1D reshaped to 2D if needed)
                        fig = plt.figure(figsize=(10, 2))
                        sns.heatmap(weights[None, :], cmap="viridis", cbar=True)
                        plt.title(f"{hadm} Gating weights for chunk {idx}")
                        plt.xlabel("Event Index")
                        plt.yticks([])

                        # Log to wandb
                        wandb.log({f"attention_weights/chunk_{idx}": wandb.Image(fig)}, step=batch_idx + 1)
                        plt.close(fig)"""
                gates_weights_hadm.append(temp)
        if batch_idx % 500 == 0:
            print(f"made {batch_idx} visits in {timing.time()-start}")


    end_time_train = timing.time()
    print(f"time taken for 1 epoch of training : {end_time_train - start}\n")
    print(f"peak memory used in training : {torch.cuda.max_memory_allocated(device) / (1024 ** 2)} MB")
    # print(f"Batch {batch_idx}, Step {step}, Loss: {loss.item():.4f}")

    """Evaluate model on validation set and save Saved_models to csv."""
    train_metrics = mymetrics.from_numpy(
        np.asarray(preds["hyps"]), np.asarray(preds["refs"])
    )
    cutoff_times = ["2d", "5d", "13d", "noDS"]
    train_metrics_temp = {}
    if config["evaluate_temporal"]:
        train_metrics_temp = {
            time: mymetrics.from_numpy(
                np.asarray(preds["hyps_temp"][time]),
                np.asarray(preds["refs_temp"][time]),
            )
            for time in cutoff_times
        }

    train_metrics["loss"] = np.mean(train_loss["loss_total"])
    train_metrics["loss_aux"] = np.mean(train_loss["loss_aux"])
    train_metrics["loss_cls"] = np.mean(train_loss["loss_cls"])

    return model, train_metrics, train_metrics_temp, preds, hadm_ls_check, gates_weights_hadm


def inference(model, dataloader, device, config, mymetrics, tokenizer=None):
    model.eval()
    all_predictions = []
    loss_cls = 0
    with torch.no_grad():
        ids = []
        preds = {"hyps": [], "refs": [], "hyps_aux": [], "refs_aux": []}
        preds["hyps_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}
        preds["refs_temp"] = {"2d": [], "5d": [], "13d": [], "noDS": []}

        start_time_test = timing.time()
        for batch_idx, (hadm, event_type_sequence, sequences, sequences_tokenized, timestamps, labels) in enumerate(
               dataloader):
            torch.cuda.reset_peak_memory_stats(device)

            event_type_seq, seq, seq_tokenized, tstamp, input_ids, attention_mask, seq_ids, category_ids, cutoffs, pos_encodings, rev_encodings \
                = select_sequence(event_type_sequence, sequences, sequences_tokenized, timestamps, tokenizer,
                                  mode="eval")  # to change to train

            complete_sequence_output = []
            # run through data in chunks of max_chunks
            t = 0
            for i in range(0, input_ids.shape[0], model.max_chunks):
                # only get the document embeddings
                sequence_output, gate_lists = model(
                    event_types=event_type_seq[t], seq_tokenized=seq_tokenized[t],
                    input_ids=input_ids[i: i + model.max_chunks].to(
                        device, dtype=torch.long
                    ),
                    attention_mask=attention_mask[i: i + model.max_chunks].to(
                        device, dtype=torch.long
                    ),
                    seq_ids=seq_ids[i: i + model.max_chunks].to(
                        device, dtype=torch.long
                    ),
                    category_ids=category_ids[i: i + model.max_chunks].to(
                        device, dtype=torch.long
                    ),
                    cutoffs=None,  # None,  # cutoffs, #None,
                    is_evaluation=True,
                    pos_encodings=pos_encodings[t].to(
                        device, dtype=torch.long
                    ) if pos_encodings is not None else None,
                    rev_encodings=rev_encodings[t].to(
                        device, dtype=torch.long
                    ) if rev_encodings is not None else None
                    # note_end_chunk_ids=note_end_chunk_ids,
                )
                complete_sequence_output.append(sequence_output)
                t += 1
            # concatenate the sequence output
            sequence_output = torch.cat(complete_sequence_output, dim=0)

            # run through LWAN to get the scores
            scores = model.label_attn(sequence_output, cutoffs=cutoffs)

            labels = torch.tensor(labels).to(device, dtype=torch.float)

            loss_cls += F.binary_cross_entropy_with_logits(
                    scores[-1, :][None, :],
                    labels.to(device)[None, :],
            ).item()

            # print(f"Inference Batch {batch_idx}, {len(seqs)} sequences processed")
            probs = F.sigmoid(scores)
            ids.append(hadm)
            preds["hyps"].append(probs[-1, :].detach().cpu().numpy())

            preds["refs"].append(labels.detach().cpu().numpy())
            if config["evaluate_temporal"]:
                cutoff_times = ["2d", "5d", "13d", "noDS"]
                for n, time in enumerate(cutoff_times):
                    if cutoffs[time][0] != -1:
                        if config["reduce_computation"]:
                            preds["hyps_temp"][time].append(
                                probs[n, :].detach().cpu().numpy()
                            )
                        else:
                            preds["hyps_temp"][time].append(
                                probs[cutoffs[time][0], :].detach().cpu().numpy()
                            )
                        preds["refs_temp"][time].append(labels.detach().cpu().numpy())
            """if (batch_idx % 16) == 0 and batch_idx !=0 :
                break"""

            # print(f"loging attention weights for hadm {hadm}")

            if batch_idx % 500 == 0:
                print(f"made {batch_idx} visits in {timing.time() - start_time_test}")
        end_time_test = timing.time()

        print(f"time taken for 1 epoch of validation : {end_time_test - start_time_test}\n")
        print(f"peak memory used in training : {torch.cuda.max_memory_allocated(device) / (1024 ** 2)} MB")

    pred_cutoff = 0.5

    val_metrics = mymetrics.from_numpy(
        np.asarray(preds["hyps"]),
        np.asarray(preds["refs"]),
        pred_cutoff=pred_cutoff,
    )

    val_metrics["val_loss_cls"] = loss_cls / batch_idx


    if config["evaluate_temporal"]:
        cutoff_times = ["2d", "5d", "13d", "noDS"]
        val_metrics_temp = {
            time: mymetrics.from_numpy(
                np.asarray(preds["hyps_temp"][time]),
                np.asarray(preds["refs_temp"][time]),
                pred_cutoff=pred_cutoff,
            )
            for time in cutoff_times
        }
    else:
        val_metrics_temp = None

    return val_metrics, val_metrics_temp, mymetrics