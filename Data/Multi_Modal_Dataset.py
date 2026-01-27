from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import torch
from Multi_Modal_Continuous.utils import get_cutoffs, random_sample_chronological
from Multi_Modal_Continuous.Models.event_transformer_icd import *
from collections import Counter
import numpy as np
import pickle
import json
import os
import time
import ast
from joblib import Parallel, delayed
import re
import itertools
from torch.utils.data._utils.collate import default_collate


def tolerant_collate(batch):
    # Remove None entries

    """batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None  # or raise an error, or return a default value
    for s in range(len(batch[0][1])):
        for t in range(len(batch[0][1][s])):
            if batch[0][1][s][t] is None:
                batch[0][1][s][t] = "None"
    """

    (hadm, event_type_sequence, sequences, sequences_tokenized, timestamps, labels) = map(list, zip(*batch))

    return hadm[0], event_type_sequence[0], sequences[0], sequences_tokenized[0], timestamps[0], labels[
        0]


def parse_list_of_strings(s):
    if pd.isna(s):
        return []
    return re.findall(r"'(.*?)'", s)


def parse_list_of_lists(s):
    if pd.isna(s):
        return []
    outer = re.findall(r"\[(.*?)\]", s)
    result = []
    for inner in outer:
        # Find strings and numeric fragments
        elements = re.findall(r"'(.*?)'|([0-9\.\-eE]+)", inner)
        flat = [a if a else b for a, b in elements]
        # Try to cast to float or int if possible
        parsed = []
        for x in flat:
            try:
                if '.' in x or 'e' in x.lower():
                    parsed.append(float(x))
                else:
                    parsed.append(int(x))
            except:
                parsed.append(x)
        result.append(parsed)
    return result


def make_json_serializable(obj):
    """
    Recursively convert dict keys to str and cast types like np.int64 to int, etc.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, (int, float, str, type(None))):
        return obj
    elif hasattr(obj, 'tolist'):  # e.g., numpy arrays
        return obj.tolist()
    else:
        return str(obj)  # fallback: convert to string


class Multi_Modal_Dataset(Dataset):
    """Multi_Modal_Dataset for real-time ICD-9 code prediction.

    The complete EHR sequence is very long, therefore the custom dataset
    allows for multiple configurations to select the chunks to use.

    In particular, the following set-ups are available:
    - latest: only the last max_chunks chunks are used
    - uniform: the first and last note are always used, and the remaining
    chunks are randomly sampled (this is the "random" set-up in the paper)
    - random: all notes are used (this is the "complete EHR set-up in the paper)
    - limit_ds: limit DS to 4 chunks (only for the "random" setup, corresponding to
        the Limited DS set-up in the paper)

    The elements of the resulting dataset include:
    - input_ids: tokenized input
    - attention_mask: attention mask
    - seq_ids: sequence ids
    - category_ids: category ids
    - label: ICD-9 code
    - hadm_id: HADM_ID
    - hours_elapsed: hours elapsed since admission
    - cutoffs: cutoffs for the different time windows (2d, 5d, 13d, noDS, all
    - percent_elapsed: percentage of time elapsed since admission (only for the latest setup
    """

    def __init__(
            self, **kwargs
    ):
        self.name = kwargs["name"]
        self.file_path = kwargs["file_path"]
        self.splits = kwargs["splits"]
        self.tokenizer = kwargs["tokenizer"]
        self.saved_path = kwargs["saved_path"]
        self.token_dicts = kwargs["token_dicts"]

        load_path = self.saved_path + self.name + '_dataset.pkl'

        if os.path.exists(load_path):
            print("loading dataset from {}".format(self.saved_path + self.name))
            self.load_dataset(load_path)
        else:

            self.event_type_sequence = []
            self.sequences = []
            self.sequences_tokenized = []
            self.timestamps = []
            self.hadm_ids = []
            self.labels = []

            self._vocabulary_icd()
            self._prepare_data()

            self.save_dataset(self.saved_path + self.name + '_dataset.pkl')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        """txt = [self.sequences[idx][i] for i in range(len(self.event_type_sequence[idx])) if
               self.event_type_sequence[idx][i] == 'Text']

        output = [self.tokenizer(doc[0],
                                 truncation=True,
                                 return_overflowing_tokens=True,
                                 padding="max_length",
                                 return_tensors="pt") for doc in txt]

        input_ids = torch.cat(
            [doc["input_ids"] for doc in output]
        )  # this concatenates to (overall # chunks, 512)
        attention_mask = torch.cat([doc["attention_mask"] for doc in output])
        seq_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [[i] * len(output[i]["input_ids"]) for i in range(len(output))]
                )
            )
        )  # Appartient au même document
        category_ids = np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [txt[i][2]] * len(output[i]["input_ids"])
                        for i in range(len(output))
                    ]
                )
            )
        )"""

        return (self.hadm_ids[idx], self.event_type_sequence[idx], self.sequences[idx], self.sequences_tokenized[idx],
                self.timestamps[idx], self.labels[idx])

    def _prepare_data(self):
        print("reading in events..")
        # df = pd.read_csv(self.file_path)
        df = pd.read_json(self.file_path, orient="records", lines=True)
        split = pd.read_csv(self.splits)
        split = split[split["SPLIT_50"] == self.name]

        df = df[df.HADM_ID.isin(split.HADM_ID.unique())]
        print("tokenizing the events...")
        for hadm_id in tqdm(df.HADM_ID.unique()):
            df_hadm = df[df["HADM_ID"] == hadm_id]
            event_type_sequence = []
            sequences = []
            sequences_tokenized = []
            timestamps = []
            labels = np.zeros(50)
            self.hadm_ids.append(hadm_id)
            for _, row in df_hadm.iterrows():
                for i in range(len(row["CHARTTIME"])):
                    event_type_sequence.append(row["EVENT_TYPE"][i])
                    timestamps.append(pd.to_datetime(row["CHARTTIME"][i], unit='ms'))
                    sequences.append(row["EVENT_INFO"][i])
                    if row["EVENT_TYPE"][i] == "Lab":
                        sequences_tokenized.append(
                            [self.token_dicts['lab_ITEMID_tok2ind'][str(row["EVENT_INFO"][i][0])],
                             self.token_dicts['lab_FLAG_tok2ind'][str(row["EVENT_INFO"][i][1])]])
                    elif row["EVENT_TYPE"][i] == "Drug":
                        sequences_tokenized.append(
                            [self.token_dicts['drug_GSN_tok2ind'][str(row["EVENT_INFO"][i][0])],
                             self.token_dicts['drug_ROUTE_tok2ind'][str(row["EVENT_INFO"][i][1])]])
                    elif row["EVENT_TYPE"][i] == "Microbio":
                        sequences_tokenized.append(
                            [self.token_dicts['micro_SPEC_ITEMID_tok2ind'][str(row["EVENT_INFO"][i][0])],
                             self.token_dicts['micro_ORG_ITEMID_tok2ind'][str(row["EVENT_INFO"][i][1])],
                             self.token_dicts['micro_AB_ITEMID_tok2ind'][str(row["EVENT_INFO"][i][2])],
                             self.token_dicts['micro_INTERPRETATION_tok2ind'][str(row["EVENT_INFO"][i][3])]])
                    elif row["EVENT_TYPE"][i] == "Text":
                        data = self.tokenizer(row["EVENT_INFO"][i][0],
                                              truncation=True,
                                              return_overflowing_tokens=True,
                                              padding="max_length",
                                              return_tensors="pt",
                                              )
                        input_ids = []
                        attn_masks = []
                        for j in range(len(data["input_ids"])):
                            input_ids.append(data["input_ids"][j])
                            attn_masks.append(data["attention_mask"][j])
                        #  row["EVENT_INFO"][2] is CATEGORY_INDEX and row["EVENT_INFO"][3] is CATEGORY_REVERSE_SEQID
                        sequences_tokenized.append(
                            [input_ids, attn_masks, row["EVENT_INFO"][i][2], row["EVENT_INFO"][i][3]])

            # Only keep the codes that are in the top50
            for icd in set(eval(split[split["HADM_ID"] == hadm_id]["absolute_code"].item())).intersection(
                    set(self.c2ind.keys())):
                idx = self.c2ind[icd]
                labels[idx] = 1

            self.sequences_tokenized.append(sequences_tokenized)
            self.sequences.append(sequences)
            self.event_type_sequence.append(event_type_sequence)
            self.timestamps.append(timestamps)
            self.labels.append(labels)

    def _vocabulary_icd(self):
        # Read the CSV
        split = pd.read_csv(self.splits)
        unique_splits = split.SPLIT_50.unique()

        # Collect all ICD codes into a list for counting
        icd_counter = Counter()
        for spl in tqdm(unique_splits):
            temp = split[split["SPLIT_50"] == spl]
            print(f"fetching the codes in {spl}")
            for hadm_id in temp["HADM_ID"].unique():
                icds_hadm = eval(temp[temp["HADM_ID"] == hadm_id].absolute_code.iloc[0])
                icd_counter.update(icds_hadm)

        # Get the top 50 most common ICD codes
        # Correct
        top_50_icds = [icd for icd, _ in icd_counter.most_common(50)]

        # Build mapping dicts only for the top 50
        self.c2ind = {}
        self.ind2c = {}
        for i, icd in enumerate(top_50_icds):
            self.c2ind[icd] = i
            self.ind2c[i] = icd

        print("done")

    def save_dataset(self, pickle_path):
        """
        Save features, labels, and hadm_ids into a pickle file,
        and save all vocabularies/mappings into a JSON file.
        """

        # Ensure directories exist
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        print("saving file in {}...".format(pickle_path))
        data = {
            "event_type_sequence": self.event_type_sequence,
            "sequences": self.sequences,
            "sequences_tokenized": self.sequences_tokenized,
            "timestamps": self.timestamps,
            "hadm_ids": self.hadm_ids,
            "labels": self.labels
        }
        start = time.time()
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
        print(f"saved {pickle_path} in {time.time() - start} seconds")

    def load_dataset(self, filepath):
        print("loading file in {}...".format(filepath))
        start = time.time()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"loaded {filepath} in {time.time() - start} seconds")

        self.event_type_sequence = data["event_type_sequence"]
        self.sequences = data["sequences"]
        self.sequences_tokenized = data["sequences_tokenized"]
        self.timestamps = data["timestamps"]
        self.hadm_ids = data["hadm_ids"]
        self.labels = data["labels"]
