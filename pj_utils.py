import transformers
import random
import pandas as pd
import json

def save_token_dicts_to_json(token_dicts, filepath="token_dicts.json"):
    # Convert all keys in ind2tok dicts to string (for JSON compatibility)
    serializable_dicts = {}
    for key, d in token_dicts.items():
        if "ind2tok" in key:
            # Convert integer keys to strings
            d = {str(k): v for k, v in d.items()}
        serializable_dicts[key] = d

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_dicts, f, indent=4, ensure_ascii=False)


def get_tokenizer(checkpoint):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, model_max_length=512
    )
    return tokenizer

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits,
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def get_cutoffs(admit_time, timestamps):
    """
    admit_time: datetime (ADMITTIME of the admission)
    timestamps: list of datetime timestamps (per fused vector)
    category_ids: list of category_id (e.g., from note=5, other=0-4)

    Returns: dict with cutoff indices {"2d": idx, "5d": idx, ...}
    """
    cutoffs = {"2d": -1, "5d": -1, "13d": -1, "noDS": -1, "all": -1}
    for i, ts in enumerate(timestamps):
        hours_elapsed = (ts - admit_time).total_seconds() / 3600.0


        if hours_elapsed < 2 * 24:
            cutoffs["2d"] = i
        if hours_elapsed < 5 * 24:
            cutoffs["5d"] = i
        if hours_elapsed < 13 * 24:
            cutoffs["13d"] = i
        cutoffs["noDS"] = i

        cutoffs["all"] = i  # always update to last index

    return cutoffs


def random_sample_chronological(timestamps, sample_size):
    """
    Randomly sample timestamps but return them sorted chronologically.

    Args:
        timestamps (list): List of timestamps.
        sample_size (int): Number of samples to pick.

    Returns:
        list: Randomly sampled timestamps in chronological order.
    """
    if sample_size >= len(timestamps):
        return sorted(timestamps)

    sampled = random.sample(timestamps, sample_size)
    return sorted(sampled)


def build_token_dicts(lab_df, drug_df, micro_df):
    from collections import defaultdict

    def extract_tokens(values):
        # Drop NaNs, convert to str, remove duplicates, sort
        return values.unique().tolist()

    def build_dict(tokens):
        tok2ind = {tok: idx for idx, tok in enumerate(tokens)}
        ind2tok = {idx: tok for tok, idx in tok2ind.items()}
        return tok2ind, ind2tok

    token_dicts = {}

    # LAB: ITEMID, FLAG
    lab_itemid_tokens = extract_tokens(lab_df["ITEMID"])
    lab_flag_tokens = extract_tokens(lab_df["FLAG"])

    token_dicts["lab_ITEMID_tok2ind"], token_dicts["lab_ITEMID_ind2tok"] = build_dict(lab_itemid_tokens)
    token_dicts["lab_FLAG_tok2ind"], token_dicts["lab_FLAG_ind2tok"] = build_dict(lab_flag_tokens)

    # DRUG: GSN, ROUTE
    drug_gsn_tokens = extract_tokens(drug_df["GSN"])
    drug_route_tokens = extract_tokens(drug_df["ROUTE"])

    token_dicts["drug_GSN_tok2ind"], token_dicts["drug_GSN_ind2tok"] = build_dict(drug_gsn_tokens)
    token_dicts["drug_ROUTE_tok2ind"], token_dicts["drug_ROUTE_ind2tok"] = build_dict(drug_route_tokens)

    # MICROBIO: SPEC_ITEMID, ORG_ITEMID, AB_ITEMID, INTERPRETATION
    for col in ["SPEC_ITEMID", "ORG_ITEMID", "AB_ITEMID", "INTERPRETATION"]:
        tokens = extract_tokens(micro_df[col])
        tok2ind, ind2tok = build_dict(tokens)
        token_dicts[f"micro_{col}_tok2ind"] = tok2ind
        token_dicts[f"micro_{col}_ind2tok"] = ind2tok

    return token_dicts

def load_token_dicts_from_json(filepath="token_dicts.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    token_dicts = {}
    for key, d in loaded.items():
        if "ind2tok" in key:
            # Convert keys back to integers
            d = {int(k): v for k, v in d.items()}
        token_dicts[key] = d

    return token_dicts
