import pandas as pd
import os
from  FINAL.pj_utils import build_token_dicts, save_token_dicts_to_json
from tqdm import tqdm

class DataProcessor:
    """Prepare data for model.

    Some of the functions of this class are extracted from the HTDC (Ng et al, 2023):
    aggregate_hadm_id, add_category_information and multi_hot_encode.

    We added the _add_temporal_information function to add time information,
    which is used for the temporal experiments for real-time ICD-9 coding.
    """

    def __init__(self, dataset_path, note_path, config):
        # self.notes_df = pd.read_csv(os.path.join(dataset_path, "NOTEEVENTS.csv.gz"), compression="gzip", low_memory=False)
        print("reading notes...")
        self.notes_df = pd.read_csv(os.path.join(note_path, "NOTEEVENTS.csv.gz"),compression='gzip')
        print("reading lab, micro and drug events...")
        self.lab_df = pd.read_csv(os.path.join(dataset_path, "LABEVENTS.csv.gz"), compression='gzip')
        self.micro_df = pd.read_csv(os.path.join(dataset_path, "MICROBIOLOGYEVENTS.csv.gz"), compression='gzip')
        self.drug_df = pd.read_csv(os.path.join(dataset_path, "PRESCRIPTIONS.csv.gz"), compression='gzip')
        self.labels_df = pd.read_csv(
            config["splits"]
        )
        self.config = config

    def _prepare_data(self):
        notes_agg_df, lab_agg_df, micro_agg_df, drug_agg_df = self._aggregate_data()
        notes_agg_df, categories_mapping = self.add_category_information(notes_agg_df)
        events_agg_df = self._aggregate_events(notes_agg_df, lab_agg_df, micro_agg_df, drug_agg_df)

    def _aggregate_data(self):
        self.notes_df = self.notes_df[self.notes_df.HADM_ID.isna() == False]
        self.lab_df = self.lab_df[self.lab_df.HADM_ID.isna() == False]
        self.micro_df = self.micro_df[self.micro_df.HADM_ID.isna() == False]


        self.drug_df = (self.drug_df[self.drug_df.HADM_ID.isna() == False].dropna(subset=['HADM_ID', 'GSN']))

        self.notes_df["HADM_ID"] = self.notes_df["HADM_ID"].apply(int)
        self.lab_df["HADM_ID"] = self.lab_df["HADM_ID"].apply(int)
        self.micro_df["HADM_ID"] = self.micro_df["HADM_ID"].apply(int)
        self.drug_df["HADM_ID"] = self.drug_df["HADM_ID"].apply(int)

        # no nan values so we convert directly to datetime
        self.lab_df["CHARTTIME"] = pd.to_datetime(self.lab_df.CHARTTIME)

        # if time is missing -> assume 00:10:00
        self.micro_df["CHARTTIME"] = self.micro_df.CHARTTIME.fillna(
            pd.to_datetime(self.micro_df["CHARTDATE"]) + pd.Timedelta("00:10:00")
        )
        self.micro_df["CHARTTIME"] = pd.to_datetime(self.micro_df.CHARTTIME)

        # only keep records of drugs that have a gsn
        self.drug_df = self.drug_df[self.drug_df.ICUSTAY_ID.isna() == False].drop_duplicates(
            subset=["HADM_ID", "ICUSTAY_ID", "STARTDATE", "ENDDATE", "GSN", "ROUTE"])

        self.drug_df["CHARTTIME"] = pd.to_datetime(self.drug_df["STARTDATE"]) + pd.Timedelta("00:20:00")
        """merged_drug_df = self.drug_df.merge(
            self.transfers[["ICUSTAY_ID", "INTIME"]],
            on="ICUSTAY_ID",
            how="left"
        )

        # Ensure INTIME is datetime
        merged_drug_df["INTIME"] = pd.to_datetime(merged_drug_df["INTIME"])

        # Compute CHARTTIME as INTIME + 10 minutes
        merged_drug_df["CHARTTIME"] = merged_drug_df["INTIME"] + pd.Timedelta(minutes=10)

        # Drop the now-redundant INTIME column if you don't need it anymore
        merged_drug_df = merged_drug_df.drop(columns=["INTIME"])"""

        # if time is missing -> assume 12:00:00
        self.notes_df.CHARTTIME = self.notes_df.CHARTTIME.fillna(
            self.notes_df.CHARTDATE + " 12:00:00"
        )
        self.notes_df["CHARTTIME"] = pd.to_datetime(self.notes_df.CHARTTIME)

        self.notes_df["is_discharge_summary"] = (
                self.notes_df.CATEGORY == "Discharge summary"
        )
        notes_agg_df = (
            self.notes_df.sort_values(
                by=["CHARTDATE", "CHARTTIME", "is_discharge_summary"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg({"TEXT": list, "CHARTDATE": list, "CHARTTIME": list, "CATEGORY": list})
        ).reset_index()

        lab_agg_df = (
            self.lab_df.sort_values(
                by=["CHARTTIME"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg({"ITEMID": list, "FLAG": list, "CHARTTIME": list})
        ).reset_index()

        micro_agg_df = (
            self.micro_df.sort_values(
                by=["CHARTTIME"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg(
                {"SPEC_ITEMID": list, "ORG_ITEMID": list, "AB_ITEMID": list, 'INTERPRETATION': list, "CHARTTIME": list})
        ).reset_index()

        drug_agg_df = (
            self.drug_df.sort_values(
                by=["CHARTTIME"],
                na_position="last",
            )
            .groupby(["SUBJECT_ID", "HADM_ID"])
            .agg({"GSN": list, "ROUTE": list, "CHARTTIME": list})
        ).reset_index()

        # Modify the TEXT, CHARTDATE, CHARTTIME, CATEGORY columns
        # by limiting the list to include all elements until the first DS
        # the apply function is used to apply the lambda function to each row
        # and it should filter based on the CATEGORY column
        notes_agg_df["TEXT"] = notes_agg_df[["TEXT", "CATEGORY"]].apply(
            lambda x: x.TEXT[: x.CATEGORY.index("Discharge summary") + 1]
            if "Discharge summary" in x.CATEGORY
            else x.TEXT,
            axis=1,
        )

        notes_agg_df["CHARTDATE"] = notes_agg_df[["CHARTDATE", "CATEGORY"]].apply(
            lambda x: x.CHARTDATE[: x.CATEGORY.index("Discharge summary") + 1]
            if "Discharge summary" in x.CATEGORY
            else x.CHARTDATE,
            axis=1,
        )
        notes_agg_df["CHARTTIME"] = notes_agg_df[["CHARTTIME", "CATEGORY"]].apply(
            lambda x: x.CHARTTIME[: x.CATEGORY.index("Discharge summary") + 1]
            if "Discharge summary" in x.CATEGORY
            else x.CHARTTIME,
            axis=1,
        )
        notes_agg_df["CATEGORY"] = notes_agg_df["CATEGORY"].apply(
            lambda x: x[: x.index("Discharge summary") + 1]
            if "Discharge summary" in x
            else x,
        )

        # Aggregate with the labels df
        notes_agg_df = notes_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")
        lab_agg_df = lab_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")
        micro_agg_df = micro_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")
        drug_agg_df = drug_agg_df.merge(self.labels_df, on=["HADM_ID"], how="left")

        # Keep only rows for top 50 ICD9 codes
        notes_agg_df = notes_agg_df[notes_agg_df.SPLIT_50.isna() != True]
        lab_agg_df = lab_agg_df[lab_agg_df.SPLIT_50.isna() != True]
        micro_agg_df = micro_agg_df[micro_agg_df.SPLIT_50.isna() != True]
        drug_agg_df = drug_agg_df[drug_agg_df.SPLIT_50.isna() != True]

        return notes_agg_df, lab_agg_df, micro_agg_df, drug_agg_df

    def _func(self, x):
        if x == 0:
            return 2
        else:
            return x

    def _build_dicts(self):
        labs_df = self.lab_df[self.lab_df.HADM_ID.isin(self.labels_df.HADM_ID.unique())]
        drug_df = self.drug_df[self.drug_df.HADM_ID.isin(self.labels_df.HADM_ID.unique())]
        micro_df = self.micro_df[self.micro_df.HADM_ID.isin(self.labels_df.HADM_ID.unique())]
        token_dicts = build_token_dicts(labs_df, drug_df, micro_df)
        save_token_dicts_to_json(token_dicts, filepath=self.config["event_tokens_file"])
        return token_dicts

    def _get_reverse_seqid_by_category(self, category_ids):
        # This creates the CATEGORY_REVERSE_SEQID field for use in note selection later
        # For each category, the last note is assigned to index 0, the second last note is assigned index 1, and so on
        category_ids = pd.Series(category_ids)
        category_ranks = category_ids.groupby(category_ids).cumcount(ascending=False)
        return list(category_ranks)

    def add_category_information(self, notes_agg_df):
        # Create Category IDs
        categories = list(
            notes_agg_df["CATEGORY"]
            .apply(lambda x: pd.Series(x))
            .stack()
            .value_counts()
            .index
        )
        categories_mapping = {categories[i]: i for i in range(len(categories))}
        print(categories_mapping)

        notes_agg_df["CATEGORY_INDEX"] = notes_agg_df["CATEGORY"].apply(
            lambda x: [categories_mapping[c] for c in x]
        )

        # The "Nursing/Other" category is present in the train set but not the dev/test sets
        # We group them together with notes in the "Nursing" category as described in our paper

        notes_agg_df["CATEGORY_INDEX"] = notes_agg_df["CATEGORY_INDEX"].apply(
            lambda x: [self._func(y) for y in x]
        )

        notes_agg_df["CATEGORY_REVERSE_SEQID"] = notes_agg_df["CATEGORY_INDEX"].apply(
            self._get_reverse_seqid_by_category
        )
        return notes_agg_df, categories_mapping

    def _aggregate_events(self, notes_agg_df, lab_agg_df, micro_agg_df, drug_agg_df):

        all_events = []
        for hadm_id in tqdm(self.labels_df.HADM_ID.dropna().unique()):
            # --- Extract per-HADM_ID events ---
            notes_hadm = notes_agg_df[notes_agg_df["HADM_ID"] == hadm_id].copy()
            lab_hadm = lab_agg_df[lab_agg_df["HADM_ID"] == hadm_id].copy()
            micro_hadm = micro_agg_df[micro_agg_df["HADM_ID"] == hadm_id].copy()
            drug_hadm = drug_agg_df[drug_agg_df["HADM_ID"] == hadm_id].copy()

            # --- Format each type ---
            events = []

            # LAB EVENTS
            for _, row in lab_hadm.iterrows():
                for i in range(len(row["CHARTTIME"])):
                    events.append({
                        "HADM_ID": hadm_id,
                        "EVENT_TYPE": "Lab",
                        "CHARTTIME": row["CHARTTIME"][i],
                        "EVENT_INFO": [row["ITEMID"][i], row["FLAG"][i]]
                    })

            # DRUG EVENTS
            for _, row in drug_hadm.iterrows():
                for i in range(len(row["CHARTTIME"])):
                    events.append({
                        "HADM_ID": hadm_id,
                        "EVENT_TYPE": "Drug",
                        "CHARTTIME": row["CHARTTIME"][i],
                        "EVENT_INFO": [row["GSN"][i], row["ROUTE"][i]]
                    })

            # MICROBIO EVENTS
            for _, row in micro_hadm.iterrows():
                for i in range(len(row["CHARTTIME"])):
                    events.append({
                        "HADM_ID": hadm_id,
                        "EVENT_TYPE": "Microbio",
                        "CHARTTIME": row["CHARTTIME"][i],
                        "EVENT_INFO": [
                            row["SPEC_ITEMID"][i],
                            row["ORG_ITEMID"][i],
                            row["AB_ITEMID"][i],
                            row["INTERPRETATION"][i]
                        ]
                    })

            # TEXT EVENTS
            for _, row in notes_hadm.iterrows():
                for i in range(len(row["CHARTTIME"])):
                    events.append({
                        "HADM_ID": hadm_id,
                        "EVENT_TYPE": "Text",
                        "CHARTTIME": row["CHARTTIME"][i],
                        "EVENT_INFO": [
                            row["TEXT"][i],
                            row["CATEGORY"][i],
                            row["CATEGORY_INDEX"][i],
                            row["CATEGORY_REVERSE_SEQID"][i]
                        ]
                    })
            # Sort events by time
            events_sorted = sorted(events, key=lambda x: x["CHARTTIME"])
            all_events.extend(events_sorted)

        # Combine all HADM_ID event logs into a full dataframe
        event_df = pd.DataFrame(all_events)
        # Assumes event_df is already sorted by CHARTTIME and contains HADM_ID, EVENT_TYPE, CHARTTIME, EVENT_INFO
        final_hadm_rows = []

        for hadm_id, group in tqdm(event_df.groupby("HADM_ID")):
            sorted_group = group.sort_values("CHARTTIME")
            final_hadm_rows.append({
                "HADM_ID": hadm_id,
                "EVENT_TYPE": sorted_group["EVENT_TYPE"].tolist(),
                "CHARTTIME": sorted_group["CHARTTIME"].tolist(),
                "EVENT_INFO": sorted_group["EVENT_INFO"].tolist()
            })

        # Create the final dataframe: one row per HADM_ID
        print(f"saving events file to {self.config['file_path']}")
        self.final_hadm_df = pd.DataFrame(final_hadm_rows)
        self.final_hadm_df.to_json(self.config["file_path"], orient="records", lines=True)
