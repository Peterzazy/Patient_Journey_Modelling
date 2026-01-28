import os
import json
from Data.Preprocessing.pj_preprocessor import DataProcessor
from pj_utils import load_token_dicts_from_json, get_tokenizer
from Data.Multi_Modal_Dataset import Multi_Modal_Dataset, tolerant_collate
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import pandas as pd
import numpy as np
import torch.cuda.amp as amp
from FINAL.Evaluation.metrics_pj import MyMetrics
from loop_pj import train, inference
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from FINAL.Models.PJ.models import Model

def build_optimizer_and_scheduler(model, config, steps_per_epoch, epoch):
    """
    Build optimizer and LR scheduler based on current epoch phase.
    Applies dynamic learning rates for different training stages.
    """
    # Phase-dependent learning rates
    if epoch < 5:
        lr = 3e-4  # higher LR for learning event embeddings
    elif epoch < 10:
        lr = 1e-5    # lower LR to carefully tune PLM
    else:
        lr = 1e-5  # balance for full fine-tuning

    # Filter trainable parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        three_phase=True,
        total_steps=config["max_epochs"] * steps_per_epoch,
    )

    return optimizer, scheduler

def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True

def update_trainable_params(model, epoch):
    """
    Freezes/unfreezes model parts based on epoch number.

    Phase 1 (epoch < 5): freeze PLM, train events + fusion
    Phase 2 (5 <= epoch < 10): unfreeze PLM, freeze event embedders
    Phase 3 (epoch >= 10): unfreeze all
    """

    if epoch < 5:
        print("🔒 Phase 1: Freeze text encoder, train event embedders + fusion")
        freeze(model.transformer)
        unfreeze(model.lab_embedder)
        unfreeze(model.drug_embedder)
        unfreeze(model.microbio_embedder)
        unfreeze(model.fusion_layer)
        unfreeze(model.label_attn)
        unfreeze(model.document_regressor)

    elif epoch < 10:
        print("🔁 Phase 2: Unfreeze text encoder, freeze event embedders")
        unfreeze(model.transformer)
        freeze(model.lab_embedder)
        freeze(model.drug_embedder)
        freeze(model.microbio_embedder)
        unfreeze(model.fusion_layer)
        unfreeze(model.label_attn)
        unfreeze(model.document_regressor)

    else:
        print("🚀 Phase 3: Fine-tune everything")
        unfreeze(model.transformer)
        unfreeze(model.lab_embedder)
        unfreeze(model.drug_embedder)
        unfreeze(model.microbio_embedder)
        unfreeze(model.fusion_layer)
        unfreeze(model.label_attn)
        unfreeze(model.document_regressor)

def load_model():
    current_state_dict = model.state_dict()

    print("loading model which is trained on notes only.. LAHST")
    checkpoint = torch.load(
        "/Saved_models/LAHST/BEST_MMULA_evaluate.pth",
        map_location="cuda")
    saved_state_dict = checkpoint["model_state_dict"]

    # Define shared components you want to load
    shared_prefixes = (
        "transformer",  # Text encoder
        "label_attn",  # Label attention
        "document_regressor",  # Doc regressor
        "pelookup", "reversepelookup",
        "delookup", "reversedelookup",
        "celookup"
    )

    # Build filtered dict
    filtered_state_dict = {
        k: v for k, v in saved_state_dict.items()
        if k.startswith(shared_prefixes) and k in current_state_dict
    }

    # Load into current model
    model.load_state_dict(filtered_state_dict, strict=False)
    # Define whitelist prefixes to keep trainable
    trainable_prefixes = (
        "fusion_layer",
        "lab_embedder",
        "drug_embedder",
        "microbio_embedder",
        "event_norm",
    )

    # Apply requires_grad
    for name, param in model.named_parameters():
        if name.startswith(trainable_prefixes):
            param.requires_grad = True
            print(f"✅ Trainable: {name}")
        else:
            param.requires_grad = False

def save_torch_model(model, optimizer, scaler, lr_scheduler, result, training_args, save_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "Saved_models": result,
            "config": config,
            "epochs": training_args["TOTAL_COMPLETED_EPOCHS"],
            "current_best": training_args["CURRENT_BEST"],
            "current_patience_count": training_args["CURRENT_PATIENCE_COUNT"],
        },
        save_path,
    )


def wandb_login(project_name, model_name):
    wandb.login(key="API/Key")
    wandb.init(project=project_name, name=model_name)
def save_results(train_metrics, validation_metrics, training_args, lr_scheduler, config, timeframe="all"
):
    """Save resulting metrics (train and val) to csv.
    The argument timeframe specifies the time frame used for evaluation."""
    a = {
        f"validation_{key}": validation_metrics[key]
        for key in validation_metrics.keys()
    }
    b = {f"train_{key}": train_metrics[key] for key in train_metrics.keys()}
    result = {**a, **b}

    # print(result)

    print(
        {
            k: result[k] if type(result[k]) != np.ndarray else {}
            for k in result.keys()
        }
    )
    result["epoch"] = training_args["TOTAL_COMPLETED_EPOCHS"]
    result["curr_lr"] = lr_scheduler.get_last_lr()
    result.update(config)  # add config fields
    result_list = {k: [v] for k, v in result.items()}
    df = pd.DataFrame.from_dict(result_list)  # convert to datframe

    results_path = os.path.join(
        config["project_path"],
        f"Saved_models/{config['run_name']}_{timeframe}.csv",
    )
    results_df = pd.read_csv(results_path)
    results_df = pd.concat((results_df, df), axis=0, ignore_index=True)
    results_df.to_csv(results_path)  # update Saved_models

    return result


def load_config(config_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    with open(config_path, "r") as f:
        config_data = json.load(f)

    return config_data

config_path = "/Users/p-a/PycharmProjects/ICD_Coding/FINAL/Configs/config_pj.json"

do_preprocessing = False



if __name__ == "__main__":
    config = load_config(config_path)

    if do_preprocessing:
        do = DataProcessor(dataset_path=config["mimic_dir"],note_path=config["note_path"], config=config)

        if not os.path.exists(config["event_file"]):
            do._prepare_data()

        if not os.path.exists(config["event_tokens_file"]):
            do._build_dicts()

            # THE USER MUST CONVERT THE NaN into the string "None" in the token_dicts.json !!!! code flaw

    print("loading tokens...")
    token_dicts = load_token_dicts_from_json(config["event_tokens_file"])
    tokenizer = get_tokenizer(config["base_checkpoint"])

    training_set = Multi_Modal_Dataset(name="TRAIN", file_path=config["file_path"], splits=config["splits"],
                                       token_dicts=token_dicts,
                                       mimic_dir=config["mimic_dir"], tokenizer=tokenizer,
                                       saved_path=config["saved_path"])
    validation_set = Multi_Modal_Dataset(name="VALIDATION", file_path=config["file_path"], splits=config["splits"],
                                         token_dicts=token_dicts,
                                         mimic_dir=config["mimic_dir"], tokenizer=tokenizer,
                                         saved_path=config["saved_path"])
    """test_set = Multi_Modal_Dataset(name="TEST",file_path=config["file_path"],splits=config["splits"],
                                        mimic_dir=config["mimic_dir"], tokenizer=tokenizer, saved_path=config["saved_path"])
    """

    train_loader = DataLoader(training_set, batch_size=config["batch_size"], shuffle=True, collate_fn=tolerant_collate)
    validation_loader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=False,
                                   collate_fn=tolerant_collate)

    device = torch.device("mps" if torch.backends.mps.is_available() else "else")

    train_loader = DataLoader(training_set, batch_size=config["batch_size"], shuffle=True, collate_fn=tolerant_collate)
    validation_loader = DataLoader(validation_set, batch_size=config["batch_size"], shuffle=False, collate_fn=tolerant_collate)

    device = torch.device("mps" if torch.backends.mps.is_available() else "else")
    model = Model(config, device=device)

    # param = view_model(model)
    model.to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param = trainable
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total} ({trainable / total:.2%})")

    steps_per_epoch = int(
        np.ceil(len(train_loader) / config["grad_accumulation"])
    )

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        three_phase=True,
        total_steps=config["max_epochs"] * steps_per_epoch,
    )
    scaler = amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = config["epochs"]
    my_metrics = MyMetrics(debug=config["debug"])

    training_args = {
        "TOTAL_COMPLETED_EPOCHS": 0,
        "CURRENT_BEST": 0,
        "CURRENT_PATIENCE_COUNT": 0,
    }

    cutoff_times = ["all", "2d", "5d", "13d", "noDS"]
    for time in cutoff_times:
        pd.DataFrame({}).to_csv(
            os.path.join(
                config["project_path"], f"Saved_models/{config['run_name']}_{time}.csv"
            )
        )  # Create dummy csv because of GDrive bug
    results = {}

    if config["load_from_checkpoint"]:
        checkpoint = torch.load(
            os.path.join(
                config["project_path"], f"Saved_models/{config['run_name']}.pth"
            )
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Move optimizer to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        training_args["TOTAL_COMPLETED_EPOCHS"] = checkpoint["epochs"]
        training_args["CURRENT_BEST"] = checkpoint["current_best"]
        training_args["CURRENT_PATIENCE_COUNT"] = checkpoint["current_patience_count"]

    # wandb_login("Patient Journey Modelling", model_name="Multi_Modal_Differential")
    # wandb.log({"model_param": param})
    for epoch in tqdm(range(num_epochs)):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # update_trainable_params(model, epoch + 1)  # ← Insert this h
        # optimizer, lr_scheduler = build_optimizer_and_scheduler(model, config, steps_per_epoch, epoch)

        # Training
        model, train_metrics, train_metrics_temp, preds, hadm_ls_check, gates_weights_hadm = (
            train(model, train_loader, optimizer, lr_scheduler, config, device, my_metrics, tokenizer, scaler,
                  config["grad_accumulation"]))



        # Optionally: validation
        print("Running inference on validation set...")
        val_metrics, val_metrics_temp, my_metrics = inference(model, validation_loader, device, config, my_metrics
                                                              , tokenizer)


        result = save_results(
            train_metrics, val_metrics, training_args, lr_scheduler, config, timeframe="all"
        )
        # save Saved_models of aux task (only if there are some hyps and preds)
        print(result)

        cutoff_times = ["2d", "5d", "13d", "noDS"]

        wandb.log({"loss_val": result["validation_val_loss_cls"], "loss_train": result["train_loss_cls"]}, )

        for hadm, gate_weights_list in zip(hadm_ls_check, gates_weights_hadm):
            for chunk_idx, weights in enumerate(gate_weights_list):
                fig = plt.figure(figsize=(10, 2))
                sns.heatmap(weights[None, :], cmap="viridis", cbar=True)
                plt.title(f"HADM {hadm} - Gating weights for chunk {chunk_idx}")
                plt.xlabel("Event Index")
                plt.yticks([])

                # Log to wandb
                wandb.log({f"attention_weights/HADM_{hadm}_chunk_{chunk_idx}": wandb.Image(fig)})

                plt.close(fig)
        # save Saved_models of temp task
        if config["evaluate_temporal"]:
            for time in cutoff_times:
                _ = save_results(
                    train_metrics_temp[time],
                    val_metrics_temp[time],
                    training_args,
                    lr_scheduler,
                    config,
                    timeframe=time,
                )
                wandb.log({f"F1_val_{time}": _["validation_f1_micro"], f"F1_train_{time}": _["train_f1_micro"]}, )
                wandb.log({f"AUC_val_{time}": _["validation_auc_micro"], f"AUC_train_{time}": _["train_auc_micro"]}, )
                wandb.log({f"P@5_val_{time}": _["validation_p_5"], f"P@5_train_{time}": _["train_p_5"]}, )


        # cutoff_times = ["2d", "5d", "13d", "noDS"]


        training_args["CURRENT_PATIENCE_COUNT"] += 1
        training_args["TOTAL_COMPLETED_EPOCHS"] += 1

        if result["validation_f1_micro"] > training_args["CURRENT_BEST"]:
            training_args["CURRENT_BEST"] = result["validation_f1_micro"]
            training_args["CURRENT_PATIENCE_COUNT"] = 0
            best_path = os.path.join(
                config["project_path"],
                f"Saved_models/BEST_{config['run_name']}.pth",
            )
            if config["save_model"]:
                save_torch_model(model=model, optimizer=optimizer, scaler=scaler, lr_scheduler=lr_scheduler
                                 , result=result, training_args=training_args, save_path=best_path)

        if config["save_model"]:
            model_path = os.path.join(
                config["project_path"],
                f"Saved_models/{config['run_name']}.pth",
            )
            save_torch_model(model=model,optimizer=optimizer , scaler=scaler, lr_scheduler=lr_scheduler
                                 , result=result, training_args=training_args, save_path=model_path)

        if (config["patience_threshold"] > 0) and (
                training_args["CURRENT_PATIENCE_COUNT"]
                >= config["patience_threshold"]
        ):
            print("Stopped upon hitting early patience threshold ")
            break

        if (config["max_epochs"] > 0) and (
                training_args["TOTAL_COMPLETED_EPOCHS"] >= config["max_epochs"]
        ):
            print("Stopped upon hitting max number of training epochs")
            break
