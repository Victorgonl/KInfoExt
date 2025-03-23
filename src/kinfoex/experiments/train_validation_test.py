import datetime
from typing import Literal
import copy
import os
import json

from transformers import TrainingArguments
from datasets import concatenate_datasets
import optuna
import cpuinfo

# import igpu

from ..trainers import TrainerForRelationExtraction, TrainerForTokenClassification
from ..data_collator import KinfoextDataCollator
from ..callback import KinfoextCallback


def ensure_directory_exists(directory):
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)


def set_seed(seed):
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_dict(obj):
    """Tries to convert an object to a dictionary."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        return [convert_to_dict(item) for item in obj]
    if hasattr(obj, "__dict__"):
        return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}
    if isinstance(obj, set):
        return convert_to_dict(list(obj))
    return str(obj)


def train_validation_test(
    experiment_name,
    task: Literal["token_classification", "relation_extraction"],
    model_init,
    train_dataset,
    test_dataset,
    args: TrainingArguments,
    compute_metrics,
    hp_space=None,
    max_trials=0,
    compute_objective=None,
    direction=None,
    data_collator=KinfoextDataCollator(),
    validation_dataset=None,
    callbacks=None,
    experiment_directory=None,
    pruner=None,
    join_validation_on_train_after=True,
    save_model=False,
):
    assert task in {"token_classification", "relation_extraction"}
    Trainer = (
        TrainerForTokenClassification
        if task == "token_classification"
        else TrainerForRelationExtraction
    )

    experiment_directory = os.path.join(experiment_directory or "", experiment_name)
    ensure_directory_exists(experiment_directory)

    if callbacks is None:
        callbacks = []

    args_original = args
    set_seed(args_original.seed)

    hp_search_dir = os.path.join(experiment_directory, "hp_search")
    ensure_directory_exists(hp_search_dir)

    best_hp_path = os.path.join(hp_search_dir, "best_hp_found.json")
    optuna_log_path = os.path.join(hp_search_dir, "optuna_logs.log")

    if not os.path.exists(best_hp_path):
        hp_search_args = copy.deepcopy(args_original)
        hp_search_args.output_dir = hp_search_dir
        hp_search_args.logging_dir = os.path.join(hp_search_dir, "logs")
        hp_search_args.save_strategy = "no"

        cbks = [
            KinfoextCallback(
                experiment_name=experiment_name,
                experiment_directory=experiment_directory,
                max_trials=max_trials,
                status="Searching best hyper-parameters set...",
            )
        ] + callbacks

        if not os.path.exists(optuna_log_path):
            open(optuna_log_path, "w").close()

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(optuna_log_path)
        )

        try:
            study = optuna.load_study(study_name=experiment_name, storage=storage)
            n_previous_trials = len(study.trials)
            n_failed_trials = sum(
                1
                for trial in study.trials
                if trial.state
                in {optuna.trial.TrialState.FAIL, optuna.trial.TrialState.RUNNING}
            )
            n_trials = max_trials - (n_previous_trials - n_failed_trials)
        except:
            n_trials = max_trials

        if n_trials > 0 and validation_dataset is not None:
            ensure_directory_exists(os.path.join(hp_search_dir, "info"))

            hp_search_trainer = Trainer(
                model_init=model_init,
                train_dataset=train_dataset,
                eval_dataset=validation_dataset,
                args=hp_search_args,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                callbacks=cbks,
            )

            hp_search_results = hp_search_trainer.hyperparameter_search(
                hp_space=hp_space,
                backend="optuna",
                n_trials=n_trials,
                direction=direction,
                compute_objective=compute_objective,
                pruner=pruner,
                study_name=experiment_name,
                storage=storage,
                load_if_exists=True,
            )

            best_hyperparameters = {
                "hyperparameters": hp_search_results.hyperparameters,
                "run_id": hp_search_results.run_id,
                "objective": hp_search_results.objective,
            }
            with open(best_hp_path, "w") as f:
                json.dump(best_hyperparameters, f, indent=4)

    # Fine-tuning
    finetuning_dir = os.path.join(experiment_directory, "finetuning")
    ensure_directory_exists(finetuning_dir)
    ensure_directory_exists(os.path.join(finetuning_dir, "info"))

    cpu_info = cpuinfo.get_cpu_info()
    with open(os.path.join(finetuning_dir, "info", "cpu_info.json"), "w") as f:
        json.dump(cpu_info, f, indent=4)

    with open(os.path.join(finetuning_dir, "info", "training_started.txt"), "w") as f:
        f.write(datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S"))

    if join_validation_on_train_after and validation_dataset is not None:
        train_dataset = concatenate_datasets([train_dataset, validation_dataset])

    args = copy.deepcopy(args_original)
    args.output_dir = os.path.join(finetuning_dir, "model")
    args.logging_dir = os.path.join(finetuning_dir, "logs")

    with open(os.path.join(finetuning_dir, "info", "training_args.json"), "w") as f:
        json.dump(args.to_dict(), f, indent=4)

    trainer = Trainer(
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()
    finetuning_result = trainer.evaluate()

    trainer.save_state()
    if save_model:
        trainer.save_model()
    else:
        trainer.model.config.to_json_file(os.path.join(args.output_dir, "config.json"))

    with open(os.path.join(finetuning_dir, "finetuning_result.json"), "w") as f:
        json.dump(finetuning_result, f, indent=4)
