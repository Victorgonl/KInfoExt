import datetime
from typing import Literal
import copy
import os
import json

from transformers import TrainingArguments
from datasets import concatenate_datasets
import optuna
import cpuinfo
import igpu

from ..trainers import TrainerForRelationExtraction, TrainerForTokenClassification
from ..data_collator import KinfoextDataCollator
from ..callback import KinfoextCallback


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
    """
    Tries to convert an object to a dictionary.

    Parameters:
        obj: The object to be converted.

    Returns:
        A dictionary representation of the object.
    """
    # If object is already a dictionary, return it as is
    if isinstance(obj, dict):
        return obj

    # If object is a list or tuple, convert each element to dictionary recursively
    if isinstance(obj, (list, tuple)):
        return [convert_to_dict(item) for item in obj]

    # If object is an object of a class, convert its attributes to dictionary
    if hasattr(obj, "__dict__"):
        return {key: convert_to_dict(value) for key, value in obj.__dict__.items()}

    # If object is a set, convert it to a list and then to dictionary
    if isinstance(obj, set):
        return convert_to_dict(list(obj))

    # For other types of objects, return the string representation
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

    assert task == "token_classification" or task == "relation_extraction"
    if task == "token_classification":
        Trainer = TrainerForTokenClassification
    elif task == "relation_extraction":
        Trainer = TrainerForRelationExtraction

    if experiment_directory is not None:
        experiment_directory = f"{experiment_directory}/{experiment_name}"

    if callbacks is None:
        callbacks = []

    args_original = args

    set_seed(args_original.seed)

    if not os.path.exists(f"{experiment_directory}/hp_search/best_hp_found.json"):

        hp_search_args = copy.deepcopy(args_original)
        hp_search_args.output_dir = f"{experiment_directory}/hp_search"
        hp_search_args.logging_dir = f"{experiment_directory}/hp_search/logs"
        hp_search_args.save_strategy = "no"

        cbks = [
            KinfoextCallback(
                experiment_name=experiment_name,
                experiment_directory=experiment_directory,
                max_trials=max_trials,
                status="Searching best hyper-parameters set...",
            )
        ] + callbacks

        hp_search_trainer = Trainer(
            model_init=model_init,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            args=hp_search_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=cbks,  # type: ignore
        )

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(
                f"{experiment_directory}/hp_search/optuna_logs.log"
            )
        )

        try:
            study = optuna.load_study(
                study_name=f"{experiment_name}",
                storage=storage,
            )
            n_previous_trials = len(study.trials)
            n_failed_trials = 0
            for trial in study.trials:
                if trial.state in [
                    optuna.trial.TrialState.FAIL,
                    optuna.trial.TrialState.RUNNING,
                ]:
                    n_failed_trials += 1
            n_trials = max_trials - (n_previous_trials - n_failed_trials)
        except:
            n_trials = max_trials

        if n_trials > 0 and validation_dataset is not None:

            hp_search_results = hp_search_trainer.hyperparameter_search(
                hp_space=hp_space,
                backend="optuna",
                n_trials=n_trials,
                direction=direction,
                compute_objective=compute_objective,
                pruner=pruner,
                study_name=f"{experiment_name}",
                storage=storage,
                load_if_exists=True,
            )

            best_hyperparameters = {}
            best_hyperparameters["hyperparameters"] = hp_search_results.hyperparameters  # type: ignore
            best_hyperparameters["run_id"] = hp_search_results.run_id  # type: ignore
            best_hyperparameters["objective"] = hp_search_results.objective  # type: ignore

            with open(f"{experiment_directory}/hp_search/best_hp_found.json", "w") as f:
                json.dump(best_hyperparameters, f, indent=4)

            os.makedirs(
                f"{experiment_directory}/hp_search/info/",
                exist_ok=True,
            )
            train_dataset_info = {}
            train_dataset_info["dataset_name"] = train_dataset.info.dataset_name
            train_dataset_info["samples"] = [sample["id"] for sample in train_dataset]  # type: ignore
            with open(
                f"{experiment_directory}/hp_search/info/train_dataset.json", "w"
            ) as f:
                json.dump(train_dataset_info, f, indent=4)

            validation_dataset_info = {}
            validation_dataset_info["dataset_name"] = train_dataset.info.dataset_name
            validation_dataset_info["samples"] = [
                sample["id"] for sample in validation_dataset  # type: ignore
            ]
            with open(
                f"{experiment_directory}/hp_search/info/validation_dataset.json",
                "w",
            ) as f:
                json.dump(validation_dataset_info, f, indent=4)

    # Fine-tuning on best hyper-parameters
    if not os.path.exists(f"{experiment_directory}/finetuning/finetuning_result.json"):

        os.makedirs(
            f"{experiment_directory}/finetuning/info/",
            exist_ok=True,
        )

        cpu_info = cpuinfo.get_cpu_info()
        with open(
            f"{experiment_directory}/finetuning/info/cpu_info.json", "w"
        ) as outfile:
            json.dump(cpu_info, outfile, indent=4)
        devices = igpu.devices_index()
        gpu_info = {
            device: convert_to_dict(igpu.get_device(device)) for device in devices
        }
        with open(
            f"{experiment_directory}/finetuning/info/gpu_info.json", "w"
        ) as outfile:
            json.dump(gpu_info, outfile, indent=4)
        with open(
            f"{experiment_directory}/finetuning/info/training_started.txt", "w"
        ) as f:
            f.write(f"{datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')}")

        if join_validation_on_train_after and validation_dataset is not None:
            train_dataset = concatenate_datasets([train_dataset, validation_dataset])

        args = copy.deepcopy(args_original)
        args.output_dir = f"{experiment_directory}/finetuning/model/"
        args.logging_dir = f"{experiment_directory}/finetuning/logs/"
        args.logging_first_step = True
        if validation_dataset is not None:
            with open(f"{experiment_directory}/hp_search/best_hp_found.json", "r") as f:
                best_hyperparameters = json.load(f)
            for hyperparameter in best_hyperparameters["hyperparameters"]:
                setattr(
                    args,
                    hyperparameter,
                    best_hyperparameters["hyperparameters"][hyperparameter],
                )

        with open(
            f"{experiment_directory}/finetuning/info/training_args.json", "w"
        ) as f:
            json.dump(args.to_dict(), f, indent=4)

        cbks = [
            KinfoextCallback(
                experiment_name=experiment_name,
                experiment_directory=experiment_directory,
                status="Training with best hyper-parameters set found...",
            )
        ] + callbacks

        trainer = Trainer(
            model_init=model_init,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=cbks,  # type: ignore
        )

        trainer.train()
        finetuning_result = trainer.evaluate()
        trainer.save_state()
        if save_model:
            trainer.save_model()
        else:
            trainer.model.config.to_json_file(f"{args.output_dir}/config.json")

        with open(
            f"{experiment_directory}/finetuning/info/training_finished.txt", "w"
        ) as f:
            f.write(f"{datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')}")

        with open(
            f"{experiment_directory}/finetuning/finetuning_result.json", "w"
        ) as f:
            json.dump(finetuning_result, f, indent=4)

        train_dataset_info = {}
        train_dataset_info["dataset_name"] = train_dataset.info.dataset_name
        train_dataset_info["samples"] = [sample["id"] for sample in train_dataset]  # type: ignore
        with open(
            f"{experiment_directory}/finetuning/info/train_dataset.json", "w"
        ) as f:
            json.dump(train_dataset_info, f, indent=4)

        test_dataset_info = {}
        test_dataset_info["dataset_name"] = test_dataset.info.dataset_name
        test_dataset_info["samples"] = [sample["id"] for sample in test_dataset]  # type: ignore
        with open(
            f"{experiment_directory}/finetuning/info/test_dataset.json",
            "w",
        ) as f:
            json.dump(test_dataset_info, f, indent=4)

    with open(f"{experiment_directory}/finetuning/finetuning_result.json", "r") as f:
        finetuning_result = json.load(f)

    results = {"metrics": finetuning_result}
    if validation_dataset is not None:
        results["best_hp_found"] = best_hyperparameters
    with open(f"{experiment_directory}/results.json", "w") as f:
        json.dump(results, f, indent=4)
