import datetime
from typing import Literal
import copy
import os
import json
import statistics

from transformers import TrainingArguments
from datasets import DatasetDict, concatenate_datasets
import optuna
import cpuinfo
import igpu

from ..trainers import TrainerForRelationExtraction, TrainerForTokenClassification
from ..data_collator import UFLAFORMSDataCollator
from ..callback import UFLAFORMSCallback


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


def k_fold_cross_validation(
    experiment_name,
    task: Literal["token_classification", "relation_extraction"],
    model_init,
    partitioned_dataset: DatasetDict,
    args: TrainingArguments,
    compute_metrics,
    hp_space,
    max_trials,
    compute_objective,
    direction,
    data_collator=UFLAFORMSDataCollator(),
    callbacks=None,
    experiment_directory=None,
    pruner=None,
):
    """Runs k-fold cross-validation algorithm using `k` partitions according to `dataset_dict` splits (keys)."""

    assert task == "token_classification" or task == "relation_extraction"
    if task == "token_classification":
        Trainer = TrainerForTokenClassification
    elif task == "relation_extraction":
        Trainer = TrainerForRelationExtraction

    if experiment_directory is not None:
        experiment_directory = f"{experiment_directory}/{experiment_name}"

    args_original = args

    set_seed(args_original.seed)

    partitions = list(partitioned_dataset.keys())
    k = len(partitions)
    j = 0

    for i in range(j, k):

        test_partition = partitions[i]
        validation_partition = partitions[(i + 1) % k]
        train_partitions = [
            p for p in partitions if p != test_partition and p != validation_partition
        ]

        test_dataset = partitioned_dataset[test_partition]
        validation_dataset = partitioned_dataset[validation_partition]
        train_dataset = concatenate_datasets(
            [
                partitioned_dataset[partition]
                for partition in partitioned_dataset
                if partition in train_partitions
            ]
        )

        if not os.path.exists(
            f"{experiment_directory}/{str(i)}/hp_search/best_hp_found.json"
        ):

            hp_search_args = copy.deepcopy(args_original)
            hp_search_args.output_dir = f"{experiment_directory}/{str(i)}/hp_search"
            hp_search_args.logging_dir = (
                f"{experiment_directory}/{str(i)}/hp_search/logs"
            )

            cbks = [
                UFLAFORMSCallback(
                    experiment_name=experiment_name,
                    experiment_directory=experiment_directory,
                    i_iteration=i,
                    k_iteration=k,
                    max_trials=max_trials,
                    status="Searching best hyper-parameters set...",
                )
            ]
            if callbacks is not None:
                cbks += callbacks

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
                    f"{experiment_directory}/{str(i)}/hp_search/optuna_logs.log"
                )
            )

            try:
                study = optuna.load_study(
                    study_name=f"{experiment_name}-{i}",
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

            if n_trials >= 0:

                hp_search_results = hp_search_trainer.hyperparameter_search(
                    hp_space=hp_space,
                    backend="optuna",
                    n_trials=n_trials,
                    direction=direction,
                    compute_objective=compute_objective,
                    pruner=pruner,
                    study_name=f"{experiment_name}-{str(i)}",
                    storage=storage,
                    load_if_exists=True,
                )

                best_hyperparameters = {}
                best_hyperparameters["hyperparameters"] = hp_search_results.hyperparameters  # type: ignore
                best_hyperparameters["run_id"] = hp_search_results.run_id  # type: ignore
                best_hyperparameters["objective"] = hp_search_results.objective  # type: ignore

                with open(
                    f"{experiment_directory}/{str(i)}/hp_search/best_hp_found.json", "w"
                ) as f:
                    json.dump(best_hyperparameters, f, indent=4)

                os.makedirs(
                    f"{experiment_directory}/{str(i)}/hp_search/info/",
                    exist_ok=True,
                )
                train_dataset_info = {}
                train_dataset_info["dataset_name"] = train_dataset.info.dataset_name
                train_dataset_info["partitions"] = train_partitions
                train_dataset_info["samples"] = [sample["id"] for sample in train_dataset]  # type: ignore
                with open(
                    f"{experiment_directory}/{str(i)}/hp_search/info/train_dataset.json",
                    "w",
                ) as f:
                    json.dump(train_dataset_info, f, indent=4)

                validation_dataset_info = {}
                validation_dataset_info["dataset_name"] = (
                    train_dataset.info.dataset_name
                )
                validation_dataset_info["partitions"] = [validation_partition]
                validation_dataset_info["samples"] = [
                    sample["id"] for sample in validation_dataset  # type: ignore
                ]
                with open(
                    f"{experiment_directory}/{str(i)}/hp_search/info/validation_dataset.json",
                    "w",
                ) as f:
                    json.dump(validation_dataset_info, f, indent=4)

        # Fine-tuning on best hyper-parameters
        with open(
            f"{experiment_directory}/{str(i)}/hp_search/best_hp_found.json", "r"
        ) as f:
            best_hyperparameters = json.load(f)

        if not os.path.exists(
            f"{experiment_directory}/{str(i)}/finetuning/finetuning_result.json"
        ):

            os.makedirs(
                f"{experiment_directory}/{str(i)}/finetuning/info/",
                exist_ok=True,
            )

            cpu_info = cpuinfo.get_cpu_info()
            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/cpu_info.json", "w"
            ) as outfile:
                json.dump(cpu_info, outfile, indent=4)
            devices = igpu.devices_index()
            gpu_info = {
                device: convert_to_dict(igpu.get_device(device)) for device in devices
            }
            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/gpu_info.json", "w"
            ) as outfile:
                json.dump(gpu_info, outfile, indent=4)
            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/training_started.txt",
                "w",
            ) as f:
                f.write(f"{datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')}")

            train_partitions.append(validation_partition)
            train_partitions.sort()
            train_dataset = concatenate_datasets(
                [
                    partitioned_dataset[partition]
                    for partition in partitioned_dataset
                    if partition in train_partitions
                ]
            )
            args = copy.deepcopy(args_original)
            args.output_dir = f"{experiment_directory}/{str(i)}/finetuning/model/"
            args.logging_dir = f"{experiment_directory}/{str(i)}/finetuning/logs/"
            args.logging_first_step = True
            for hyperparameter in best_hyperparameters["hyperparameters"]:
                setattr(
                    args,
                    hyperparameter,
                    best_hyperparameters["hyperparameters"][hyperparameter],
                )
            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/training_args.json",
                "w",
            ) as f:
                json.dump(args.to_dict(), f, indent=4)

            cbks = [
                UFLAFORMSCallback(
                    experiment_name=experiment_name,
                    experiment_directory=experiment_directory,
                    status="Training with best hyper-parameters set found...",
                    i_iteration=i,
                    k_iteration=k,
                )
            ]
            if callbacks is not None:
                cbks += callbacks

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
            trainer.model.config.to_json_file(f"{args.output_dir}/config.json")

            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/training_finished.txt",
                "w",
            ) as f:
                f.write(f"{datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')}")

            with open(
                f"{experiment_directory}/{str(i)}/finetuning/finetuning_result.json",
                "w",
            ) as f:
                json.dump(finetuning_result, f, indent=4)

            train_dataset_info = {}
            train_dataset_info["dataset_name"] = train_dataset.info.dataset_name
            train_dataset_info["partitions"] = train_partitions
            train_dataset_info["samples"] = [sample["id"] for sample in train_dataset]  # type: ignore
            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/train_dataset.json",
                "w",
            ) as f:
                json.dump(train_dataset_info, f, indent=4)

            test_dataset_info = {}
            test_dataset_info["dataset_name"] = test_dataset.info.dataset_name
            test_dataset_info["partitions"] = [test_partition]
            test_dataset_info["samples"] = [sample["id"] for sample in test_dataset]  # type: ignore
            with open(
                f"{experiment_directory}/{str(i)}/finetuning/info/test_dataset.json",
                "w",
            ) as f:
                json.dump(test_dataset_info, f, indent=4)

    results = {
        "metrics": {},
        "best_hp_found": {"hyperparameters": {}, "objective": []},
    }
    for i in range(k):
        with open(
            f"{experiment_directory}/{str(i)}/finetuning/finetuning_result.json", "r"
        ) as f:
            finetuning_result = json.load(f)
            for key in finetuning_result:
                if (
                    key.endswith("_precision")
                    or key.endswith("_recall")
                    or key.endswith("_f1")
                ):
                    if key not in results["metrics"]:
                        results["metrics"][key] = []
                    results["metrics"][key].append(finetuning_result[key])
        with open(
            f"{experiment_directory}/{str(i)}/hp_search/best_hp_found.json", "r"
        ) as f:
            best_hp_found = json.load(f)
            results["best_hp_found"]["objective"].append(best_hp_found["objective"])
            for hp in best_hp_found["hyperparameters"]:
                if hp not in results["best_hp_found"]["hyperparameters"]:
                    results["best_hp_found"]["hyperparameters"][hp] = []
                results["best_hp_found"]["hyperparameters"][hp].append(
                    best_hp_found["hyperparameters"][hp]
                )

    results["statistics"] = {"mean": {}, "stdev": {}}
    for key in results["metrics"]:
        results["statistics"]["mean"][key] = statistics.mean(results["metrics"][key])
        results["statistics"]["stdev"][key] = statistics.stdev(results["metrics"][key])
    with open(f"{experiment_directory}/results.json", "w") as f:
        json.dump(results, f, indent=4)
