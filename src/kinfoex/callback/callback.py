import os
import pprint

import optuna
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

# pandas.options.display.float_format = "{:,.2f}".format


def print_feedback(message: str) -> None:
    message = f"|| {message} ||"
    print()
    print("=" * len(message))
    print(message)
    print("=" * len(message))
    print()


class KinfoextCallback(TrainerCallback):

    def __init__(
        self,
        experiment_name,
        experiment_directory,
        k_iteration=None,
        i_iteration=None,
        max_trials=None,
        status=None,
    ) -> None:
        self.experiment_name = experiment_name
        self.experiment_directory = experiment_directory
        self.k_iteration = k_iteration
        self.i_iteration = i_iteration
        self.max_trials = max_trials
        self.status = status

    def __get_number_of_hpsearch_trials(self):
        if self.i_iteration:
            storage_path = f"{self.experiment_directory}/{str(self.i_iteration)}/hp_search/optuna_logs.log"
            study_name = f"{self.experiment_name}-{self.i_iteration}"
        else:
            storage_path = f"{self.experiment_directory}/hp_search/optuna_logs.log"
            study_name = f"{self.experiment_name}"
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(storage_path)
        )
        study = optuna.load_study(
            study_name=study_name,
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
        return study, n_previous_trials - n_failed_trials + 1

    def __feedback(self, args, state, control, kwargs):
        if self.status and self.max_trials:
            study, n_trials = self.__get_number_of_hpsearch_trials()
        os.system("cls" if os.name == "nt" else "clear")
        print_feedback(f"Experiment: {self.experiment_name}")
        if self.status:
            print(f"Status: {self.status}")
            print()
        if self.i_iteration is not None and self.k_iteration is not None:
            print(f"Iteration: {self.i_iteration + 1}/{self.k_iteration}")
            print()
        if self.status and self.max_trials:
            print(f"Trial: {n_trials}/{self.max_trials}")
            print()
            if n_trials > 1:
                hp = {
                    "trial_id": study.best_trial.number,
                    "objective": study.best_trial.value,
                    "hyperparameters": study.best_trial.params,
                }
                print(f"Best hyper-parameters found:")
                pprint.pprint(hp, sort_dicts=False)
                print()
        if "metrics" in kwargs:
            print("Metrics:")
            pprint.pprint(kwargs["metrics"], sort_dicts=False)
        print()

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        self.__feedback(args, state, control, kwargs)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.__feedback(args, state, control, kwargs)
