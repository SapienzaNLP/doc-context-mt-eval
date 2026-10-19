"""Configuration dataclasses for meta-evaluation pipeline."""

from dataclasses import dataclass, field


@dataclass
class WandBConfig:
    project_name: str
    run_name: str
    mode: str = "online"
    new_run: bool = True


@dataclass
class MetaEvaluationConfig:
    dataset: str
    language_pair: str
    model_name: str
    is_reference_less: bool
    include_document_context: bool
    wandb: WandBConfig
    are_aggregated_results: bool = False
    ratings_level: str = "mqm"

    @property
    def responses_path(self) -> str:
        run_name = self.wandb.run_name
        if self.are_aggregated_results:
            return (
                f"outputs/metrics_results/"
                f"{self.dataset}/aggregated_results_mqm/{self.model_name}/"
                f"{run_name}/predicted_score/all_responses.csv"
            )
        return (
            f"outputs/metrics_results/"
            f"{self.dataset}/{self.language_pair}/{self.model_name}/"
            f"{run_name}/predicted_score/all_responses.csv"
        )

    @property
    def scores_dir(self) -> str:
        if self.language_pair != "all":
            return (
                f"outputs/metrics_results/"
                f"{self.dataset}/{self.language_pair}/{self.model_name}/"
                f"{self.wandb.run_name}/predicted_score"
            )
        return (
            f"outputs/metrics_results/"
            f"{self.dataset}/aggregated_results_mqm/{self.model_name}/"
            f"{self.wandb.run_name}/predicted_score"
        )

