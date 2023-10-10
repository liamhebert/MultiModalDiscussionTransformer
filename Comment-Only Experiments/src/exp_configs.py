EXP_GROUPS = {}
DATASETS = {
    "mm-reddit": {
        "name": "mm-reddit",
        "num_labels": 2,
    },
}


def get_base_config(
    dname="mm-reddit",
    modalities=["text"],
    metrics=["accuracy", "f1", "precision", "recall"],
):
    d_config = {"config": "original"}
    d_config.update(DATASETS[dname])

    return {
        "dataset": d_config,
        "lr": 3e-5,
        "batch_size": 4,
        "max_steps": 5000,
        "warmup_steps": 400,
        "weight_decay": 0.01,
        "metrics": [metrics],
        "metric_best": "accuracy",
        "modalities": modalities,
        "eval_accumulation_steps": 30,
    }
