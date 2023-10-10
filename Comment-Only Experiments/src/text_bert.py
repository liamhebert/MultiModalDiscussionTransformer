from src.utils import compute_metrics
from src.dataset_loader import DatasetLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import pandas as pd

def main(exp_dict, split):
    dataset = exp_dict["dataset"]["name"]
    ds = DatasetLoader(
        data_dir=f"./data/{dataset}/big",
        dname=dataset,
        modalities=exp_dict["modalities"],
        split=split
    ).ds
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def preprocess(example):
        return tokenizer(example["text"], max_length=100, truncation=True)

    encoded_dataset = ds.map(preprocess, batched=True)

    backbone = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=exp_dict["output_dir"],
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        learning_rate=exp_dict["lr"],
        logging_dir='logs/' + exp_dict['output_dir'].split('/')[-1],
        warmup_steps=exp_dict["warmup_steps"],
        max_steps=exp_dict["max_steps"],
        #max_steps=200,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        per_device_eval_batch_size=48,
        per_device_train_batch_size=48,
        save_total_limit=3,
        report_to=['wandb'],
    )
    print(training_args)

    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Trainer
    trainer = Trainer(
        backbone,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(
        trainer.evaluate(
            eval_dataset=encoded_dataset["validation"], metric_key_prefix="valid"
        )
    )
    trainer.save_model('fine_tuned_text_model')
    predictions, labels, metrics = trainer.predict(encoded_dataset["validation"])
    
    print(predictions)
    print(labels)
    print('OUTPUT DIR', exp_dict["output_dir"])
    data = {
        'y_pred': list(predictions),
        'y_true': labels
    }
     
    df = pd.DataFrame(data)
    df.to_parquet(exp_dict['output_dir'] + '/predictions.parquet')

if __name__ == "__main__":
    main()
