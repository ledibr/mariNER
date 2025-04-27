from preprocessing_utils import get_reg_pandas_df, get_aug_pandas_df
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, DataCollatorForTokenClassification, Trainer
import numpy as np
import torch
import evaluate

#load train file
spanish_reg_train = get_reg_pandas_df("./data/spa/train.txt")
spanish_aug_train = get_aug_pandas_df("./data/spa/train.txt")
#load dev file
spanish_reg_dev = get_reg_pandas_df("./data/spa/dev.txt")
spanish_aug_dev = get_aug_pandas_df("./data/spa/dev.txt")
#load test file
spanish_reg_test = get_reg_pandas_df("./data/spa/test.txt")
spanish_aug_test = get_aug_pandas_df("./data/spa/test.txt")

#now we can create the dataset object
spanish_reg_dataset = DatasetDict()
spanish_aug_dataset = DatasetDict()
#we manually set train, dev, and test
spanish_reg_dataset["train"] = Dataset.from_pandas(spanish_reg_train)
spanish_aug_dataset["train"] = Dataset.from_pandas(spanish_aug_train)
spanish_reg_dataset["dev"] = Dataset.from_pandas(spanish_reg_dev)
spanish_aug_dataset["dev"] = Dataset.from_pandas(spanish_aug_dev)
spanish_reg_dataset["test"] = Dataset.from_pandas(spanish_reg_test)
spanish_reg_dataset["test"] = Dataset.from_pandas(spanish_reg_test)

#the next step is tokenization!
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
label_list = ["O", "B-PER", "I-PER", "O-PER", "B-ORG", "I-ORG", "O-ORG", "B-LOC", "I-LOC", "O-LOC", "B-MISC", "I-MISC", "O-MISC"]
label2id = {label : index for index, label in enumerate(label_list)}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], padding= "max_length", truncation=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(int(-100))
            elif word_idx != previous_word_idx:
                if word_idx < len(label):
                    label_ids.append(int(label2id[label[word_idx]]))
                else:
                    label_ids.append(int(-100))
            else:
                label_ids.append(int(-100))
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

spa_tokenized_reg_dataset = spanish_reg_dataset.map(tokenize_and_align_labels, batched=True)
spa_tokenized_aug_dataset = spanish_aug_dataset.map(tokenize_and_align_labels, batched=True)



#now we initialize the model
reg_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))
aug_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))

#setting up gpu stuff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reg_model.to(device)
aug_model.to(device)

#also training_args
reg_training_args = TrainingArguments(
    output_dir="reg_fin",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to = 'none',
)

aug_training_args = TrainingArguments(
    output_dir="aug_fin",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to = 'none',
)

#set up compute metrics
def compute_metrics(p) -> dict[str, float]:
    seqeval = evaluate.load("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#making a data collator to convert to long
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

#now we train
reg_trainer = Trainer(reg_model, args = reg_training_args, train_dataset=spa_tokenized_reg_dataset["train"],
                      eval_dataset=spa_tokenized_reg_dataset["dev"], data_collator=data_collator, compute_metrics=compute_metrics)
reg_trainer.train()
print(f"Baseline F1: {reg_trainer.evaluate(eval_dataset=spa_tokenized_reg_dataset['dev'])['eval_f1'] * 100:0.2f}")

aug_trainer = Trainer(aug_model, args= aug_training_args, train_dataset=spa_tokenized_aug_dataset["train"],
                      eval_dataset=spa_tokenized_aug_dataset["dev"], data_collator=data_collator, compute_metrics=compute_metrics)
aug_trainer.train()
print(f"Aug F1: {aug_trainer.evaluate(eval_dataset=spa_tokenized_aug_dataset['dev'])['eval_f1'] * 100:0.2f}")
