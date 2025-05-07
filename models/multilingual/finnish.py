from preprocessing_utils import get_reg_pandas_df, get_aug_pandas_df, compute_metrics
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification


#load train file
fin_aug_train = get_aug_pandas_df("./data/fin/train.txt", test=False)
fin_reg_train = get_reg_pandas_df("./data/fin/train.txt")
#load dev file: we don't augment here
fin_dev = get_reg_pandas_df("./data/fin/dev.txt")
#load test file: we want regular, upper, and lower
fin_test_reg, fin_test_lower, fin_test_upper = get_aug_pandas_df("./data/fin/test.txt", test=True)

#now we can create the dataset object
fin_reg_dataset = DatasetDict()
fin_aug_dataset = DatasetDict()
fin_test_dataset = DatasetDict()
#we manually set train, dev, and test
fin_reg_dataset["train"] = Dataset.from_pandas(fin_reg_train)
fin_aug_dataset["train"] = Dataset.from_pandas(fin_aug_train)
fin_reg_dataset["dev"] = Dataset.from_pandas(fin_dev)
fin_aug_dataset["dev"] = Dataset.from_pandas(fin_dev)
fin_test_dataset["reg"] = Dataset.from_pandas(fin_test_reg)
fin_test_dataset["lower"] = Dataset.from_pandas(fin_test_lower)
fin_test_dataset["upper"] = Dataset.from_pandas(fin_test_upper)

#the next step is tokenization!
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
label_list = ["O", "B-PER", "I-PER", "O-PER", "B-ORG", "I-ORG", "O-ORG", "B-LOC", "I-LOC", "O-LOC"]
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

fin_tokenized_reg_dataset = fin_reg_dataset.map(tokenize_and_align_labels, batched=True)
fin_tokenized_aug_dataset = fin_aug_dataset.map(tokenize_and_align_labels, batched=True)
fin_tokenized_test_dataset = fin_test_dataset.map(tokenize_and_align_labels, batched=True)

#now we initialize the model
reg_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))
aug_model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))

#setting up gpu stuff
reg_model.to("cuda")
aug_model.to("cuda")

#making a data collator to convert to long
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


training_args = TrainingArguments(output_dir="temp_output", overwrite_output_dir=True, learning_rate=5e-5,
    per_device_train_batch_size=16,per_device_eval_batch_size=16, num_train_epochs=3, weight_decay=0.05,
    eval_strategy="epoch", logging_strategy="epoch", save_strategy="no", load_best_model_at_end=False,
    push_to_hub=False, report_to='none', disable_tqdm=True, seed=1)

reg_trainer = Trainer(reg_model, args = training_args, train_dataset=fin_tokenized_reg_dataset["train"],
                      eval_dataset=fin_tokenized_reg_dataset["dev"], data_collator=data_collator, compute_metrics=compute_metrics)
reg_trainer.train()
print(f"Baseline F1 on Reg: {reg_trainer.evaluate(eval_dataset=fin_tokenized_test_dataset['reg'])['eval_f1'] * 100:0.2f}")
print(f"Baseline F1 on Upper: {reg_trainer.evaluate(eval_dataset=fin_tokenized_test_dataset['upper'])['eval_f1'] * 100:0.2f}")
print(f"Baseline F1 on Lower: {reg_trainer.evaluate(eval_dataset=fin_tokenized_test_dataset['lower'])['eval_f1'] * 100:0.2f}")

aug_trainer = Trainer(aug_model, args= training_args, train_dataset=fin_tokenized_aug_dataset["train"],
                      eval_dataset=fin_tokenized_aug_dataset["dev"], data_collator=data_collator, compute_metrics=compute_metrics)
aug_trainer.train()
print(f"Augmented F1 on Reg: {aug_trainer.evaluate(eval_dataset=fin_tokenized_test_dataset['reg'])['eval_f1'] * 100:0.2f}")
print(f"Augmented F1 on Upper: {aug_trainer.evaluate(eval_dataset=fin_tokenized_test_dataset['upper'])['eval_f1'] * 100:0.2f}")
print(f"Augmented F1 on Lower: {aug_trainer.evaluate(eval_dataset=fin_tokenized_test_dataset['lower'])['eval_f1'] * 100:0.2f}")
