from load_data import conll, da_train, lower_test, upper_test, label_list, id2label, label2id
from transformers import (AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
import evaluate
import numpy as np

class NERModel:
    def __init__(self, model_path: str, lr: float = 2e-5,
                 train_batch_size: int = 16, num_epochs: int = 2,
                 seed: int = 42, da: bool = False) -> None:
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.seqeval = evaluate.load('seqeval')
        self.conll = None
        self.da_train_data = None
        self.lower_test = None
        self.upper_test = None
        self.model = None
        self.da = da
        self.training_args = TrainingArguments(
            output_dir="test_ner_model",
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=16,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to='none',
            seed=seed
        )

    def model_init(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.model_path, num_labels=9, id2label=id2label, label2id=label2id
        )

    def tokenize_and_align_labels(self, examples: dict[str, list[str]]) -> dict[str, list[str]]:
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_and_align_all_data(self) -> None:
        self.conll = conll.map(self.tokenize_and_align_labels, batched=True)
        self.da_train_data = da_train.map(self.tokenize_and_align_labels, batched=True)
        self.lower_test = lower_test.map(self.tokenize_and_align_labels, batched=True)
        self.upper_test = upper_test.map(self.tokenize_and_align_labels, batched=True)

    def compute_metrics(self, p: tuple[list[list[int]], list[list[int]]]) -> dict[str, float]:
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

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self) -> None:
        if self.da:
            self.model = Trainer(
                model_init=self.model_init,
                args=self.training_args,
                train_dataset=self.da_train_data,
                eval_dataset=self.conll["test"],
                processing_class=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )

            self.model.train()
        else:
            self.model = Trainer(
                model_init=self.model_init,
                args=self.training_args,
                train_dataset=self.conll["train"],
                eval_dataset=self.conll["test"],
                processing_class=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )

            self.model.train()

    def eval(self) -> None:
        print(f"Original Dataset F1: {self.model.evaluate(eval_dataset=self.conll['test'])['eval_f1'] * 100:0.2f}")
        print(f"Lowercase Dataset F1: {self.model.evaluate(eval_dataset=self.lower_test)['eval_f1'] * 100:0.2f}")
        print(f"Uppercase Dataset F1: {self.model.evaluate(eval_dataset=self.upper_test)['eval_f1'] * 100:0.2f}")

    def finetune(self):
        for lr in [1e-6, 1e-5, 1e-4]:
            for batch_size in [8, 16, 32]:
                training_args = TrainingArguments(
                    output_dir="test_ner_model",
                    overwrite_output_dir=True,
                    learning_rate=lr,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=16,
                    num_train_epochs=5,
                    weight_decay=0.01,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    push_to_hub=False,
                    report_to='none',
                    seed=42,
                    logging_strategy='epoch'
                )

                if self.da:
                    trainer = Trainer(
                        model_init=self.model_init,
                        args=training_args,
                        train_dataset=self.da_train_data,
                        eval_dataset=self.conll["validation"],
                        processing_class=self.tokenizer,
                        data_collator=self.data_collator,
                        compute_metrics=self.compute_metrics,
                    )

                    trainer.train()

                else:
                    trainer = Trainer(
                        model_init=self.model_init,
                        args=training_args,
                        train_dataset=self.conll["train"],
                        eval_dataset=self.conll["validation"],
                        processing_class=self.tokenizer,
                        data_collator=self.data_collator,
                        compute_metrics=self.compute_metrics,
                    )

                    trainer.train()

if __name__ == '__main__':
    print(80 * '=')
    print('BERT-base-cased Baseline')
    print(80 * '=')
    baseline_bert_ner = NERModel('bert-base-cased')
    baseline_bert_ner.tokenize_and_align_all_data()
    baseline_bert_ner.train()
    baseline_bert_ner.eval()
    # print(baseline_bert_ner.finetune())

    print(80 * '=')
    print('BERT-base-cased Data Augmented')
    print(80 * '=')
    da_bert_ner = NERModel('bert-base-cased', da=True)
    da_bert_ner.tokenize_and_align_all_data()
    da_bert_ner.train()
    da_bert_ner.eval()
    # print(da_bert_ner.finetune())

    print(80 * '=')
    print('XLM-R Baseline')
    print(80 * '=')
    baseline_xlmr_ner = NERModel('xlm-roberta-base')
    baseline_xlmr_ner.tokenize_and_align_all_data()
    baseline_xlmr_ner.train()
    baseline_xlmr_ner.eval()

    print(80 * '=')
    print('XLM-R Data Augmented')
    print(80 * '=')
    da_xlmr_ner = NERModel('xlm-roberta-base', da=True)
    da_xlmr_ner.tokenize_and_align_all_data()
    da_xlmr_ner.train()
    da_xlmr_ner.eval()

    print('Done!')