import evaluate
import numpy as np
from transformers import (AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
from load_data import AugmentedDataLoader


class NERModel:
    def __init__(self, model_path: str, data_loader: AugmentedDataLoader, lr: float = 5e-5,
                 train_batch_size: int = 16, decay: float = 0.05, num_epochs: int = 5,
                 seed: int = 42, aug: bool = False) -> None:
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.seqeval = evaluate.load('seqeval')
        self.data_loader = data_loader
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.aug_data = None
        self.lower_test = None
        self.upper_test = None
        self.model = None
        self.aug = aug
        self.seed = seed
        self.training_args = TrainingArguments(
            output_dir='test_ner_model',
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=16,
            num_train_epochs=num_epochs,
            weight_decay=decay,
            warmup_ratio=0.1,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to='none',
            disable_tqdm=True,
            seed=seed
        )

        self._tokenize_and_align_all_data()

    def model_init(self) -> AutoModelForTokenClassification:
        return AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=len(self.data_loader.labels),
            id2label=self.data_loader.id2label,
            label2id=self.data_loader.label2id
        )

    def _tokenize_and_align_labels(self, examples: dict[str, list[str]]) -> dict[str, list[str]]:
        tokenized_inputs = self.tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples['ner_tags']):
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

        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    def _tokenize_and_align_all_data(self) -> None:
        self.train_data = self.data_loader.train.map(self._tokenize_and_align_labels, batched=True)
        self.dev_data = self.data_loader.dev.map(self._tokenize_and_align_labels, batched=True)
        self.test_data = self.data_loader.test.map(self._tokenize_and_align_labels, batched=True)
        self.aug_data = self.data_loader.aug_train.map(self._tokenize_and_align_labels, batched=True)
        self.lower_test = self.data_loader.lower_test.map(self._tokenize_and_align_labels, batched=True)
        self.upper_test = self.data_loader.upper_test.map(self._tokenize_and_align_labels, batched=True)

    def compute_metrics(self, p: tuple[list[list[int]], list[list[int]]]) -> dict[str, float]:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.data_loader.labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.data_loader.labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            'precision': results['overall_precision'],
            'recall': results['overall_recall'],
            'f1': results['overall_f1'],
            'accuracy': results['overall_accuracy'],
        }

    def train(self) -> None:
        train_set = self.train_data
        if self.aug:
            train_set = self.aug_data

        self.model = Trainer(
            model_init=self.model_init,
            args=self.training_args,
            train_dataset=train_set,
            eval_dataset=self.dev_data,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        if self.data_loader.corpus in ['eng', 'twt', 'wnut']:
            print(f'TRAINING SET: eriktks/conll2003')
        else:
            print(f'TRAINING SET: {self.data_loader.config}')

        self.model.train()

    def eval(self) -> None:
        if self.data_loader.corpus == 'eng':
            print(f'TEST SET: {self.data_loader.path}')
        else:
            print(f'TEST SET: {self.data_loader.config}')
        print(f"Original test set F1: {self.model.evaluate(eval_dataset=self.test_data)['eval_f1'] * 100:0.2f}")
        print(f"Lowercase test set F1: {self.model.evaluate(eval_dataset=self.lower_test)['eval_f1'] * 100:0.2f}")
        print(f"Uppercase test set F1: {self.model.evaluate(eval_dataset=self.upper_test)['eval_f1'] * 100:0.2f}")

    def grid_search(self) -> None:
        for lr in [1e-5, 5e-5, 1e-4]:
            for decay in [0.1, 0.05, 0.01]:
                for batch_size in [8, 16, 32]:
                    training_args = TrainingArguments(
                        output_dir='test_ner_model',
                        overwrite_output_dir=True,
                        learning_rate=lr,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=16,
                        num_train_epochs=5,
                        weight_decay=decay,
                        warmup_ratio=0.1,
                        eval_strategy='epoch',
                        save_strategy='epoch',
                        load_best_model_at_end=True,
                        push_to_hub=False,
                        report_to='none',
                        seed=self.seed,
                        disable_tqdm=True
                    )

                    train_set = self.train_data
                    if self.aug:
                        train_set = self.aug_data

                    trainer = Trainer(
                        model_init=self.model_init,
                        args=training_args,
                        train_dataset=train_set,
                        eval_dataset=self.dev_data,
                        processing_class=self.tokenizer,
                        data_collator=self.data_collator,
                        compute_metrics=self.compute_metrics
                    )

                    print(f'GRID SEARCH PARAMETERS: lr = {lr}, decay = {decay}, batch_size = {batch_size}')
                    trainer.train()