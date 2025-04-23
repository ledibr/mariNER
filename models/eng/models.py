from load_data import conll, da_train, lower_test, upper_test, label_list, id2label, label2id
from transformers import (AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification,
                          TrainingArguments, Trainer)
import typing
import evaluate
import numpy as np

class NERModel:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.seqeval = evaluate.load('seqeval')
        self.conll = None
        self.da_train_data = None
        self.lower_test = None
        self.upper_test = None
        self.baseline = None
        self.da = None
        self.training_args = TrainingArguments(
            output_dir="test_ner_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to='none',
        )

    def model_init(self):
        return AutoModelForTokenClassification.from_pretrained(
        self.model_path, num_labels=9, id2label=id2label, label2id=label2id
    )

    #Figure out how to type annotate dataset dicts, but this will do for now
    def tokenize_and_align_labels(self, examples: dict[str,list[str]]) -> dict[str,list[str]]:
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

    #I... actually do not know what p is, whoops
    def compute_metrics(self, p) -> dict[str,float]:
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

    def baseline_train(self) -> None:
        self.baseline = Trainer(
            model_init=self.model_init,
            args=self.training_args,
            train_dataset=self.conll["train"],
            eval_dataset=self.conll["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.baseline.train()

    def da_train(self) -> None:
        self.da = Trainer(
            model_init=self.model_init,
            args=self.training_args,
            train_dataset=self.da_train_data,
            eval_dataset=self.conll["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.da.train()

    def eval(self) -> None:
        print("--------Lowercase Evaluation--------")
        print(f"Baseline F1: {self.baseline.evaluate(eval_dataset=self.lower_test)['eval_f1'] * 100:0.2f}")
        print(f"DA F1: {self.da.evaluate(eval_dataset=self.lower_test)['eval_f1'] * 100:0.2f}")
        print("--------Uppercase Evaluation--------")
        print(f"Baseline F1: {self.baseline.evaluate(eval_dataset=self.upper_test)['eval_f1'] * 100:0.2f}")
        print(f"DA F1: {self.da.evaluate(eval_dataset=self.upper_test)['eval_f1'] * 100:0.2f}")

    def compute_objective(self, metrics: dict[str, float]) -> float:
        """
        The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
        metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

        Args:
            metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

        Return:
            :obj:`float`: The objective to minimize or maximize
        """
        f1 = metrics.pop("eval_f1", None)
        _ = metrics.pop("epoch", None)
        return f1 if len(metrics) == 0 else sum(metrics.values())

    def finetune(self):
        self.baseline = Trainer(
            model_init=self.model_init,
            args=self.training_args,
            train_dataset=self.conll["train"],
            eval_dataset=self.conll["test"],
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        return self.baseline.hyperparameter_search(
            direction='maximize',
            compute_objective=self.compute_objective,
            n_trials=10
        )

if __name__ == '__main__':
    print(80 * '=')
    print('BERT-base-cased')
    print(80 * '=')
    bert_ner = NERModel('bert-base-cased')
    bert_ner.tokenize_and_align_all_data()
    # bert_ner.baseline_train()
    # bert_ner.da_train()
    # bert_ner.eval()
    print(bert_ner.finetune())

    # print(80 * '=')
    # print('XLM-R Base')
    # print(80 * '=')
    # xlmr_ner = NERModel('xlm-roberta-base')
    # xlmr_ner.tokenize_and_align_all_data()
    # xlmr_ner.baseline_train()
    # xlmr_ner.da_train()
    # xlmr_ner.eval()

    print('Done!')

