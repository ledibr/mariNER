from datasets import load_dataset, concatenate_datasets

conll = load_dataset('conll2003', trust_remote_code=True)
conll_lower = load_dataset('conll2003', trust_remote_code=True)
conll_upper = load_dataset('conll2003', trust_remote_code=True)
label_list = conll['train'].features[f"ner_tags"].feature.names

def lower(example):
  example["tokens"] = [token.lower() for token in example["tokens"]]
  return example

def upper(example):
  example["tokens"] = [token.upper() for token in example["tokens"]]
  return example

conll_lower = conll_lower.map(lower)
conll_upper = conll_upper.map(upper)
da_train = concatenate_datasets([conll["train"], conll_lower["train"], conll_upper["train"]])
lower_test = conll_lower['test']
upper_test = conll_upper['test']

id2label = {id: label for id, label in enumerate(label_list)}
label2id = {label:id for id, label in enumerate(label_list)}

if __name__ == '__main__':
    print(f"Original train: {conll['train'][0]['tokens']}")
    print(50 * '-')
    print(f"Data Augment train (Original): {da_train[0]['tokens']}")
    print(f"Data Augment train (Lowercase): {da_train[14041]['tokens']}")
    print(f"Data Augment train (Uppercase): {da_train[28082]['tokens']}")
    print(50 * '-')
    print(f"Original Test: {conll['test'][10]['tokens']}")
    print(f"Lower Test: {lower_test[10]['tokens']}")
    print(f"Upper Test: {upper_test[10]['tokens']}")

