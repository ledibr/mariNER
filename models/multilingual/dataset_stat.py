from preprocessing_utils import load_conll_file, decode_bio, get_reg_pandas_df
from datasets import DatasetDict, Dataset
from collections import Counter

#create the german dataset
deu_dataset=DatasetDict()
# deu_dataset["train"] = get_reg_pandas_df()("./data/deu/train.txt")
# deu_dataset["dev"] = get_reg_pandas_df()("./data/deu/dev.txt")
# deu_dataset["test"] = get_reg_pandas_df()("./data/deu/test.txt")
deu_dataset["train"] = Dataset.from_pandas(get_reg_pandas_df("./data/deu/train.txt"))
deu_dataset["dev"] = Dataset.from_pandas(get_reg_pandas_df("./data/deu/dev.txt"))
deu_dataset["test"] = Dataset.from_pandas(get_reg_pandas_df("./data/deu/test.txt"))
print(deu_dataset.num_rows)

#create the dutch dataset
dutch_dataset = DatasetDict()
dutch_dataset["train"] = Dataset.from_pandas(get_reg_pandas_df("./data/nld/train.txt"))
dutch_dataset["dev"] = Dataset.from_pandas(get_reg_pandas_df("./data/nld/dev.txt"))
dutch_dataset["test"] = Dataset.from_pandas(get_reg_pandas_df("./data/nld/test.txt"))
print(dutch_dataset.num_rows)

#create the spanish dataset
spa_dataset=DatasetDict()
spa_dataset["train"] = Dataset.from_pandas(get_reg_pandas_df("./data/spa/train.txt"))
spa_dataset["dev"] = Dataset.from_pandas(get_reg_pandas_df("./data/spa/dev.txt"))
spa_dataset["test"] = Dataset.from_pandas(get_reg_pandas_df("./data/spa/test.txt"))
print(spa_dataset.num_rows)

#create the finnish dataset
fin_dataset=DatasetDict()
fin_dataset["train"] = Dataset.from_pandas(get_reg_pandas_df("./data/fin/train.txt"))
fin_dataset["dev"] = Dataset.from_pandas(get_reg_pandas_df("./data/fin/dev.txt"))
fin_dataset["test"] = Dataset.from_pandas(get_reg_pandas_df("./data/fin/test.txt"))
print(fin_dataset.num_rows)

#create the zulu dataset
zul_dataset=DatasetDict()
zul_dataset["train"] = Dataset.from_pandas(get_reg_pandas_df("./data/zul/train.txt"))
zul_dataset["dev"] = Dataset.from_pandas(get_reg_pandas_df("./data/zul/dev.txt"))
zul_dataset["test"] = Dataset.from_pandas(get_reg_pandas_df("./data/zul/test.txt"))
print(zul_dataset.num_rows)

def count_mentions(dataset):
    """counts the mentions in an entire dataset"""
    mention_counter = Counter()
    mention_lists = [decode_bio(mention_list) for mention_list in dataset["labels"]]
    flat_mentions = [mention for mention_list in mention_lists for mention in mention_list]
    for mention in flat_mentions:
        mention_counter[mention.entity_type] += 1

    token_counter = 0
    token_lists = [token_list for token_list in dataset["text"]]
    flat_tokens = [token for token_list in token_lists for token in token_list]
    for token in flat_tokens:
        token_counter += 1
    return (mention_counter, token_counter)

deu_train_counts = count_mentions(deu_dataset["train"])
deu_dev_counts = count_mentions(deu_dataset["dev"])
deu_test_counts = count_mentions(deu_dataset["test"])

dutch_train_counts = count_mentions(dutch_dataset["train"])
dutch_dev_counts = count_mentions(dutch_dataset["dev"])
dutch_test_counts = count_mentions(dutch_dataset["test"])

spa_train_counts = count_mentions(spa_dataset["train"])
spa_dev_counts = count_mentions(spa_dataset["dev"])
spa_test_counts = count_mentions(spa_dataset["test"])

fin_train_counts = count_mentions(fin_dataset["train"])
fin_dev_counts = count_mentions(fin_dataset["dev"])
fin_test_counts = count_mentions(fin_dataset["test"])

zul_train_counts = count_mentions(zul_dataset["train"])
zul_dev_counts = count_mentions(zul_dataset["dev"])
zul_test_counts = count_mentions(zul_dataset["test"])


def get_metrics(lang, train, dev, test):
    """given dictionary counters from a dataset, gives some metrics"""
    print('=' * 50)
    print(f'{lang} Metrics')
    print('=' * 50)
    total_train = train[0]["PER"] + train[0]["LOC"] + train[0]["ORG"]
    total_dev = dev[0]["PER"] + dev[0]["LOC"] + dev[0]["ORG"]
    total_test = test[0]["PER"] + test[0]["LOC"] + test[0]["ORG"]
    total_mentions = total_train + total_dev + total_test
    total_per = train[0]["PER"] + dev[0]["PER"] + test[0]["PER"]
    total_org = train[0]["ORG"] + dev[0]["ORG"] + test[0]["ORG"]
    total_loc = train[0]["LOC"] + dev[0]["LOC"] + test[0]["LOC"]
    total_tokens = train[1] + dev[1] + test[1]
    print(f"{lang} total tokens ={total_tokens}")
    print(f"{lang} total mentions ={total_mentions}")
    print(f"{lang} percent PER ={total_per/total_mentions * 100}")
    print(f"{lang} percent ORG ={total_org/total_mentions * 100}")
    print(f"{lang} percent LOC = {total_loc/total_mentions * 100}")
    print('\n')
    print(f"{lang} train tokens ={train[1]}")
    print(f"{lang} train mentions ={total_train}")
    print(f"{lang} train percent PER = {train[0]["PER"]/total_train * 100}")
    print(f"{lang} train percent ORG = {train[0]["ORG"] / total_train * 100}")
    print(f"{lang} train percent LOC = {train[0]["LOC"] / total_train * 100}")
    print('\n')
    print(f"{lang} dev tokens ={dev[1]}")
    print(f"{lang} dev mentions ={total_dev}")
    print(f"{lang} dev percent PER = {dev[0]["PER"]/total_dev * 100}")
    print(f"{lang} dev percent ORG = {dev[0]["ORG"] / total_dev * 100}")
    print(f"{lang} dev percent LOC = {dev[0]["LOC"] / total_dev * 100}")
    print('\n')
    print(f"{lang} test mentions ={test[1]}")
    print(f"{lang} test mentions ={total_test}")
    print(f"{lang} test percent PER = {test[0]["PER"] / total_test * 100}")
    print(f"{lang} test percent ORG = {test[0]["ORG"] / total_test * 100}")
    print(f"{lang} test percent LOC = {test[0]["LOC"] / total_test * 100}")

#print(ita_train_counts, ita_dev_counts, ita_test_counts)
get_metrics("german", deu_train_counts, deu_dev_counts, deu_test_counts)
get_metrics("dutch", dutch_train_counts, dutch_dev_counts, dutch_test_counts)
get_metrics("spanish", spa_train_counts, spa_dev_counts, spa_test_counts)
get_metrics("finnish", fin_train_counts, fin_dev_counts, fin_test_counts)
get_metrics("zulu", zul_train_counts, zul_dev_counts, zul_test_counts)



