from preprocessing_utils import load_conll_file, decode_bio
from datasets import DatasetDict
from collections import Counter

#create the italian dataset
ita_dataset=DatasetDict()
ita_train = load_conll_file("./data/ita/train.txt")
ita_dataset["train"] = ita_train[:-7000]
ita_dataset["dev"] = ita_train[-7000:]
ita_dataset["test"] = load_conll_file("./data/ita/test.txt")

#create the dutch dataset
dutch_dataset = DatasetDict()
dutch_dataset["train"] = load_conll_file("./data/dut/train.txt")
dutch_dataset["dev"] = load_conll_file("./data/dut/dev.txt")
dutch_dataset["test"] = load_conll_file("./data/dut/test.txt")

#create the spanish dataset
spa_dataset=DatasetDict()
spa_dataset["train"] = load_conll_file("./data/spa/train.txt")
spa_dataset["dev"] = load_conll_file("./data/spa/dev.txt")
spa_dataset["test"] = load_conll_file("./data/spa/test.txt")

def count_mentions(dataset):
    """counts the mentions in an entire dataset"""
    counter = Counter()
    mention_lists = [decode_bio(mention_list) for mention_list in dataset["labels"]]
    flat_mentions = [mention for mention_list in mention_lists for mention in mention_list]
    for mention in flat_mentions:
        counter[mention.ent_type] += 1
    return counter

ita_train_counts = count_mentions(ita_dataset["train"])
ita_dev_counts = count_mentions(ita_dataset["dev"])
ita_test_counts = count_mentions(ita_dataset["test"])

dutch_train_counts = count_mentions(dutch_dataset["train"])
dutch_dev_counts = count_mentions(dutch_dataset["dev"])
dutch_test_counts = count_mentions(dutch_dataset["test"])

spa_train_counts = count_mentions(spa_dataset["train"])
spa_dev_counts = count_mentions(spa_dataset["dev"])
spa_test_counts = count_mentions(spa_dataset["test"])


def get_metrics(lang, train, dev, test):
    """given dictionary counters from a dataset, gives some metrics"""
    total_train = train["PER"] + train["LOC"] + train["ORG"]
    total_dev = dev["PER"] + dev["LOC"] + dev["ORG"]
    total_test = test["PER"] + test["LOC"] + test["ORG"]
    total_mentions = total_train + total_dev + total_test
    total_per = train["PER"] + dev["PER"] + test["PER"]
    total_org = train["ORG"] + dev["ORG"] + test["ORG"]
    total_loc = train["LOC"] + dev["LOC"] + test["LOC"]
    print(f"{lang} total mentions ={total_mentions}")
    print(f"{lang} percent PER ={total_per/total_mentions * 100}")
    print(f"{lang} percent ORG ={total_org/total_mentions * 100}")
    print(f"{lang} percent LOC = {total_loc/total_mentions * 100}")
    print(f"{lang} train percent PER = {train["PER"]/total_train * 100}")
    print(f"{lang} train percent ORG = {train["ORG"] / total_train * 100}")
    print(f"{lang} train percent LOC = {train["LOC"] / total_train * 100}")
    print(f"{lang} dev percent PER = {dev["PER"]/total_dev * 100}")
    print(f"{lang} dev percent ORG = {dev["ORG"] / total_dev * 100}")
    print(f"{lang} dev percent LOC = {dev["LOC"] / total_dev * 100}")
    print(f"{lang} test percent PER = {test["PER"] / total_test * 100}")
    print(f"{lang} test percent ORG = {test["ORG"] / total_test * 100}")
    print(f"{lang} test percent LOC = {test["LOC"] / total_test * 100}")

print(ita_train_counts, ita_dev_counts, ita_test_counts)
print(get_metrics("italian", ita_train_counts, ita_dev_counts, ita_test_counts))
print(get_metrics("dutch", dutch_train_counts, dutch_dev_counts, dutch_test_counts))
print(get_metrics("spanish", spa_train_counts, spa_dev_counts, spa_test_counts))



