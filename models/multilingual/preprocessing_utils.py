from typing import Any, NamedTuple
import pandas as pd
import evaluate
import numpy as np

BEGIN = "B"
INSIDE = "I"
OUTSIDE = "O"
DELIM = "-"

label_list = ["O", "B-PER", "I-PER", "O-PER", "B-ORG", "I-ORG", "O-ORG", "B-LOC", "I-LOC", "O-LOC"]


class Document:
    """a class that trivially holds a sentences tokens and labels, based on AnnotatedSentence class from HW2"""
    def __init__(self, tokens: list[str], labels: list[str]) -> None:
        """initializes an instance of the document class, checking for tokens and mentions"""
        if not tokens:
            raise ValueError("No tokens provided.")
        elif not labels:
            raise ValueError("No label sequence provided.")
        self.tokens = tokens
        self.labels = labels #notice we're not decoding here, we want all the labels

    def __str__(self) -> str:
        """returns the document as a string"""
        return repr(self)

    def __repr__(self) -> str:
        """returns the document as a string"""
        return f"Document: {self.tokens}, {self.labels}"

    def __eq__(self, other : Any) -> bool:
        """indicates w boolean if another object is a document with the same tokens and labels"""
        #we will probably never use this but it is good coding practice to include it
        return (
                isinstance(other, Document)
                and self.tokens == other.tokens
                and self.labels == other.labels
        )


class Mention(NamedTuple):
    """An immutable mention with an entity type and start/end indices.

    Like standard slicing operations, the start index is inclusive
    and the end index is inclusive. For example, if the tokens of
    a sentence are ["Brandeis", "University", "is", "awesome"],
    an ORG mention for the first two tokens would have a start
    index of 0 and an end index of 2. Note that the length of the
    mention is simply end - start."""

    entity_type: str
    start: int
    end: int


def load_conll_file(path: str, delimiter: str = " ") -> list[Document]:
    """given the path to a conll format file and its delimiter, return list of annotated sentences for all sentences in the file """
    #no major changes to this from my implementation in hw2, just changing to document versus annotatedsentence
    sentence_list = []
    this_tokens = []
    this_labels = []
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.strip().split(delimiter)
            if line[0] == "-DOCSTART-": #if it's the beginning of a file
                this_tokens = []
                this_labels = []
            elif this_tokens == [] and line == [""]: #if it's the blank line after the beginning
                pass
            elif line == [""]: #if we see a blank line indicating EOS
                sentence_list.append(Document(this_tokens, this_labels))
                this_tokens = []
                this_labels = []
            else: #if we see a content line, add the contents
                this_tokens.append(line[0])
                this_labels.append(line[-1])
    #add the last line of the file
    if this_tokens:
        sentence_list.append(Document(this_tokens, this_labels))
    return sentence_list


def get_reg_pandas_df(path: str) -> pd.DataFrame:
    """given a filepath, returns a pandas dataframe with regular case data"""
    pre_pd = load_conll_file(path)
    document_tokens = [" ".join(document.tokens) for document in pre_pd]
    document_labels = [document.labels for document in pre_pd]
    df = pd.DataFrame(zip(document_tokens, document_labels), columns=["text", "labels"])
    return df


def get_aug_pandas_df(path: str, test: bool) -> pd.DataFrame|tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """given a filepath, returns a pandas dataframe with unaltered, lowercased, and uppercased data"""
    #call our function to process the data
    pre_pd = load_conll_file(path)
    #use listcomps to interact w tokens and data separately
    document_tokens = [" ".join(document.tokens) for document in pre_pd]
    document_labels = [document.labels for document in pre_pd]
    #now that we have access to the tokens themselves, we can lower and upper case them
    #this is a bit of functional programming fuckery, so let's break it down
    #map allows us to call a function on a sequence of imputs without a for loop
    #because of .lower() and .upper()'s syntax, we need to use a lambda expression
    #if it was a function we call like lower(x), we wouldn't use a lambda expression
    #map returns a map object, so we need to call list on it to get our actual list of cased strings
    #then we join the list bc the tokenizer expects a regular sentence :(
    lower_tokens = [" ".join(list(map(lambda x: x.lower(), document.tokens))) for document in pre_pd]
    upper_tokens = [" ".join(list(map(lambda x: x.upper(), document.tokens))) for document in pre_pd]
    #now we can create three separate dataframe, passing the columns as an iterable and manually naming the columns
    df = pd.DataFrame(zip(document_tokens, document_labels), columns=["text", "labels"])
    lower_df = pd.DataFrame(zip(lower_tokens, document_labels), columns=["text", "labels"])
    upper_df = pd.DataFrame(zip(upper_tokens, document_labels), columns=["text", "labels"])
    #we want to keep these separate in our test frame
    if test:
        return df, lower_df, upper_df
    #figuring out how to concatenate pandas dfs is always trial and error so you're just gonna have to trust me that this does what we want
    else:
        full_df = pd.concat([df, lower_df, upper_df], ignore_index=True)
        return full_df

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


def decode_bio(labels: list[str]) -> list[Mention]:
    in_mention = False
    start = 0
    end = start
    entity_type = ' '
    mentions = []
    count = 0

    for label in labels:
        label_parts = label.rsplit("-")
        boundary = label_parts[0]
        # If it's O, then label_parts will only have length 1, so use -1 indexing
        curr_type = label_parts[-1]
        # handles case if I starts without a B
        if not in_mention and boundary != OUTSIDE:
            entity_type = curr_type
            start = count
            in_mention = True
        elif in_mention:
            if not boundary == INSIDE or not curr_type == entity_type:
                end = count
                mentions.append(Mention(entity_type, start, end))

                if boundary == OUTSIDE:
                    in_mention = False
                # Handles case of two mentions next to each other (B-LOC B-LOC)
                # Handles invalid sequence (B-PER I-ORG)
                else:
                    start = count
                    entity_type = curr_type
        count += 1
    # If final label is an entity
    if in_mention:
        mentions.append(Mention(entity_type, start, count))

    return mentions