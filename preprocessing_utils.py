from typing import Any
import pandas as pd


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


def get_pandas_df(path: str, test:Bool) -> pd.DataFrame:
    """given a filepath, returns a pandas dataframe with unaltered, lowercased, and uppercased data"""
    #call our function to process the data
    pre_pd = load_conll_file(path)
    #use listcomps to interact w tokens and data separately
    document_tokens = [document.tokens for document in pre_pd]
    document_labels = [document.labels for document in pre_pd]
    #now that we have access to the tokens themselves, we can lower and upper case them
    #this is a bit of functional programming fuckery, so let's break it down
    #map allows us to call a function on a sequence of imputs without a for loop
    #because of .lower() and .upper()'s syntax, we need to use a lambda expression
    #if it was a function we call like lower(x), we wouldn't use a lambda expression
    #map returns a map object, so we need to call list on it to get our actual list of cased strings
    #we could also have used a nested list comprehension but who wants to debug that
    lower_tokens = [list(map(lambda x: x.lower(), document.tokens)) for document in pre_pd]
    upper_tokens = [list(map(lambda x: x.upper(), document.tokens)) for document in pre_pd]
    #now we can create three separate dataframe, passing the columns as an iterable and manually naming the columns
    df = pd.DataFrame(zip(document_tokens, document_labels), columns=["Tokens", "Labels"])
    lower_df = pd.DataFrame(zip(lower_tokens, document_labels), columns=["Tokens", "Labels"])
    upper_df = pd.DataFrame(zip(upper_tokens, document_labels), columns=["Tokens", "Labels"])
    #now we can concatenate into one dataframe
    #figuring out how to concatenate pandas dfs is always trial and error so you're just gonna have to trust me that this does what we want
    if test == False:
        full_df = pd.concat([df, lower_df, upper_df], ignore_index=True)
        return full_df
    else:
        return df, lower_df, upper_df

