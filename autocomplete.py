#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import nltk
import pandas as pd
import numpy as np
import sklearn
import json
import os
from queue import Queue
import pdb
import string

class TrieNode:
    """
    Represents nodes in a prefix tree

    Input:
    value - char representing character for this node

    Class Variables (used for string embedding to store single copy of string):
    __numToStringDict - map of numbers to strings
    __stringToNumDict - map of strings to numbers
    __totalDictSize - num of keys in dictionary

    """
    __numToStringDict = {}
    __stringToNumDict = {}
    __totalDictSize = 0

    def __init__(self,value=None):
        # list of strings that pass through this node, ordered by occurrence
        self.messageMatches = []

        # count for each matching string
        self.messageCount = {}

        # children trie nodes
        self.children = {}

        # character contained at this node
        self.value = value

    @classmethod
    def getEmbeddedString(cls, num):
        """
        Returns string associated with embedding num

        Inputs:
        num - integer

        Return Type:
        str
        """
        return cls.__numToStringDict[num]

    @classmethod
    def embedString(cls, str):
        """
        Set embedding num associated with string
        If str already stored return key

        Inputs:
        str - string

        Return Type:
        None
        """
        if str.lower() in cls.__stringToNumDict:
            return cls.__stringToNumDict[str.lower()]
        else:
            cls.__stringToNumDict[str.lower()] = cls.__totalDictSize
            cls.__numToStringDict[cls.__totalDictSize] = str.lower()
            cls.__totalDictSize+=1
            return cls.__totalDictSize-1

    def updateMessageCounts(self, message):
        """
        Updates dictionary to keep track of count of some string message at this node
        This memory can be relinquished after model training
        Aids in ordering strings by instance count

        Inputs:
        message - string

        Return Type:
        None
        """
        message = message.lower()
        self.messageCount[message] = self.messageCount[message]+1 if message in self.messageCount else 1

    def createMessageMatches(self, resetMessageCount = False):
        """
        Create messageMatch array and update with messageCount data
        messageCount memory can be relinquished after model training
        Aids in ordering strings by instance count

        Inputs:
        None

        Return Type:
        None
        """
        messageSorted = sorted(self.messageCount,key=lambda x: self.messageCount[x])
        messageEmbedded = [TrieNode.embedString(message) for message in messageSorted]
        self.messageMatches = messageEmbedded
        if resetMessageCount:
            self.messageCount = {}

    def getMessageMatches(self):
        """
        Returns matching strings associated with node

        Inputs:
        None

        Return Type:
        list of str
        """
        return [TrieNode.getEmbeddedString(message) for message in self.messageMatches]

    def getMessageCount(self):
        """
        Returns message count associated with node

        Inputs:
        None

        Return Type:
        dict
        """
        return self.messageCount

    def getChildren(self):
        """
        Returns children node dict

        Inputs:
        None

        Return Type:
        dict
        """
        return self.children

    def getChild(self, value):
        """
        Returns child with child.value=value

        Inputs:
        value - char

        Return Type:
        TrieNode if child exists else None
        """
        value = value.lower()
        if value in self.children:
            return self.children[value]
        else:
            return None

    def updateChild(self, value):
        """
        Returns child with child.value=value if it exists otherwise it creates it

        Inputs:
        value - char

        Return Type:
        TrieNode
        """
        value = value.lower()
        if value in self.children:
            return self.children[value]
        else:
            self.children[value] = TrieNode(value)
            return self.children[value]

def legalizeOutput(baseMessage,foundMessages):
    """
    Fixes parentheses in output string and joins string tokens

    Inputs:
    baseMessage - str, input from user
    foundMessages - list of str, messages stored at final TrieNode

    Returns:
    list of strings
    """
    arr = []
    baseLength = len(baseMessage)

    #update output after input from user
    #just fixing apostrope at this time
    for message in foundMessages:
        str = baseMessage
        tokens = nltk.word_tokenize(message[baseLength:])
        for i,token in enumerate(tokens):
            if tokens[i][0]=="'":
                str+=tokens[i]
            elif i==0 and message[baseLength]!=" ":
                    str+=tokens[i]
            else:
                str+=" "+tokens[i]
        arr.append(str)
    return arr

def stripPunctuation(message):
    """
    Removes all punctuation from end of str

    Inputs:
    message - str

    Returns:
    str with no ending punctuation
    """
    strippedMessage = message
    # Ignore ending punctuation
    while len(strippedMessage)>0 and strippedMessage[-1] in ["?", ".", "!", "'"]:
        strippedMessage = strippedMessage[:-1]
    return strippedMessage

def autocomplete(head, message, k):
    """
    Traverses Trie to find matches for message and returns top k matches

    Inputs:
    head - TrieNode root
    message - str, user input
    k - int, number of strings to return

    Returns:
    dict
    """

    strippedMessage = stripPunctuation(message)
    currNode = head.getChild(strippedMessage[0])

    # Traverse Trie
    for char in strippedMessage[1:]:
        if currNode == None:
            break
        currNode = currNode.getChild(char)

    # Might not find base string
    if currNode:
        matches = legalizeOutput(strippedMessage,currNode.getMessageMatches()[-k:][::-1])
        return {"Completions":matches}
    else:
        return {"Completions": [""]}

def process_json_data(jsonFilePath):
    """
    Processes all customer and service rep messages and splits them

    Inputs:
    jsonFilePath - str path to file

    Returns:
    list of list of strings
    """
    # Read in json data
    with open(jsonFilePath) as json_data:
        data = json.load(json_data)

    customerMessages = []
    serviceRepMessages = []

    # Split up messages
    # Make a corpus of service and customer messages
    for issue in data["Issues"]:
        for message in issue["Messages"]:
            for sent in nltk.sent_tokenize(message["Text"]):
                # https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
                sent = stripPunctuation(sent)
                sent = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in nltk.word_tokenize(sent)]).strip()
                if message["IsFromCustomer"]:
                    customerMessages.append(sent)
                else:
                    serviceRepMessages.append(sent)

    return [serviceRepMessages, customerMessages]

def createTrie(messages):
    """
    Processes all service rep messages and generates Trie

    Inputs:
    messages - list of strings of service rep statements

    Returns:
    TrieNode
    """
    head = TrieNode()

    # Add each message to trie
    for message in messages:

        # Ignore ending punctuation, same result but matches better
        strippedMessage = stripPunctuation(message)

        # Split up messages so not all messages stored in head
        if strippedMessage!="":
            currNode = head.updateChild(strippedMessage[0])
        else:
            currNode = head.updateChild("")

        # Used first char earlier
        for char in strippedMessage[1:]:
            currNode.updateMessageCounts(strippedMessage)
            currNode = currNode.updateChild(char)

    # Update messageMatches in each child based off number of occurrences of a string
    queue = Queue()
    queue.put(head)
    while queue.qsize():
        currNode = queue.get()
        currNode.createMessageMatches(resetMessageCount = True)

        #Add children to next iter
        children = currNode.getChildren()
        for pointer in children:
            queue.put(children[pointer])

    return head

def process_data():
    """
    Wrapper call for process_json_data and createTrie

    Inputs:
    None

    Returns:
    TrieNode
    """
    serviceText, customerText = process_json_data('./sample_conversations.json')
    trie = createTrie(serviceText)

    return trie

if __name__ == "__main__":
    # input = input()
    serviceText, customerText = process_json_data('./sample_conversations.json')
    # trie = process_data()
    # print(autocomplete(trie, "what's your account num", 3))
    pdb.set_trace()
