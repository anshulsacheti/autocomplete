#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import unittest
import autocomplete
import nltk
import pdb

class TestProcessJsonData(unittest.TestCase):
    smallMessageSet = ["The cat jumps over the lazy fox"]

    head = autocomplete.createTrie(smallMessageSet)

    def test_datasize(self):
        self.assertEqual(len(autocomplete.process_json_data('./sample_conversations.json')[0]),16508)
        self.assertEqual(len(autocomplete.process_json_data('./sample_conversations.json')[1]),13981)

    def test_trie(self):
        self.assertEqual(list(self.__class__.head.getChildren().keys()), ["t"])
        self.assertEqual(list(self.__class__.head.getChild("T").getChildren().keys()), ["h"])
        self.assertEqual(self.__class__.head.getChild("T").getMessageMatches()[0], "The cat jumps over the lazy fox".lower())

    def test_autocomplete(self):
        self.assertEqual(autocomplete.autocomplete(self.__class__.head, "the", 1)['Completions'],["the cat jumps over the lazy fox"])
        self.assertEqual(autocomplete.autocomplete(self.__class__.head, "The", 1)['Completions'],["The cat jumps over the lazy fox"])
if __name__ == "__main__":
    unittest.main()
