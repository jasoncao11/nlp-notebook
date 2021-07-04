# -*- coding: utf-8 -*-
import re

class SentenceHandler(object):

    delete_re = re.compile(r'\s+')
    split_re = re.compile(r'[。？！]')
    
    def __init__(self, min_length, max_length):
        """
        :param min_length: The minimum length a sentence should be to be considered.
        :param max_length: The maximum length a sentence should be to be considered.
        """
        self.min_length = min_length
        self.max_length = max_length

    def process(self, doc):
        """
        Processes a given document and turns them into sentences.
        :param doc: The raw document to process.
        :return: A list of sentences.
        """
        to_return = []
        doc = self.delete_re.sub('', doc)
        sents = self.split_re.split(doc)
        for c in sents:
            if self.max_length > len(c) > self.min_length:
                    to_return.append(c)
        return to_return

    def __call__(self, doc):
        return self.process(doc)