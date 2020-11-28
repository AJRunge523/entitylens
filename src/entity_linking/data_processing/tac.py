"""
Processes the TAC-KBP 2010 entity linking dataset, combining it with pre-computed candidate sets and extracting context
for each of the query mentions from the source documents. Each of the 3 models used in our paper (CNN, RNN, Transformer)
have slightly different preprocessing requirements, so this outputs separate train and evaluation files formatted for
each of the three models
"""

import re
from wikireader import WikiRegexes
from xml.etree import ElementTree
import os
from nltk.tokenize import sent_tokenize
from enum import Enum
import argparse


class OutputType(Enum):
    SENT_CONTEXT = 1
    WORD_CONTEXT = 2


def load_queries(query_file, tab_file):
    """
    Load queries from the TAC-KBP datasets.
    :param query_file: XML-formatted query file containing query IDs, mention text, and gold entity information. Ex:
        tac_kbp_2010_english_entity_linking_training_queries.xml
    :param tab_file: Tab-separated file with query ID, gold entity ID, and entity type information, ex:
        tac_kbp_2010_english_entity_linking_training_KB_links.tab
    :return: A list of triples containing the query ID, mention text, and document ID for each query in the dataset.
    """
    queries = []
    qid = ""
    name = ""
    docid = ""
    entity = ""
    qmap = dict()
    with open(query_file, encoding='utf8') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith("<query"):
                qid = line[11:-2]
            if line.startswith("<name"):
                name = re.sub("<(/)*name>", "", line)
            if line.startswith("<docid"):
                docid = re.sub("<(/)*docid>", "", line)
            if line.startswith("<entity"):
                entity = re.sub("<(/)*entity>", "", line)
            if line.startswith("</query"):
                qmap[qid] = [name, docid, entity]
                qid = name = docid = entity = ""
    with open(tab_file, encoding='utf8') as infile:
        for line in infile:
            line = line.strip().split('\t')
            qid = line[0]
            gold_id = line[1]
            if gold_id.startswith('NIL'):
                qmap[qid][2] = 'NIL'
            else:
                qmap[qid][2] = gold_id
            queries.append([qid] + qmap[qid])
    queries.sort(key=lambda x: x[0])
    return queries


def gen_train_file(queries, cand_file, src_file_dir, out_path, context_type, word_context_size=64, strip_punc_num=True):
    """
    Creates a TAC entity linking dataset by combining a previously computed list of candidates with the query files
    and source documents from the TAC dataset, producing a double pipe-delimited file with the following format.
        - Query ID
        - Mention text
        - Mention context
        - Mention's 2nd context (used to split the left and right context windows, if necessary)
        - Document ID
        - Gold entity ID
        - Candidate entity IDs

    :param queries: Queries loaded from the TAC dataset by `load_queries`
    :param cand_file: Path to pipe-delimited entity candidate file, where each line contains a query ID,
        the gold entity ID, and a list of candidate entity IDs, previously aligned with Wikipedia document identifiers
    :param src_file_dir: Path to the TAC-KBP source documents.
    :param out_path: Path where the combined dataset will be written.
    :param context_type: Whether to take the surrounding sentence as the mention's context, or an X-word context window.
    :param word_context_size: The number of words on each side of the mention to take as the context. To reproduce our
        results, use 20 words for the RNN and 64 for the transformer.
    :param strip_punc_num: Whether to strip punctuation and replace all standalone numbers with a single token. Used
        in both the CNN and RNN preprocessing.
    """
    with open(cand_file, encoding='utf8') as cands, open(out_path, encoding='utf8', mode='w+') as outfile:
        if context_type == OutputType.SENT_CONTEXT:
            outfile.write("Query ID||Mention's Text Context||Document ID||Gold Entity ID||Candidate Entity IDs\n")
        else:
            outfile.write("Query ID||Mention's Left Context||Mention's Right Context||Document ID||"
                          "Gold Entity ID||Candidate Entity IDs\n")
        for cand_set, query in zip(cands, queries):
            cand_set = cand_set.strip().split("||")
            gold_id = cand_set[1]
            candidates = set(cand_set[2].split(','))

            qid = query[0]
            qname = query[1]
            docid = query[2]

            context_doc = ElementTree.parse(os.path.join(src_file_dir, docid + '.xml'))
            text_nodes = context_doc.getroot().find('BODY').find('TEXT').findall('POST')
            if len(text_nodes) == 0:
                text_nodes = context_doc.getroot().find('BODY').findall('TEXT')
            text = ''
            for n in text_nodes:
                text += ' ' + ' '.join(n.itertext()).strip().replace('\n', ' ')

            if '&amp;amp;' in qname:
                qname = qname.replace('&amp;amp;', '&')

            if context_type == OutputType.SENT_CONTEXT:

                context = None
                sents = sent_tokenize(text)

                for sent in sents:
                    if qname in sent:
                        context = sent
                if not context:
                    for i in range(len(sents)-1):
                        if qname in sents[i] + ' ' + sents[i+1]:
                            context = sents[i] + ' ' + sents[i+1]
                            break
                    else:
                        print(qid, qname, sents)
                        raise Exception
                # Write out a single-context entry
                outfile.write(f"{qid}||{qname}||{WikiRegexes._wikiToText(context, strip_punc_num=strip_punc_num)}||"
                              f"{docid}||{gold_id}||{','.join(candidates)}\n")
            else:
                try:
                    mention_loc = text.index(qname)
                except ValueError:
                    print(qname, "\n", text)
                    raise Exception("AAAAH")
                if mention_loc == -1:
                    print(qid, qname, text)
                    raise Exception

                left_text = text[:mention_loc]
                right_text = text[mention_loc+len(qname):]
                left = WikiRegexes._wikiToText(left_text, strip_punc_num=strip_punc_num)
                right = WikiRegexes._wikiToText(right_text, strip_punc_num=strip_punc_num)

                # Context size - 64 for BERT, 20 for RNN
                left_context = " ".join(left.strip().split(" ")[-word_context_size:])
                right_context = " ".join(right.strip().split(" ")[:word_context_size])

                # Write out a dual-context entry
                outfile.write(f"{qid}||{qname}||{left_context}||{right_context}||{docid}||{gold_id}||{','.join(candidates)}\n")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", help="Path to TAC-KBP directory, containing two folders, train and eval, each of"
                                            "which in turn should contain the query and tab files, as well as a directory"
                                            "called 'source_documents' containing the original news articles.", default="../data/tac2010")
    parser.add_argument("-c", "--candidates", help="Path to directory structure where candidate files are stored. Similarly to the above,"
                                                   "there should be two sub-directories, train and eval, each of which contain a file"
                                                   "called 'candidates'.txt'", default="../data/tac2010")
    parser.add_argument("-o", "--output", help="Directory where entity linking datasets will be written ", default="../data/tac2010/")

    args = parser.parse_args()

    train_queries = load_queries(f"{args.src}/train/tac_kbp_2010_english_entity_linking_training_queries.xml",
                                 f"{args.src}/train/tac_kbp_2010_english_entity_linking_training_KB_links.tab")
    eval_queries = load_queries(f"{args.src}/eval/tac_kbp_2010_english_entity_linking_evaluation_queries.xml",
                                f"{args.src}/eval/tac_kbp_2010_english_entity_linking_evaluation_KB_links.tab")


    gen_train_file(train_queries, f"{args.candidates}/train/candidates.txt",
                   f"{args.src}/train/source_documents/", f"{args.output}/train/cnn-candidates.txt", OutputType.SENT_CONTEXT, strip_punc_num=True)
    gen_train_file(eval_queries, f"{args.candidates}/eval/candidates.txt",
                   f"{args.src}/eval/source_documents/", f"{args.output}/eval/cnn-candidates.txt", OutputType.SENT_CONTEXT, strip_punc_num=True)

    gen_train_file(train_queries, f"{args.candidates}/train/candidates.txt",
                   f"{args.src}/train/source_documents/", f"{args.output}/train/rnn-candidates.txt", OutputType.WORD_CONTEXT, word_context_size=20, strip_punc_num=True)
    gen_train_file(eval_queries, f"{args.candidates}/eval/candidates.txt",
                   f"{args.src}/eval/source_documents/", f"{args.output}/eval/rnn-candidates.txt", OutputType.WORD_CONTEXT, word_context_size=20, strip_punc_num=True)

    gen_train_file(train_queries, f"{args.candidates}/train/candidates.txt",
                   f"{args.src}/train/source_documents/", f"{args.output}/train/transformer-candidates.txt",  OutputType.WORD_CONTEXT, word_context_size=64, strip_punc_num=False)
    gen_train_file(eval_queries, f"{args.candidates}/eval/candidates.txt",
                   f"{args.src}/eval/source_documents/", f"{args.output}/eval/transformer-candidates.txt", OutputType.WORD_CONTEXT, word_context_size=64, strip_punc_num=False)



if __name__ == "__main__":
    main()