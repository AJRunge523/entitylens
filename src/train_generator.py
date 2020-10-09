import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse


def generate_dataset_from_ids(dataset_dir, source, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(dataset_dir):

        if f.endswith('.txt'):
            with open(os.path.join(dataset_dir, f)) as infile:
                outfile = f.replace('.txt', '')
                instances = []
                for line in infile:
                    parts = line.strip().split('\t')
                    if len(parts) == 2: # Single entity probing task
                        v1 = source[parts[0]]
                        if v1 is None:
                            continue
                            # raise Exception(f"Missing embedding for entity {parts[0]}")
                        instances.append((parts[0], v1, parts[1]))
                    elif len(parts) == 3: # Pair entity probing task
                        v1 = source[parts[0]]
                        v2 = source[parts[1]]
                        if v1 is None:
                            continue
                            # raise Exception(f"Missing embedding for entity {parts[0]}")
                        elif v2 is None:
                            continue
                            # raise Exception(f"Missing embedding for entity {parts[1]}")
                        instances.append((parts[0], v1, parts[1], v2, parts[2]))
                    elif len(parts) == 6: #
                        v1 = source[parts[0]]
                        v2 = source[parts[1]]
                        label = parts[2]
                        instances.append((parts[0], v1, parts[1], v2, label))
                    else:
                        raise Exception(f"Unknown format, has {len(parts)} number of pieces.")
                pickle.dump(instances, open(os.path.join(output_dir, outfile), 'wb'))


class TSVTextSource:

    def __init__(self, *text_files):
        self.emb_dict = dict()
        for text_file in text_files:
            with open(text_file, encoding='utf8') as infile:
                for line in tqdm(infile):
                    line = line.strip().split('\t')
                    emb = np.array([float(x) for x in line[1:]])
                    self.emb_dict[line[0]] = emb

    def __getitem__(self, docid):
        if docid in self.emb_dict:
            return self.emb_dict[docid]
        print('Missing document ID: {}'.format(docid))
        return None

    def __iter__(self):
        sorted_keys = sorted([x for x in self.emb_dict])
        for item in sorted_keys:
            yield item

    def clear(self):
        self.emb_dict.clear()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embeds", help="Directory containing embedding files", required=True)
    parser.add_argument("-t", "--tasks", help="Directory containing task datasets", default="../probing-tasks")
    parser.add_argument("-o", "--output", help="Directory where probing datasets will be written ", default="../probing-datasets")
    parser.add_argument("-m", "--models", help="Comma-separated list of which embedding model(s) should be used to "
                                               "create probing datasets. Default: All - creates a dataset for each of the 8"
                                               "models used in the paper.", default="All")

    args = parser.parse_args()

    embed_dir = args.embeds
    task_dir = args.tasks
    output_dir = args.output
    models = args.models.split(",") if args.models.lower() != "all" else ["bow", "bert-base", "bert-large", "biggraph",
                                                                  "cnn", "rnn", "ganea", "wiki2vec"]

    for embed_name in models:
        print(f"Loading {embed_name} model embeddings.")
        source = TSVTextSource(f"{embed_dir}/{embed_name}.tsv")
        for task in os.listdir(task_dir):
            print(f"Creating datasets for task {task}")
            task_path = os.path.join(task_dir, task)
            for sub_task in os.listdir(task_path):
                path = os.path.join(task_path, sub_task)
                generate_dataset_from_ids(path, source, os.path.join(output_dir, embed_name + "-datasets", task, sub_task))
        source.clear()


if __name__ == '__main__':
    main()
