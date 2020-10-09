from sklearn.linear_model import LogisticRegression, SGDRegressor
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from math import sqrt, log
import argparse


def create_paired_features(instances):
    train_data = list()
    train_data.append(np.array([x[1] for x in instances]))
    train_data.append(np.array([x[3] for x in instances]))
    train_data.append(np.stack([np.multiply(np.array(x[1]), np.array(x[3])) for x in instances]))
    train_data.append(np.stack([np.array(x[1]) - np.array(x[3]) for x in instances]))
    return np.concatenate(train_data, axis=1)


def train_regression_model(exp_name, train, test, outfile, logfile):
    train_data = np.array([x[1] for x in train])
    train_labels = [log(float(x[2])) for x in train]
    test_data = np.array([x[1] for x in test])
    test_labels = [log(float(x[2])) for x in test]

    outfile.write("{}\t{}\t".format(exp_name, len(train)))

    mean_train = sum(train_labels) / len(train_labels)

    model = SGDRegressor(loss="huber", max_iter=3000, tol=0.0001, alpha=0.1, random_state=2, n_iter_no_change=50)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    model.fit(train_data, train_labels)
    logfile.write("Results of experiment {}\n".format(exp_name))
    logfile.write("Train Instances: {}, Test Instances: {}\n".format(len(train_labels), len(test_labels)))
    logfile.write("==============Test================\n")
    test_predicted = model.predict(scaler.transform(test_data))

    loss = sqrt(mean_squared_error(test_labels, test_predicted))

    baseline_err = sqrt(mean_squared_error(test_labels, [mean_train for _ in test_labels]))

    print(f"Baseline: {baseline_err}")

    outfile.write(f"{loss}\n")
    logfile.write("*******\n*******\n")


def train_classification_model(exp_name, train, test, outfile, logfile):
    label_vocab = dict()
    ind2labels = []
    if len(train[0]) == 3: # Single entity task
        train_data = np.array([x[1] for x in train])
        train_labels = [x[2] for x in train]
        test_data = np.array([x[1] for x in test])
        test_labels = [x[2] for x in test]
    else: # Paired entity task
        train_data = create_paired_features(train)
        train_labels = [x[4] for x in train]
        test_data = create_paired_features(test)
        test_labels = [x[4] for x in test]

    outfile.write("{}\t{}\t".format(exp_name, len(train)))

    for l in train_labels:
        if l not in label_vocab:
            label_vocab[l] = len(label_vocab)
            ind2labels.append(l)

    train_labels = [label_vocab[l] for l in train_labels]
    test_labels = [label_vocab[x] for x in test_labels]

    label_set = set(train_labels)
    outfile.write("{}\t".format(len(label_set)))


    model = LogisticRegression(solver="newton-cg", multi_class="multinomial", max_iter=250)
    model.fit(train_data, train_labels)
    logfile.write("Results of experiment {}\n".format(exp_name))
    logfile.write("Train Instances: {}, Test Instances: {}\n".format(len(train_labels), len(test_labels)))
    logfile.write("==============Test================\n")
    test_predicted = model.predict(test_data)
    micro_f1 = output_results(test_predicted, test_labels, ind2labels, outfile, logfile)
    logfile.write("*******\n*******\n")
    return micro_f1


def output_results(predicted, labels, ind2labels, outfile, logfile):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted)
    cm = confusion_matrix(labels, predicted)
    for l in ind2labels:
        logfile.write("\t{}".format(l))
    logfile.write("\n")
    for i, row in enumerate(cm):
        logfile.write("{}\t{}\n".format(ind2labels[i], "\t".join(str(x) for x in row)))
    for l, p, r, f1 in zip(ind2labels, precision.tolist(), recall.tolist(), f1.tolist()):
        logfile.write("{}\t{}\t{}\t{}\n".format(l, p, r, f1))
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, predicted, average="micro")
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predicted,
                                                                                 average="macro")
    logfile.write("Test Micro -- {}/{}/{}\n".format(micro_precision, micro_recall, micro_f1))
    logfile.write("Test Macro -- {}/{}/{}\n".format(macro_precision, macro_recall, macro_f1))
    outfile.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t".format(micro_precision, micro_recall, micro_f1,
                                                    macro_precision, macro_recall, macro_f1))
    return micro_f1


def run_experiment(task_dir, task, model, out_dir):

    os.makedirs(os.path.join(out_dir, model), exist_ok=True)

    with open(f"{out_dir}/{model}/{task}-experiment-full-results.txt", encoding="utf8", mode="w+") as logfile, \
            open(f"{out_dir}/{model}/{task}-exp-results.txt", encoding="utf8", mode="w+") as outfile:

        scores = []
        subtasks = [x for x in os.listdir(task_dir)]

        # Put the regression task at the end to make the results files more readable.
        if "popularity-regression-links" in subtasks:
            subtasks.remove("popularity-regression-links")
            subtasks.append("popularity-regression-links")

        outfile.write("Dataset\tTrain Size\t# Labels\tTest Micro P\tTest Micro R\tTest Micro F1\t"
                      "Test Macro P\tTest Macro R\tTest Macro F1\n")

        for subtask in os.listdir(task_dir):
            print(f"Running task {task}-{subtask} for dataset {model}")
            data_dir = os.path.join(task_dir, subtask)
            train = pickle.load(open(os.path.join(data_dir, "train"), "rb"))
            test = pickle.load(open(os.path.join(data_dir, "test"), "rb"))

            if subtask == "popularity-regression-links":
                outfile.write("Dataset\tTrain Size\tRMSE\n")
                train_regression_model(subtask, train, test, outfile, logfile)
            else:
                subtask_f1 = train_classification_model(subtask, train, test, outfile, logfile)
                scores.append(subtask_f1)

            outfile.write("\n")
            outfile.flush()
            logfile.flush()

        if len(scores) > 0:
            outfile.write(f"Average subtask micro F1: {sum(scores) / len(scores)}\n")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", help="Directory containing probing task datasets", default="../probing-datasets")
    parser.add_argument("-t", "--tasks", help="Comma-separated list of tasks to run. Default: All", default="All")

    parser.add_argument("-o", "--output", help="Directory where results of probing experiments will be written.",
                        default="../experiments")

    parser.add_argument("-m", "--models", help="Comma-separated list of which embedding datasets will be evaluated. "
                                               "Default: All", default="All")
    args = parser.parse_args()

    dataset_dir = args.datasets

    output_dir = args.output

    models = args.models.split(",") if args.models.lower() != "all" else ["bow", "bert-base", "bert-large", "biggraph",
                                                                  "cnn", "rnn", "ganea", "wiki2vec"]

    tasks = args.tasks.split(",") if args.tasks.lower() != "all" else \
        ["context_words_high_freq", "context_words_mid_freq", "entity_types", "factual", "popularity",
             "relation_classification", "relation_classification+identification", "relation_detection",
             "relation_identification"
             ]

    for model in models:
        for task in tasks:
            task_dir = os.path.join(dataset_dir, model + "-datasets", task)
            run_experiment(task_dir, task, model, output_dir)


if __name__ == "__main__":
    main()
