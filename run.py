import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import wandb
import os
import time
import yaml
from scipy.sparse import coo_array, csr_array
import sklearn
import sklearn.metrics as metrics
import sklearn.utils.class_weight as class_weight
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from model import *

experiments_dir = './experiments'
model_dir = './models'

def main():
    sklearn.set_config(enable_metadata_routing=True)
    args = build_parser().parse_args()
    if args.runmode == 'train':
        train(args)
    elif args.runmode == 'eval':
        eval(args)
    elif args.runmode == 'predict':
        predict(args)

def train(args):
    with wandb.init(
        project='Data571-Final-Project',
        config=args,
    ) as run:
        if run.sweep_id: 
            log_dir = os.path.join(experiments_dir, run.config.model, f'sweep_{run.sweep_id}', f'{args.model}_{run.id}')
        else:
            log_dir = os.path.join(experiments_dir, run.config.model, f'{args.model}_{run.id}')

        os.mkdir(log_dir)
        model_path = os.path.join(model_dir, run.id)

        save_cmd_args(os.path.join(log_dir, 'args.txt'), sys.argv)
        print('CONFIG: \n', run.config)

        print('Loading data...')
        trainX = parse_sparseX_to_coo(args.trainx)
        trainX = coo_to_csr(trainX)
        trainY = parse_CT(args.trainy)
        devX = parse_sparseX_to_coo(args.devx)
        devX = coo_to_csr(devX)
        devY = parse_CT(args.devy)

        train_sample_weight = class_weight.compute_sample_weight('balanced', trainY)
        dev_sample_weight = class_weight.compute_sample_weight('balanced', devY)


        print('Training...')
        model = configure_model(run.config) 

        start = time.time()
        # also save best model during training
        model.train(trainX, trainY, args.model_path) 
        end = time.time()
        training_time = end - start
        print('Training time: "{:02d}h{:02d}m{:02d}s'.format(*sec_to_hr_min_sec(training_time)))

        start = time.time()
        train_pred = model.predict(trainX)
        train_acc, train_acc_balanced = accuracy(trainY, train_pred, train_sample_weight)
        end = time.time()
        train_inference_time = end - start
        print('Train inference time: "{:02d}h{:02d}m{:02d}s'.format(*sec_to_hr_min_sec(train_inference_time)))
        
        start = time.time()
        dev_pred = model.predict(devX)
        dev_acc, dev_acc_balanced = accuracy(devY, dev_pred, dev_sample_weight)
        end = time.time()
        dev_inference_time = end - start
        print('Dev inference time: "{:02d}h{:02d}m{:02d}s'.format(*sec_to_hr_min_sec(dev_inference_time)))

        print(f' \
            Train accurary: {train_acc:.2f} \n \
            Dev accuracy: {dev_acc: .2f}')
        print(f' \
            Train balanced accurary: {train_acc_balanced:.2f} \n \
            Dev balanced accuracy: {dev_acc_balanced: .2f}') 
            
        log = {
            'train_acc': train_acc,
            'train_acc_balanced': train_acc_balanced,
            'dev_acc': dev_acc,
            'dev_acc_balanced': dev_acc_balanced,
            'training_runtime': training_time,
            'train_inference_time': train_inference_time,
            'dev_inference_time': dev_inference_time,
            '24precision': precision_score(devY, dev_pred, labels=[24], average=None),
            '24recall': recall_score(devY, dev_pred, labels=[24], average=None),
            '11precision': precision_score(devY, dev_pred, labels=[11], average=None),
            '11recall': recall_score(devY, dev_pred, labels=[11], average=None),
            '0precision': precision_score(devY, dev_pred, labels=[0], average=None),
            '0recall': recall_score(devY, dev_pred, labels=[0], average=None)
        }
        run.log(log)

        log.update({'name': run.name, 'id': run.id})
        save_yaml(log, os.path.join(log_dir, 'log.yaml'))
    run.finish()

def predict(args):
    config = vars(args)
    devX = parse_sparseX_to_coo(args.devx)
    devX = coo_to_csr(devX)

    model = load_model(config)
    y_pred = model.predict(devX)
    np.savetxt(args.preds, y_pred, fmt='%d')

def eval(args):
    config = vars(args)
    devX = parse_sparseX_to_coo(args.devx)
    devX = coo_to_csr(devX)
    devY = parse_CT(args.devy)

    model = load_model(config)
    y_pred = model.predict(devX)
    y_pred_proba = model.predict_proba(devX)
    acc, balanced_acc = accuracy(devY, y_pred)
    print(acc)

    fig, ax = plt.subplots(figsize=(20, 20))
    metrics.ConfusionMatrixDisplay.from_predictions(devY, y_pred, normalize='true', cmap='Blues', values_format=".2f", ax=ax, colorbar=False)
    plt.xlabel('Predicted Label', fontsize=40)
    plt.ylabel('True Label', fontsize=40)
    plt.tight_layout()
    fig.savefig('cm.png')
    plt.close()

    
    plot_precision_recall_curves(devY, y_pred_proba)
    

def accuracy(y_true, y_pred, sample_weight=None):
    acc = metrics.accuracy_score(y_true, y_pred)
    balanced_acc = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight) if sample_weight is not None else None
    return acc, balanced_acc


def plot_precision_recall_curves(y_true, y_pred_proba, save_path="PR_f1.png"):
    classes = np.unique(y_true)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(10, 7))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
        y_pred_class = classes[np.argmax(y_pred_proba, axis=1)]
        # print(y_pred_class)
        f1 = f1_score(y_true, y_pred_class, average=None)[i]
        plt.plot( recall, precision, lw=2, 
                 label=f"{i} (F1 = {f1:.3f})")

    plt.title("Per-Class Precision–Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", ncol=5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_sparseX_to_coo(filename) -> coo_array:
    """
    Parse a plaintext file where each line consists of a triplet of ints, delimited by the space character
        1. The row index, i, (starting from 0)
        2. Thecolumnindex,j, (starting from 0)
        3. The value that belongs in the (i,j) th entry of the the NxD feature matrix

    Returns a sparse array in COOrdinate format. Also known as the 'ijv' or 'triplet' format.
    """
    data = np.loadtxt(filename, dtype=int)

    rows = data[:, 0]
    columns = data[:, 1]
    values = data[:, 2]

    n_rows = rows.max() + 1
    n_cols = columns.max() + 1

    data_coo = coo_array((values, (rows, columns)), shape=(n_rows, n_cols)) 
    return data_coo
    
def parse_CT(filename) -> np.ndarray:
    return np.loadtxt(filename, dtype=np.int64)

def coo_to_csr(coo: coo_array)-> csr_array:
    # some scikitlearn implementations only support int32 indices in sparse arrays
    csr = coo.tocsr()
    indices = csr.indices.astype(np.int32, copy=True) 
    indptr = csr.indptr.astype(np.int32, copy=True)
    return csr_array((csr.data.copy(), indices, indptr), shape=csr.shape)

def save_cmd_args(filename, argv):
    with open(filename, 'w') as file:
        file.write(' '.join(argv[1:]))

def save_yaml(config: dict, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def get_config_yaml(filename) -> dict:
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


def _mirror_for_help(action: argparse.Action, *groups: argparse._ArgumentGroup) -> None:
    """Make an existing action appear in additional help groups without re-registering it."""
    for g in groups:
        if action not in g._group_actions:  # avoid duplicates if called twice
            g._group_actions.append(action)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # Mutually exclusive mode selection (exactly one)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--train",   dest="runmode", action="store_const", const="train",
                       help="Train a DNN model from X and Y.")
    modes.add_argument("--predict", dest="runmode", action="store_const", const="predict",
                       help="Run prediction on a dev/test set using a trained model.")
    modes.add_argument("--eval",    dest="runmode", action="store_const", const="eval",
                       help="Evaluate a trained model on a dev/test set.")


    # IO paths
    trainx_arg = p.add_argument("--trainx", help="Path to training inputs .npy.")
    trainy_arg = p.add_argument("--trainy", help="Path to training targets .npy (y values or class labels).")
    devx_arg   = p.add_argument("--devx",   help="Path to dev/test inputs .npy.")
    devy_arg   = p.add_argument("--devy",   help="Path to dev/test targets .npy (y values or class labels).")
    p.add_argument("--model-path",  default=None,
                                help='Path to model file to write/read, default "model.npz".')
    preds      = p.add_argument("--preds",  help="Path to write predictions .npy (used with --predict).")

    p.add_argument("--transform", choices=['none', 'tf-idf', 'maxabs'], 
                               help='Transform to perform on data before passing into model')
    p.add_argument('--n-jobs', help='The number of parallel jobs to run. ' \
                                            'None means 1. -1 means using all processors.', type=int, default=-1)
    p.add_argument('--max-iter', type=int, help='Maximum iterations for solvers to take to converge.')
    p.add_argument('--class-weight', type=none_or_str, choices=['inv_sqrt', 'balanced', None], default=None)
    p.add_argument('--root', type=float, help='root to apply for inv_sqrt class weight scaling. Default 2 for square root', default=2.)
    p.add_argument('--feature-percentile', type=float, default=100.)


    # Tf-idf Transformer
    p.add_argument("--tfidf-norm", choices=['l1', 'l2'], default='l2',
                   help='Normalization to apply on tf-idf output rows')
    p.add_argument("--tfidf-use-idf", type=str_to_bool, default=True,
                   help="Enable inverse-document-frequency reweighting")
    p.add_argument('--tfidf-sublinear-tf', type=str_to_bool, default=False,
                   help="Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).")

    # Model selection
    p.add_argument("--model", choices=['knn', 'cnb', 'logreg', 'linearSVC'], required=True,
                   help="" \
                   "knn: K-nearest Neighbors, " \
                   "cnb: Complement Naive Bayes, " \
                   "logreg: Logistic Regression, " \
                   "linearSVC: Linear SVC")
        
    # ComplementNB 
    p.add_argument('--cnb-alpha', type=float, default=1.0,
                   help='Additive (Laplace/Lidstone) smoothing parameter (default 1.0)')
    p.add_argument('--cnb-norm', type=str_to_bool, default=False, 
                    help='Whether or not a second normalization of the weights is performed')
    

    # Logistic regression
    p.add_argument('--logreg-penalty', type=none_or_str, choices=['l1', 'l2', 'elasticnet', None], default='l2')
    p.add_argument('--logreg-C', type=float, 
                   help='Inverse of regularization strength; must be a positive float. ' \
                   'Like in support vector machines, smaller values specify stronger regularization.')
    p.add_argument('--logreg-class-weight', choices=['balanced', None], type=none_or_str, help='Weights associated with classes.')
    p.add_argument('--logreg-solver', choices=['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'])
    p.add_argument('--logreg-l1-ratio', type=float, help='The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1')

    # K-nearest neighbors
    p.add_argument('--knn-n-neighbors', help="Number of neighbors to use by default for kneighbors queries. Default 5.",
                            type=int, default=5)
    p.add_argument('--knn-weights', help='Weight function used in prediction', type=str, 
                            choices=['uniform', 'distance'], default='uniform')
    p.add_argument('--knn-metric', help='Distance metric for knn', 
                   choices=['manhattan', 'cityblock', 'euclidean', 'l1', 'cosine', 'l2'],
                            type=str, default='euclidean')

    # LinearSVC
    p.add_argument('--svc-penalty', type=str, choices=['l1', 'l2'], default='l2')

    p.add_argument('--svc-loss', type=str, choices=['hinge', 'squared_hinge'], default='squared_hinge')

    p.add_argument('--svc-C', type=float, help='Regularization strength (inverse). Must be positive.')

    # Help-only groups to mirror args
    g_train = p.add_argument_group("train mode arguments (used with --run-mode train)")
    g_pred  = p.add_argument_group("prediction mode arguments (used with --run-mode predict)")
    g_eval  = p.add_argument_group("evaluation mode arguments (used with --run-mode eval)")

    # Mirror actions into the groups for DISPLAY ONLY
    _mirror_for_help(trainx_arg, g_train)
    _mirror_for_help(trainy_arg, g_train)
    _mirror_for_help(devx_arg,   g_train, g_pred, g_eval)
    _mirror_for_help(devy_arg,   g_train, g_eval)
    _mirror_for_help(preds,      g_pred)

    return p

def args_to_dict(args):
    args_dict = {}
    for key, val in vars(args).items():
        key = key.replace("_", "-")
        if key not in args_to_dict:
            args_dict.update(key, val)
    return args_dict


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def none_or_str(value):
    if value == 'None':
        return None
    return value

def sec_to_hr_min_sec(seconds):
    # Convert int_seconds to HH:MM:SS
    hours, remainder = divmod(seconds, 3600)  
    minutes, seconds = divmod(remainder, 60)  
    return int(hours), int(minutes), int(seconds)

if __name__ == "__main__":
    main()
