#!/usr/bin/env python3
from espnet2.tasks.classification import ClassificationTask


def get_parser():
    parser = ClassificationTask.get_parser()
    return parser


def main(cmd=None):
    r"""ClassificationTask training.

    Example:

        % python classification_train.py classification --print_config --optim adadelta \
                > conf/train_classification.yaml
        % python classification_train.py --config conf/train_classification.yaml
    """
    ClassificationTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
