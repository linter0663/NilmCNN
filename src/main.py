import argparse

from model.model import NilmCNN


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, description='.')

    parser.add_argument('--lr', nargs='?', const=0.0001, type=float, default=0.0001,
                        help='Learning Rate for the Model.')
    parser.add_argument('--dropout', nargs='?', const=0.3, type=float, default=0.3,
                        help='Dropout value for all layers.')
    parser.add_argument('--validation_split', nargs='?', const=0.1, type=float, default=0.1)
    parser.add_argument('--evaluate_split', nargs='?', const=0.1, type=float, default=0.1)
    parser.add_argument('--batch_size', nargs='?', const=4, type=int, default=4)
    parser.add_argument('--l2', nargs='?', const=0., type=float, default=0.)
    parser.add_argument('--epochs', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--output_size', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--features', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--input_size', nargs='?', const=1440, type=int, default=1440)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--input_path', type=str, help='Path to input file.')
    parser.add_argument('--log_path', type=str, help='Path for TensorBoard log file.')

    return vars(parser.parse_args())


def main():
    args = get_args()

    model = NilmCNN(args)

    model.train()

    exit(0)


if __name__ == '__main__':
    main()
