import os
import argparse

import scripts.dataparser as dp


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Image processing using GPUs')
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--set-top', '-t', type=int, default=10)
    parser.add_argument('--set-threshold', '-th', type=int, default=50*50)
    parser.add_argument('--set-nmask', '-n', type=int, default=1)
    args = parser.parse_args()

    dp.store_data(f'{ROOT_DIR}/data', args.set_top, args.set_threshold, args.force, args.set_nmask)
