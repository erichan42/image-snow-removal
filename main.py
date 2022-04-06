import os
import argparse

import scripts.dataparser as dp

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Image processing using GPUs')
parser.add_argument('--init-data', '-i', action='store_true')
parser.add_argument('--force', '-f', action='store_true')
parser.add_argument('--set-top', '-t', type=int, default=10)
args = parser.parse_args()

if args.init_data is not None:
    dp.store_data(f'{ROOT_DIR}/data', args.set_top, args.force)
