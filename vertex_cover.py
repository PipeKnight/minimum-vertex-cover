"""
Wrapper file for the 2 algorithms,
Input: -inst <filename.graph>, -alg[BnB|Approx], -time <cutoff in seconds>
Output: trace file and sol file
For example: to run it for the email.graph file with branch-and-bound
             with 30s as cutoff time: type in
             python vertex_cover.py -inst ./DATA/email.graph -alg BnB -time 30
"""

import argparse
import sys

import approximation
import branch_and_bound

def main():

    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('-inst', help='path to input graph', required=True)
    parser.add_argument(
        '-alg', help='algorithm choice[BnB|Approx|LS1|LS2]', required=True,
    )
    parser.add_argument('-time', help='cutoff time in seconds', required=True)

    # NOTE: despecated since we don't use random
    parser.add_argument('-seed', help='seed', required=False)

    args = parser.parse_args()
    input_graph = args.inst
    algs = args.alg
    cutoff_time = int(args.time)

    if args.seed is not None:
        seed = int(args.seed)

    if algs == 'BnB':
        branch_and_bound.run_bnb(input_graph, cutoff_time)
    else:
        if algs == 'Approx':
            approximation.run_approx(input_graph, cutoff_time)
        else:
            print('error: please choose among [BnB|Approx|LS1|LS2]')
            exit(1)


if __name__ == '__main__':
    main()
