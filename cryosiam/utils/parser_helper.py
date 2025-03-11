import argparse


def parser_helper(description=None):
    description = "Run CryoSiam training" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, required=True, help='')

    return parser
