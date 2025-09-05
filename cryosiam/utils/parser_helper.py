import argparse


def parser_helper(description=None):
    description = "Run CryoSiam" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, required=True, help='Path to the .yaml configuration file')
    parser.add_argument('--filename', type=str, required=False, help='Process only this specific tomogram filename',
                        default=None)
    return parser
