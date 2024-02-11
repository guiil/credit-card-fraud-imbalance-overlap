import os
import sys

IPYNB_FILENAME = 'run_estimator_analysis.ipynb'
CONFIG_FILENAME = '.config_ipynb'


def main(argv):
    with open(CONFIG_FILENAME, 'w') as f:
        f.write(' '.join(argv))

    dataset_path = argv[2]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    os.system(
        f'jupyter nbconvert --execute {IPYNB_FILENAME} --to html '
        f'--output-dir "results/{dataset_name}/" '
        f'--output "{dataset_name}.html"'
    )
    return None


if __name__ == '__main__':
    main(sys.argv)
