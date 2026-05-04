import argparse
from load_data import AugmentedDataLoader
from models import NERModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'corpus',
        nargs='*',
        choices=['eng', 'spa', 'nld', 'deu', 'fin', 'zul', 'twt', 'wnut']
    )
    parser.add_argument(
        '-t', '--type',
        choices=['base', 'aug', 'all'],
        default='all'
    )
    parser.add_argument(
        '-m', '--mode',
        choices=['test', 'search'],
        default='test'
    )

    args = parser.parse_args()

    lang_params = {
        'eng': {
            'base': {'lr': 5e-5, 'decay': 0.05, 'batch': 16},
            'aug': {'lr': 5e-5, 'decay': 0.05, 'batch': 16}
        },
        'spa': {
            'base': {'lr': 1e-5, 'decay': 0.05, 'batch': 8},
            'aug': {'lr': 5e-5, 'decay': 0.05, 'batch': 32}
        },
        'nld': {
            'base': {'lr': 1e-5, 'decay': 0.05, 'batch': 8},
            'aug': {'lr': 1e-5, 'decay': 0.1, 'batch': 8}
        },
        'deu': {
            'base': {'lr': 5e-5, 'decay': 0.01, 'batch': 32},
            'aug': {'lr': 5e-5, 'decay': 0.1, 'batch': 32}
        },
        'fin': {
            'base': {'lr': 5e-5, 'decay': 0.01, 'batch': 32},
            'aug': {'lr': 1e-5, 'decay': 0.1, 'batch': 16}
        },
        'zul': {
            'base': {'lr': 1e-4, 'decay': 0.1, 'batch': 16},
            'aug': {'lr': 5e-5, 'decay': 0.1, 'batch': 8}
        },
        'twt': {
            'base': {'lr': 5e-5, 'decay': 0.05, 'batch': 16},
            'aug': {'lr': 5e-5, 'decay': 0.05, 'batch': 16}
        },
        'wnut': {
            'base': {'lr': 5e-5, 'decay': 0.05, 'batch': 16},
            'aug': {'lr': 5e-5, 'decay': 0.05, 'batch': 16}
        }
    }

    for c in args.corpus:
        data = AugmentedDataLoader(c)

        models = {}
        if args.type != 'aug':
            params = lang_params[c]['base']
            models['XLM-R Baseline'] = NERModel('xlm-roberta-base', data_loader=data, lr=params['lr'], decay=params['decay'], train_batch_size=params['batch'])
        if args.type != 'base':
            params = lang_params[c]['aug']
            models['XLM-R Data Augmented'] = NERModel('xlm-roberta-base', data_loader=data, aug=True, lr=params['lr'], decay=params['decay'], train_batch_size=params['batch'])

        for name in models:
            print(80 * '=')
            print(name)
            print(80 * '=')
            if args.mode == 'search':
                models[name].grid_search()
            else:
                models[name].train()
                models[name].eval()

    print('Done!')