from typing import Dict, Union, List
import pstats


def safe_float_cast(str_number: str) -> float:
    try:
        number = float(str_number)
    except ValueError:
        number = float('nan')
    return number


def pstats_to_dict(stats: pstats.Stats) -> List[Dict[str, Union[str, float]]]:
    formatted_stats = list()
    stats = 'ncalls' + stats.split('ncalls')[-1]
    stats = [line.rstrip().split(None, 5) for line in
             stats.split('\n')]
    for stat in stats[1:]:
        stats_dict = dict()
        if len(stat) >= 5:
            stats_dict['n_calls'] = stat[0]
            stats_dict['tot_time'] = stat[1]
            stats_dict['per_call1'] = stat[2]
            stats_dict['cum_time'] = stat[3]
            stats_dict['per_call2'] = stat[4]
            name = stat[5].split(':')
            stats_dict['name'] = \
                f"{name[0].split('/')[-1]}_line(function)_{name[1]}"
            formatted_stats.append(stats_dict)
    return formatted_stats


def smart_string_to_float(
        string: str,
        e: str = 'could not convert string to float') -> float:
    try:
        ret = float(string)
        return ret
    except ValueError:
        raise ValueError(e)


def smart_string_to_int(
        string: str,
        e: str = 'could not convert string to int') -> int:
    try:
        ret = int(string)
        return ret
    except ValueError:
        raise ValueError(e)
    return float('inf')


def parse_config(
        config: Dict[str, Union[str, float, int]]) -> Dict[
    str, Union[str, float, int]]:
    valid_dataset = ['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet']
    if config['dataset'] not in valid_dataset:
        raise ValueError(
            f"config.yaml: unknown dataset {config['dataset']}. " +
            f"Must be one of {valid_dataset}")
    valid_models = {
        'simple_convnetCifar','densenet201', 'densenet169', 'densenet161',
        'densenet121', 'densenet201Cifar', 'densenet169Cifar', 'densenet161Cifar',
        'densenet121Cifar', 'resnet18', 'resnet18Cifar', 'resnet34', 'resnet34Cifar', 'resnet50', 'resnet50Cifar',
        'resnet101', 'resnet101Cifar', 'resnet152', 'resnet152Cifar', 'resnext50', 'resnext101', 'wide_resnet50',
        'wide_resnet101', 'efficientnet_B0Cifar', 's_ViT', 'cait', 'swin_t', 'swin_s', 'swin_b', 'swin_l',
    }
    if config['network'] not in valid_models:
        raise ValueError(
            f"config.yaml: unknown model {config['network']}." +
            f"Must be one of {valid_models}")

    config['n_trials'] = smart_string_to_int(
        config['n_trials'],
        e='config.yaml: n_trials must be an int')
    e = 'config.yaml: init_lr must be a float or list of floats'
    if not isinstance(config['init_lr'], str):
        if isinstance(config['init_lr'], list):
            for i, lr in enumerate(config['init_lr']):
                if config['init_lr'][i] != 'auto':
                    config['init_lr'][i] = smart_string_to_float(lr, e=e)
        else:
            config['init_lr'] = smart_string_to_float(config['init_lr'], e=e)
    else:
        if config['init_lr'] != 'auto':
            raise ValueError(e)
    config['max_epochs'] = smart_string_to_int(
        config['max_epochs'],
        e='config.yaml: max_epochs must be an int')
    config['early_stop_threshold'] = smart_string_to_float(
        config['early_stop_threshold'],
        e='config.yaml: early_stop_threshold must be a float')
    config['early_stop_patience'] = smart_string_to_int(
        config['early_stop_patience'],
        e='config.yaml: early_stop_patience must be an int')
    config['mini_batch_size'] = smart_string_to_int(
        config['mini_batch_size'],
        e='config.yaml: mini_batch_size must be an int')
    # config['p'] = smart_string_to_int(
    #     config['p'],
    #     e='config.yaml: p must be an int')
    config['num_workers'] = smart_string_to_int(
        config['num_workers'],
        e='config.yaml: num_works must be an int')
    if config['loss'] != 'cross_entropy':
        raise ValueError('config.yaml: loss must be cross_entropy')
    for k, v in config['optimizer_kwargs'].items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                config['optimizer_kwargs'][k][i] = smart_string_to_float(val)
        else:
            config['optimizer_kwargs'][k] = smart_string_to_float(v)
    for k, v in config['scheduler_kwargs'].items():
        if isinstance(v, list):
            for i, val in enumerate(v):
                config['scheduler_kwargs'][k][i] = smart_string_to_float(val)
        else:
            config['scheduler_kwargs'][k] = smart_string_to_float(v)
    if config['cutout']:
        if config['n_holes'] < 0 or config['cutout_length'] < 0:
            raise ValueError('N holes and length for cutout not set')
    return config
