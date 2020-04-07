import matplotlib.pyplot as plt
import os
import collections
import numpy as np
from utils import plot_utils as pu

Result = collections.namedtuple('Result', 'progress dir root_dir')

COLORS = ['darkgreen', 'red', 'blue', 'orange', 'black', 'yellow', 'magenta' , 'cyan','purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'green', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

GAIL_EXPERT_ABSORBING = {
    'HalfCheetah-v1': 3558.468496,
    'Hopper-v1': 3585.477921,
    'Walker2d-v1': 5121.836947
}


def read_progress(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError('%s not found' % filename)
    kvs = dict()
    keys = []
    file = open(filename, 'r')
    for line in file.readlines():
        line = line[:-1]
        if len(kvs) == 0:
            for key in line.split(','):
                keys.append(key)
            for key in keys:
                kvs[key] = []
        else:
            for i, val in enumerate(line.split(',')):
                if len(val) > 0:
                    kvs[keys[i]].append(float(val))
    for key, val in kvs.items():
        kvs[key] = np.array(val)
    print('load %s successful' % filename)
    return kvs


def plot_attr(progress, attrs, fig_name, smooth=False, labels=None, y_lim=None, abs_y=False, log_scale=False):
    if labels is None: labels = attrs
    batch_size = 4
    dpi = 150
    if len(attrs) > batch_size:
        nrows, ncols = 1,  len(attrs) // batch_size + 1 if len(attrs) % batch_size != 0 else len(attrs) // batch_size
        figsize = (6*ncols, 4)
        f, axarr = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False, figsize=figsize, dpi=dpi)
        for start in range(0, len(attrs), batch_size):
            end = min(len(attrs), start + batch_size)
            for i in range(start, end):
                attr, label = attrs[i], labels[i]
                y = progress[attr]
                if abs_y:
                    y = np.abs(y)
                if smooth:
                    y = pu.smooth(y, radius=5)
                axarr[0][i//batch_size].plot(y, color=COLORS[(i % batch_size) % len(COLORS)], ls='-.', label=label)
            axarr[0][start//batch_size].legend()
    else:
        plt.figure(dpi=dpi, figsize=[6, 4])
        for attr, label in zip(attrs, labels):
            y = progress[attr]
            if abs_y:
                y = np.abs(y)
            if smooth:
                y = pu.smooth(y, radius=5)
            plt.plot(y, color=COLORS[attrs.index(attr) % len(COLORS)], ls='-.', label=label)
            plt.legend()
    if log_scale:
        plt.yscale('log')
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.savefig(fig_name, bbox_inches='tight')
    print('save fig into: %s' % fig_name)


def xy_fn(r):
    progress = r.progress
    splits = r.dir.split('-')
    agent_name = splits[0]
    if agent_name in {'acer'}:
        x_name, y_name = 'ACER/iter', 'ACER/episode_returns'
    elif agent_name in {'trpo', 'sac', 'td3', 'gail'}:
        x_name, y_name = 'Evaluate/iter', 'Evaluate/episode_returns'
    else:
        raise NotImplementedError('%s is not supported' % splits[0])
    assert x_name in progress.keys(), '{} not in {}'.format(x_name, list(progress.keys()))
    assert y_name in progress.keys(), '{} not in {}'.format(y_name, list(progress.keys()))
    x = progress[x_name]
    y = progress[y_name]
    index = 0
    while y[index] == 0.:
        index += 1
    y[:index] = y[index]
    return x, y


def main(root_dir):
    all_results = []
    for root, dirs, files in os.walk(root_dir):
        for dir_ in dirs:
            progress_path = os.path.join(root, dir_, 'progress.csv')
            if os.path.exists(progress_path):
                progress = read_progress(progress_path)
                all_results.append(Result(progress=progress, dir=dir_, root_dir=os.path.join(root, dir_)))
    print('load %s results' % len(all_results))

    plt.figure(dpi=300)
    fig, axarrs = pu.plot_results(all_results, xy_fn=xy_fn, average_group=True, shaded_err=False)
    plt.subplots_adjust(hspace=0.2)
    save_path = os.path.join(root_dir, 'result.png')
    plt.savefig(save_path, bbox_inches='tight')
    print('save result fig into %s' % save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='results')

    args = parser.parse_args()
    main(root_dir=args.root_dir)
