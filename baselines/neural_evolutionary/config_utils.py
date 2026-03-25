# Copyright 2023 Andrew Holliday
# 
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
# 
# Transit Learning is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.

from lxml import etree
import datetime as dt

import numpy as np


RANGE_KEYS = set(['min', 'max', 'step'])


def parse_xml(xml_path):
    # this ensures that pretty-printing works when we write it back out
    parser = etree.XMLParser(remove_blank_text=True)
    return etree.parse(str(xml_path), parser)


def write_xml(xml_tree, filepath, **kwargs):
    """For matsim lxml, we always want to use certain lxml flags when writing.
    This wrapper function does that every time."""
    xml_tree.write(str(filepath), xml_declaration=True, pretty_print=True,
                   **kwargs)


def float_time_to_str(float_time):
    """When working with matsim, we often need to convert times in seconds to
    datetime strings."""
    return str(dt.timedelta(seconds=float_time))


def str_time_to_float(str_time):
    """Takes a string in the format %H:%M:%S, and returns an equivalent number
    of seconds as a floating-point value."""
    daytime = str_time_to_dt_time(str_time)
    delta = dt.datetime.combine(dt.date.min, daytime) - dt.datetime.min
    return delta.total_seconds()


def str_time_to_dt_time(str_time):
    """Assumes time is in the format %H:%M:%S.  But sometimes the hour value
    may be greater than 24, which datetime.strptime chokes on, so we've
    implemented our own parser here."""
    time_parts = [float(part) for part in str_time.split(':')]
    hour = time_parts[0]
    minute = time_parts[1]
    if len(time_parts) > 2:
        second = time_parts[2]
        subsecond = second % 1.0
        second -= subsecond
        ms = subsecond * (10**6)
    else:
        second = 0
        ms = 0
    # TODO are we sure we should wrap them around in this way?  I think so.
    if hour >= 24:
        hour -= 24

    # dt = datetime.datetime.strptime(str_time, "%H:%M:%S")
    daytime = dt.time(hour=int(hour), minute=int(minute), second=int(second),
                      microsecond=int(ms))
    return daytime


def od_fmt_time_to_float(od_fmt_time):
    hour = od_fmt_time // 100;
    minute = int(od_fmt_time % 100)
    second = (od_fmt_time % 1) * 60
    return hour * 3600 + minute * 60 + second


# utility functions for working with our own configuration files

def doesConfigMatchTarget(config, targetConfig):
    for targetKey, targetValue in targetConfig.items():
        if targetKey not in config:
            continue
        value = config[targetKey]

        if isinstance(targetValue, dict):
            if set(targetValue.keys()) == RANGE_KEYS:
                targetValue = _expand_range_config(targetValue)
            else:
                if doesConfigMatchTarget(value, targetValue):
                    continue
                else:
                    return False

        # it might have been a dict and become a list in the previous step
        if isinstance(targetValue, list):
            # config's value passes if it's in the target's list of values
            if value in targetValue:
                continue
            else:
                return False

        # base case
        if value != targetValue:
            return False
    return True


def expand_config_to_param_grid(base_config):
    names_with_configs = _expand_config_to_param_grid_helper(base_config)
    for name, cfg in names_with_configs:
        if 'name' not in cfg:
            cfg['name'] = ''
        if name:
            cfg['name'] += '_' + name
    _, configs = list(zip(*names_with_configs))
    return configs


def _expand_config_to_param_grid_helper(base_config):
    config_grid = [('', base_config)]
    # if element is list, create one with each element.
    for key in sorted(base_config):
        value = base_config[key]
        new_values_list = None
        is_list = isinstance(value, list)
        is_np_range = isinstance(value, dict) and \
            set(value.keys()) == RANGE_KEYS

        if is_list or is_np_range:
            # base case.
            if is_list:
                # create one config with each element
                new_values_list = value
            elif is_np_range:
                # create one with each element in the range
                new_values_list = _expand_range_config(value)
            # generate names
            new_values_names = [key + '=' + str(v) for v in new_values_list]
            new_values_list = list(zip(new_values_names, new_values_list))

        elif isinstance(value, dict):
            # recurse and get sub-grids
            new_values_list = _expand_config_to_param_grid_helper(value)

        if new_values_list is not None:
            # add the new values to the grid
            new_config_grid = []
            for basename, config in config_grid:
                for elemname, elem in new_values_list:
                    elem_config = config.copy()
                    elem_config[key] = elem
                    if basename and elemname:
                        # they're both non-empty, so concat nicely.
                        elemname = basename + '_' + elemname
                    else:
                        # one or both are empty, so just take what isn't.
                        elemname = basename + elemname
                    new_config_grid.append((elemname, elem_config))
            config_grid = new_config_grid

    return config_grid


def _expand_range_config(range_cfg):
    range_ = np.arange(range_cfg['min'], range_cfg['max'], range_cfg['step'])
    # return default python types for compatibility with numpy.
    range_ = list(map(np.asscalar, range_))
    return range_