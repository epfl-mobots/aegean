#! /usr/bin/env python
# encoding: utf-8
# Vaios Papaspyros - 2018

"""
Quick n dirty samsar detection
"""

import os
import glob
import types
import subprocess
from waflib.Configure import conf


def options(opt):
    opt.add_option('--simu-includes', type='string',
                   help='path to simu includes', dest='simu_includes')
    opt.add_option('--simu-libs', type='string',
                   help='path to simu libs', dest='simu_libs')


@conf
def check_simu(conf, *k, **kw):
    includes_check = ['/usr/include/simu',
                      '/usr/local/include/simu', '/usr/include', '/usr/local/include']
    envincs = os.getenv('SIMU_INCLUDE_DIR')
    if envincs:
        includes_check += [envincs]
    if conf.options.simu_includes:
        includes_check = [conf.options.simu_includes]
    required = kw.get('required', False)

    conf.start_msg('Checking for simu includes')
    try:
        res = conf.find_file('simulation/simulation.hpp', includes_check)
        incl = res[:-len('simulation/simulation.hpp')-1]
        conf.env.INCLUDES_SIMU = [incl]
        conf.end_msg(incl)
    except:
        if required:
            conf.fatal('Not found in %s' % str(includes_check))
        conf.end_msg('Not found in %s' % str(includes_check), 'RED')

    libs_check = ['/usr/lib/simu', '/usr/local/lib/simu', '/usr/lib',
                  '/usr/local/lib']
    if conf.options.simu_libs:
        libs_check = [conf.options.simu_libs]
    conf.start_msg('Checking for simu libs')
    try:
        res = conf.find_file('libsimu.so', libs_check)
        lib = res[:-len('libsimu.so')-1]
        conf.env.LIBPATH_SIMU = [lib]
        conf.env.SHLIB_SIMU = ['simu']
        conf.env.LIB_SIMU = ['simu']
        conf.end_msg(lib)
    except:
        if required:
            conf.fatal('Not found in %s' % str(libs_check))
        conf.end_msg('Not found in %s' % str(libs_check), 'RED')
    return 1
