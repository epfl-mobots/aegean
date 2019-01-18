#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import fnmatch
import glob
sys.path.insert(0, sys.path[0] + '/src/limbo/waf_tools')
sys.path.insert(0, sys.path[1] + '/waf_tools')

VERSION = '0.0.1'
APPNAME = 'aegean'

srcdir = '.'
blddir = 'build'

from waflib.Build import BuildContext
from waflib import Logs
from waflib.Tools import waf_unit_test
import eigen


def options(opt):
    opt.load('compiler_cxx')
    opt.load('compiler_c')
    opt.load('boost')
    opt.load('eigen')
    opt.load('libcmaes')
    opt.load('simu')

    opt.add_option('--tests', action='store_true',
                   help='compile tests or not', dest='tests')


def configure(conf):
    conf.load('compiler_cxx')
    conf.load('compiler_c')
    conf.load('waf_unit_test')
    conf.load('boost')
    conf.load('eigen')
    conf.load('libcmaes')
    conf.load('simu')

    conf.check(lib='pthread')
    conf.check_boost(
        lib='regex system filesystem unit_test_framework', min_version='1.46')
    conf.check_eigen(required=True)
    conf.check_libcmaes()
    conf.check_simu()

    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -xHost -mtune=native -unroll -g"
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -g -faligned-new"
    else:
        if int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
            common_flags = "-Wall -std=c++0x"
        else:
            common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -g -faligned-new"

    all_flags = common_flags + opt_flags
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
    conf.env.append_value("LINKFLAGS", ["-pthread"])
    print(conf.env['CXXFLAGS'])


def summary(bld):
    lst = getattr(bld, 'utest_results', [])
    total = 0
    tfail = 0
    if lst:
        total = len(lst)
        tfail = len([x for x in lst if x[1]])
    waf_unit_test.summary(bld)
    if tfail > 0:
        bld.fatal("Build failed, because some tests failed!")


def build(bld):
    if len(bld.env.INCLUDES_EIGEN) == 0 or len(bld.env.INCLUDES_BOOST) == 0:
        bld.fatal('Some libraries were not found! Cannot proceed!')

    if bld.options.tests:
        bld.recurse('src/tests')

    libs = 'BOOST EIGEN'

    # examples
    bld.program(features='cxx',
                install_path=None,
                source='examples/kmeans_example.cpp',
                includes='./src ./src/limbo/src',
                uselib=libs,
                target='kmeans_example')

    bld.program(features='cxx',
                install_path=None,
                source='examples/persistence_example.cpp',
                includes='./src ./src/limbo/src',
                uselib=libs,
                target='persistence_example')

    bld.program(features='cxx',
                install_path=None,
                source='examples/histogram_example.cpp',
                includes='./src ./src/limbo/src',
                uselib=libs,
                target='histogram_example')

    # exps
    bld.program(features='cxx',
                install_path=None,
                source='exp/zebra/zebra_etho_gen.cpp',
                includes='./src ./src/limbo/src',
                uselib=libs,
                target='zebra_etho_gen')

    bld.program(features='cxx',
                install_path=None,
                source='exp/zebra/zebra_nn_train.cpp',
                includes='./src ./src/nn/src ./src/limbo/src',
                uselib=libs,
                target='zebra_nn_train')

    bld.program(features='cxx',
                install_path=None,
                defines=['SMOOTH_LOSS'],
                source='exp/zebra/zebra_nn_train.cpp',
                includes='./src ./src/nn/src ./src/limbo/src',
                uselib=libs,
                target='zebra_nn_train_smooth')

    bld.program(features='cxx',
                install_path=None,
                defines=['USE_ORIGINAL_LABELS'],
                source='exp/zebra/zebra_mixed_sim.cpp',
                includes='./src ./src/nn/src ./src/limbo/src',
                uselib=libs,
                target='zebra_mixed_sim_original_labels')

    bld.program(features='cxx',
                install_path=None,
                source='exp/zebra/zebra_mixed_sim.cpp',
                includes='./src ./src/nn/src ./src/limbo/src',
                uselib=libs,
                target='zebra_mixed_sim')

    bld.program(features='cxx',
                install_path=None,
                source='exp/zebra/zebra_virtual_sim.cpp',
                includes='./src ./src/nn/src ./src/limbo/src',
                uselib=libs,
                target='zebra_virtual_sim')

    bld.program(features='cxx',
                install_path=None,
                defines=['USE_LIBCMAES'],
                source='exp/zebra/zebra_cmaes_nn_train.cpp',
                includes='./src ./src/nn/src ./src/limbo/src',
                uselib=libs + ' LIBCMAES',
                target='zebra_cmaes_nn_train')

    srcs = []
    incs = ['exp/zebra/sim/', 'src/nn/src', 'src/limbo/src', 'src/']
    nodes = bld.path.ant_glob('exp/zebra/sim/*.cpp', src=True, dir=False)
    for n in nodes:
        srcs += [n.bldpath()]

    bld.shlib(features='cxx cxxshlib',
              source=srcs,
              includes=incs,
              cxxflags=['-O3', '-fPIC', '-rdynamic'],
              uselib='SIMU EIGEN',
              target='aegean_simu')
    bld.env.LIBPATH_AEGEAN_SIMU = [os.getcwd() + '/build']
    bld.env.SHLIB_AEGEAN_SIMU = ['aegean_simu']
    bld.env.LIB_AEGEAN_SIMU = ['aegean_simu']

    bld.program(features='cxx',
                install_path=None,
                source='exp/zebra/zebra_sim.cpp',
                includes='./src ./src/nn/src ./src/limbo/src ./src/simu/src',
                use='SIMU AEGEAN_SIMU ' + libs,
                target='zebra_sim')

    bld.add_post_fun(summary)
