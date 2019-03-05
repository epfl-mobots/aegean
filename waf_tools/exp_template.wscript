#!/usr/bin/env python
# encoding: utf-8


def configure(conf):
    pass


def options(opt):
    pass


def build(bld):
    bld(features='cxx cxxprogram',
        source='@NAME.cpp',
        includes='. ../../src',
        target='@NAME',
        uselib='BOOST EIGEN LIBCMAES',
        use='aegean')
