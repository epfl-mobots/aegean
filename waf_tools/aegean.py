#!/usr/bin/env python
# encoding: utf-8

import os
import stat
import subprocess
import time
import threading
from waflib import Logs
from waflib.Tools import waf_unit_test


def check_lib(self, name, path):
    if self.env['DEST_OS'] == 'darwin':
        libname = name + '.dylib'
    else:
        libname = name + '.so'
    res = self.find_file(libname, path)
    lib = res[:-len(libname)-1]
    return res, lib


def create_exp(name, opt):
    if not os.path.exists('exp'):
        os.makedirs('exp')
    if os.path.exists('exp/' + name):
        Logs.pprint(
            'RED', 'ERROR: experiment \'%s\' already exists. Please remove it if you want to re-create it from scratch.' % name)
        return
    os.mkdir('exp/' + name)

    ws_tpl = ""
    for line in open("waf_tools/exp_template.wscript"):
        ws_tpl += line
    ws_tpl = ws_tpl.replace('@NAME', name)
    ws = open('exp/' + name + "/wscript", "w")
    ws.write(ws_tpl)
    ws.close()


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
