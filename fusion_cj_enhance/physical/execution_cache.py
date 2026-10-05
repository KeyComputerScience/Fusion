"""Optional, exact in-process cache for unchanged full-prefix neural probes.

No scientific source is edited. Only supplied-model, twenty-step train calls
whose target buffer exceeds the frozen online train_buffer can be cached.
The frozen external runner still selects/calibrates/tests and audits service.
This module never opens physical data, labels, selections or result records.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import inspect
import contextlib
import io
import json
import marshal
import os
from pathlib import Path
import platform
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    answer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answer)
    return answer


def _field(hasher, tag, value):
    value = value.encode() if isinstance(value, str) else value
    tag = tag.encode()
    hasher.update(struct.pack('!Q', len(tag)));hasher.update(tag)
    hasher.update(struct.pack('!Q', len(value)));hasher.update(value)


def _array(hasher, tag, value):
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise TypeError('Only nonobject NumPy arrays are cacheable.')
    # Include layout as well as every logical element byte: BLAS may choose
    # a different execution path for equal arrays with different strides.
    metadata = dict(dtype=value.dtype.str,shape=value.shape,strides=value.strides,
        c_contiguous=bool(value.flags.c_contiguous),f_contiguous=bool(value.flags.f_contiguous),
        aligned=bool(value.flags.aligned),writeable=bool(value.flags.writeable))
    _field(hasher,tag+':metadata',json.dumps(metadata,sort_keys=True))
    _field(hasher,tag+':bytes',value.tobytes(order='C'))


def _functions(nn, original):
    return dict(train=original,loss_grad=nn.loss_grad,forward=nn.forward,
        sigmoid=nn.helper.sigmoid,sm=nn.helper.sm,ce=nn.helper.ce,
        lse=nn.helper.lse,pdf_weights=nn.helper.pdf_weights)


class ExactPrefixTrainCache:
    """Serial replay optimization; callers must not concurrently mutate inputs."""

    def __init__(self, nn, train_buffer, max_entries=4096):
        if hasattr(nn.train, '_exact_prefix_train_cache'):
            raise ValueError('A cache is already installed.')
        self.nn=nn;self.original=nn.train;self.signature=inspect.signature(nn.train)
        self.train_buffer=int(train_buffer);self.max_entries=int(max_entries)
        if self.train_buffer<0 or self.max_entries<1:raise ValueError('Invalid cache size.')
        self.functions=_functions(nn,self.original)
        self.deepcopier=copy.deepcopy
        self.codes={name:fun.__code__ for name,fun in self.functions.items()}
        self.defaults={name:fun.__defaults__ for name,fun in self.functions.items()}
        self.function_records={name:dict(
            source_sha256=hashlib.sha256(inspect.getsource(fun).encode()).hexdigest(),
            code_sha256=hashlib.sha256(marshal.dumps(fun.__code__)).hexdigest(),
            defaults=repr(fun.__defaults__)) for name,fun in self.functions.items()}
        self.source_records={str(Path(fun.__code__.co_filename).resolve()):sha(fun.__code__.co_filename)
            for fun in self.functions.values()}
        config=io.StringIO()
        with contextlib.redirect_stdout(config):np.show_config()
        self.runtime=dict(numpy=np.__version__,python=sys.version,platform=platform.platform(),
            numpy_build_config=config.getvalue(),blas_thread_environment={name:os.environ.get(name)
                for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS')})
        self.fingerprint=hashlib.sha256(json.dumps(dict(functions=self.function_records,
            sources=self.source_records,runtime=self.runtime),sort_keys=True).encode()).hexdigest()
        self.entries=OrderedDict();self.calls=0;self.hits=0;self.misses=0;self.bypasses=0
        self.evictions=0;self.hash_seconds=0.;self.train_seconds=0.

    def _verify_bindings(self):
        current=_functions(self.nn,self.original)
        for name,fun in self.functions.items():
            if current[name] is not fun or fun.__code__ is not self.codes[name] or fun.__defaults__ != self.defaults[name]:
                raise RuntimeError('Frozen training dependency changed: '+name)
        # All internal references must still resolve to the audited pure
        # numerical helpers, rather than a replacement mutable-global helper.
        for fun in self.functions.values():
            if 'np' in fun.__code__.co_names and fun.__globals__.get('np') is not np:
                raise RuntimeError('NumPy training binding changed.')
        if self.original.__globals__['loss_grad'] is not self.functions['loss_grad']:
            raise RuntimeError('loss_grad binding changed.')
        if self.nn.loss_grad.__globals__['forward'] is not self.functions['forward']:
            raise RuntimeError('forward binding changed.')
        if self.nn.loss_grad.__globals__['helper'] is not self.nn.helper or self.nn.forward.__globals__['helper'] is not self.nn.helper:
            raise RuntimeError('helper binding changed.')
        if self.original.__globals__['copy'] is not copy or copy.deepcopy is not self.deepcopier:
            raise RuntimeError('deepcopy binding changed.')
        if any(os.environ.get(name)!=value for name,value in self.runtime['blas_thread_environment'].items()):
            raise RuntimeError('Numerical thread environment changed.')
        for name in ('ce','pdf_weights'):
            if self.functions[name].__globals__['sm'] is not self.functions['sm']:
                raise RuntimeError('softmax binding changed.')

    def key(self, xs, y, rule, lr, steps, classes, models):
        h=hashlib.sha256();_field(h,'frozen_train_and_dependencies',self.fingerprint)
        _field(h,'rule',str(rule));_field(h,'lr',float(lr).hex())
        _field(h,'steps',str(int(steps)));_field(h,'classes',str(int(classes)))
        _field(h,'numpy_error_state',json.dumps(np.geterr(),sort_keys=True))
        _field(h,'sources',str(len(xs)));_array(h,'y',y)
        for i,x in enumerate(xs):_array(h,'x:'+str(i),x)
        _field(h,'models',str(len(models)))
        for i,model in enumerate(models):
            _field(h,'parameter_order:'+str(i),json.dumps(list(model)))
            for name,value in model.items():_array(h,'model:'+str(i)+':'+name,value)
        return h.hexdigest()

    def __call__(self, xs, y, rule, lr, steps, classes, models=None):
        self.calls+=1;self._verify_bindings()
        eligible=(type(steps) is int and steps==20 and len(y)>self.train_buffer and
            type(classes) is int and type(rule) is str and type(lr) is float and
            type(xs) is list and type(models) is list and all(type(model) is dict for model in models))
        if eligible:
            parameters=[value for model in models for value in model.values()]
            # Alias topology can affect the unchanged in-place update. The
            # frozen runner supplies distinct parameters; uncommon aliases
            # bypass the cache instead of approximating this extra state.
            eligible=all(type(a) is np.ndarray and not a.dtype.hasobject for a in [*xs,y,*parameters])
            eligible=eligible and not any(np.shares_memory(a,b)
                for i,a in enumerate(parameters) for b in parameters[i+1:])
        if not eligible:
            self.bypasses+=1
            return self.original(xs,y,rule,lr,steps,classes,models)
        started=time.perf_counter();key=self.key(xs,y,rule,lr,steps,classes,models)
        self.hash_seconds+=time.perf_counter()-started
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key)
            return copy.deepcopy(self.entries[key])
        self.misses+=1;started=time.perf_counter()
        result=self.original(xs,y,rule,lr,steps,classes,models)
        self.train_seconds+=time.perf_counter()-started
        self.entries[key]=copy.deepcopy(result)
        if len(self.entries)>self.max_entries:self.entries.popitem(last=False);self.evictions+=1
        return copy.deepcopy(self.entries[key])

    def report(self):
        return dict(cache_kind='pure serial in-process execution optimization',
            eligible='frozen list/dict/ndarray and scalar types, supplied distinct model arrays, steps==20, len(y)>frozen train_buffer',
            train_buffer=self.train_buffer,max_entries=self.max_entries,calls=self.calls,
            hits=self.hits,misses=self.misses,bypasses=self.bypasses,evictions=self.evictions,
            retained_entries=len(self.entries),hash_seconds=self.hash_seconds,train_seconds=self.train_seconds,
            fingerprint=self.fingerprint,functions=self.function_records,source_hashes=self.source_records,
            runtime=self.runtime,key='SHA256 of full input bytes, dtype/shape/layout, ordered initial parameters, scalar arguments and frozen dependencies',
            output='deepcopy on insertion and every cache return; work fields unmodified',
            scientific_sources_modified=False,cache_persistent=False,
            limitation='same numerical runtime; serial inputs; SHA256 collision resistance assumed; numerical failures remain original failures')


def install(nn, train_buffer, max_entries=4096):
    cache=ExactPrefixTrainCache(nn,train_buffer,max_entries)
    def cached_train(xs,y,rule,lr,steps,classes,models=None):
        return cache(xs,y,rule,lr,steps,classes,models)
    cached_train._exact_prefix_train_cache=cache
    cached_train.__wrapped__=cache.original
    nn.train=cached_train
    return cache


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('calibrate','test'))
    ap.add_argument('--runner',type=Path,default=ROOT/'run_external.py')
    ap.add_argument('--audit',type=Path,required=True)
    args=ap.parse_args();assert not args.audit.exists(),'Preserve prior execution audits.'
    runner=module('pure_cached_frozen_external_runner',args.runner)
    runner.verify();frozen=runner.p.verify()
    cache=install(runner.nn,frozen['base_cfg']['train_buffer'])
    before=dict(wrapper_sha256=sha(__file__),runner_sha256=sha(args.runner),
        algorithm_freeze_sha256=sha(runner.ROOT/'algorithm_freeze.json'),phase=args.phase,
        numerical_method_changed=False,optimization_only=True)
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.audit.write_text(json.dumps(dict(before=before,status='running',cache=cache.report()),indent=2)+'\n')
    status='failed'
    try:
        getattr(runner,args.phase)();status='completed'
    finally:
        args.audit.write_text(json.dumps(dict(before=before,status=status,cache=cache.report()),indent=2)+'\n')


if __name__=='__main__':main()
