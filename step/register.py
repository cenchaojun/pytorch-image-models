import copy
import logging
import types
from typing import List


class MyRegister(object):
    # 所有的映射都放在这个字典表里面去
    REGISTERED = {}

    @staticmethod
    def register(name, func):
        """
        Args:
            name(str) the name that identifues a dataset, e.g."coco_2014_train".
            func(callble) a callable which takes no arguments and returns a list of dicts.
        """
        assert callable(func), "you must register a function with Datasetcatalog.register"
        assert name not in MyRegister.REGISTERED, "Dataset{} is already registered".format(name)
        MyRegister.REGISTERED[name] = func
        print(name)
        print(func)


    @staticmethod
    def get(name,**kwargs):
        """
        call the registered function and return its results
        return list[dict]:dataset annotations
        """
        # print(**kwargs)
        try:
            f = MyRegister.REGISTERED[name]
            print(f)
        except KeyError:
            raise KeyError("Dataset:'{} is registered Available dataset are:{}".format(name,",".join(MyRegister.REGISTERED.keys())) )
        return f(**kwargs)



# lambda word:print_f1(word)
def a(word):
    return print_f1(word)


def print_f1(word):
    print('print_f1_{word}'.format(word=word))

def print_f2(word):
    print('print_f2_{word}'.format(word=word))

def print_f3(sname,word):
    print(f'print_f3_{sname}_{word}'.format(sname=sname,word=word))

if __name__ == '__main__':
    # 实例化注册机
    register_machine = MyRegister()
    # 开始注册
    # register_machine.register(name='f1',func=print_f1())
    # register_machine.register(name='f1',func=print_f1())
    register_machine.register('f1',lambda word:print_f1(word))
    register_machine.register('f2',lambda word:print_f2(word))
    register_machine.register('f3',lambda sname,word:print_f3(sname,word))
    # 开始调用
    _FUNC_NAME = 'f1'
    register_machine.get(_FUNC_NAME,word='hello_f1')
    _FUNC_NAME = 'f2'
    register_machine.get(_FUNC_NAME,word='hello_f2')

    _FUNC_NAME = 'f3'
    register_machine.get(_FUNC_NAME,sname='Marry',word='How are you')
