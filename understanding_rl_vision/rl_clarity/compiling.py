import os
from ..svelte3 import compile_js


def construct_path(relpath):
    dir_ = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_, relpath)


SVELTE_PATH = construct_path("svelte/interface.svelte")
JS_DIR_PATH = construct_path("js")
JS_PATH = construct_path("js/interface.js")


def recompile_js():
    if not os.path.exists(JS_DIR_PATH):
        os.mkdir(JS_DIR_PATH)
    print(compile_js(SVELTE_PATH, js_path=JS_PATH)["command_output"])


def get_compiled_js():
    if not os.path.isfile(JS_PATH):
        recompile_js()
    return JS_PATH
