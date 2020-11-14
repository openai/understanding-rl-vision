from contextlib import contextmanager
import json
import tempfile
import subprocess
import os
from lucid.misc.io.writing import write_handle
from lucid.misc.io.reading import read_handle
from .json_encoding import encoder

_temp_config_dir = tempfile.mkdtemp(prefix="svelte3_")
_default_js_name = "App"
_default_div_id = "appdiv"


class CompileError(Exception):
    pass


def replace_file_extension(path, extension):
    """Replace the file extension of a path with a new extension."""
    if not extension.startswith("."):
        extension = "." + extension
    dir_, filename = os.path.split(path)
    if not filename.endswith(extension):
        filename = filename.rsplit(".", 1)[0]
    return os.path.join(dir_, filename + extension)


@contextmanager
def use_cwd(dir_):
    """Context manager for working in a different directory."""
    cwd = os.getcwd()
    try:
        os.chdir(dir_)
        yield
    finally:
        os.chdir(cwd)


def shell_command(command, **kwargs):
    """Wrapper around subprocess.check_output. Should be used with care:
    https://docs.python.org/3/library/subprocess.html#security-considerations
    """
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            shell=True,
            universal_newlines=True,
            **kwargs
        ).strip("\n")
    except subprocess.CalledProcessError as exn:
        raise CompileError(
            "Command '%s' failed with output:\n\n%s" % (command, exn.output)
        ) from exn


def compile_js(svelte_path, js_path=None, *, js_name=None, js_lint=None):
    """Compile Svelte to JavaScript.

    Arguments:
        svelte_path: path to input Svelte file
        js_path:     path to output JavaScript file
                     defaults to svelte_path with a new .js suffix
        js_name:     name of JavaScript global variable
                     defaults to _default_js_name
        js_lint:     whether to use eslint
                     defaults to True
    """
    if js_path is None:
        js_path = replace_file_extension(svelte_path, ".js")
    if js_name is None:
        js_name = _default_js_name
    if js_lint is None:
        js_lint = True

    eslint_config_fd, eslint_config_path = tempfile.mkstemp(
        suffix=".config.json", prefix="eslint_", dir=_temp_config_dir, text=True
    )
    eslint_config_path = os.path.abspath(eslint_config_path)
    rollup_config_fd, rollup_config_path = tempfile.mkstemp(
        suffix=".config.js", prefix="rollup_", dir=_temp_config_dir, text=True
    )
    rollup_config_path = os.path.abspath(rollup_config_path)

    svelte_dir = os.path.dirname(svelte_path) or os.curdir
    svelte_relpath = os.path.relpath(svelte_path, start=svelte_dir)
    js_relpath = os.path.relpath(js_path, start=svelte_dir)

    with open(eslint_config_fd, "w") as eslint_config_file:
        json.dump(
            {
                "env": {"browser": True, "es6": True},
                "extends": "eslint:recommended",
                "globals": {"Atomics": "readonly", "SharedArrayBuffer": "readonly"},
                "parserOptions": {"ecmaVersion": 2018, "sourceType": "module"},
                "plugins": ["svelte3"],
                "overrides": [{"files": ["*.svelte"], "processor": "svelte3/svelte3"}],
                "rules": {},
            },
            eslint_config_file,
        )

    with open(rollup_config_fd, "w") as rollup_config_file:
        rollup_config_file.write(
            """import svelte from 'rollup-plugin-svelte';
import resolve from 'rollup-plugin-node-resolve';
import { eslint } from 'rollup-plugin-eslint';
import babel from 'rollup-plugin-babel';
import commonjs from 'rollup-plugin-commonjs';
import path from 'path';

export default {
  input: '"""
            + svelte_relpath
            + """',
  output: {
    file: '"""
            + js_relpath
            + """',
    format: 'iife',
    name: '"""
            + js_name
            + """'
  },
  plugins: [
    eslint({
      include: ['**'],
      """
            + ("" if js_lint else "exclude: ['**'],")
            + """
      configFile: '"""
            + eslint_config_path
            + """'
    }),
    svelte({
      include: ['"""
            + svelte_relpath
            + """', '**/*.svelte']
    }),
    resolve({
      customResolveOptions: {
        paths: process.env.NODE_PATH.split( /[;:]/ )
      }
    }),
    commonjs(),
    babel({
      include: ['**', path.resolve(process.env.NODE_PATH, 'svelte/**')],
      extensions: ['.js', '.jsx', '.es6', '.es', '.mjs', '.svelte'],
      babelrc: false,
      cwd: process.env.NODE_PATH,
      presets: [['@babel/preset-env', {useBuiltIns: 'usage', corejs: 3}]]
    })
  ]
}
"""
        )

    with use_cwd(os.path.dirname(os.path.realpath(__file__)) or os.curdir):
        try:
            npm_root = shell_command("npm root --quiet")
        except CompileError as exn:
            raise CompileError(
                "Unable to find npm root.\nHave you installed Node.js?"
            ) from exn
        try:
            shell_command("npm ls")
        except CompileError as exn:
            shell_command("npm install")

    with use_cwd(svelte_dir):
        env = os.environ.copy()
        env["PATH"] = os.path.join(npm_root, ".bin") + ":" + env["PATH"]
        env["NODE_PATH"] = npm_root
        command_output = shell_command("rollup -c " + rollup_config_path, env=env)

    return {
        "js_path": js_path,
        "js_name": js_name,
        "command_output": command_output,
    }


def compile_html(
    input_path,
    html_path=None,
    *,
    props=None,
    precision=None,
    title=None,
    div_id=None,
    inline_js=None,
    svelte_to_js=None,
    js_path=None,
    js_name=None,
    js_lint=None
):
    """Compile Svelte or JavaScript to HTML.

    Arguments:
        input_path:   path to input Svelte or JavaScript file
        html_path:    path to output HTML file
                      defaults to input_path with a new .html suffix
        props:        JSON-serializable object to pass to Svelte script
                      defaults to an empty object
        precision:    number of significant figures to round numpy arrays to
                      defaults to no rounding
        title:        title of HTML page
                      defaults to html_path filename without suffix
        div_id:       HTML id of div containing Svelte component
                      defaults to _default_div_id
        inline_js:    whether to insert the JavaScript into the HTML page inline
                      defaults to svelte_to_js
        svelte_to_js: whether to first compile from Svelte to JavaScript
                      defaults to whether input_path doesn't have a .js suffix
        js_path:      path to output JavaScript file if compiling from Svelte
                      and not inserting the JavaScript inline
                      defaults to compile_js default
        js_name:      name of JavaScript global variable
                      should match existing name if compiling from JavaScript
                      defaults to _default_js_name
        js_lint:      whether to use eslint if compiling from Svelte
                      defaults to compile_js default
    """
    if html_path is None:
        html_path = replace_file_extension(input_path, ".html")
    if props is None:
        props = {}
    if title is None:
        title = os.path.basename(html_path).rsplit(".", 1)[0]
    if div_id is None:
        div_id = _default_div_id
    if svelte_to_js is None:
        svelte_to_js = not input_path.endswith(".js")
    if inline_js is None:
        inline_js = svelte_to_js

    if svelte_to_js:
        if inline_js:
            if js_path is None:
                js_path = replace_file_extension(input_path, ".js")
            prefix = "svelte_" + os.path.basename(js_path)
            if prefix.endswith(".js"):
                prefix = prefix[:-3]
            _, js_path = tempfile.mkstemp(
                suffix=".js", prefix=prefix + "_", dir=_temp_config_dir, text=True
            )
        try:
            compile_js_result = compile_js(
                input_path, js_path, js_name=js_name, js_lint=js_lint
            )
        except CompileError as exn:
            raise CompileError(
                "Unable to compile Svelte source.\n"
                "See the above advice or try supplying pre-compiled JavaScript."
            ) from exn
        js_path = compile_js_result["js_path"]
        js_name = compile_js_result["js_name"]
        command_output = compile_js_result["command_output"]
    else:
        js_path = input_path
        if js_name is None:
            js_name = _default_js_name
        command_output = None

    if inline_js:
        with read_handle(js_path, cache=False, mode="r") as js_file:
            js_code = js_file.read().rstrip("\n")
            js_html = "<script>\n" + js_code + "\n  </script>"
        js_path = None
    else:
        js_relpath = os.path.relpath(js_path, start=os.path.dirname(html_path))
        js_html = '<script src="' + js_relpath + '"></script>'

    with write_handle(html_path, "w") as html_file:
        html_file.write(
            """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>"""
            + title
            + '''</title>
</head>
<body>
  <div id="'''
            + div_id
            + """"></div>
  """
            + js_html
            + """
  <script>
  var app = new """
            + js_name
            + """({
    target: document.querySelector("#"""
            + div_id
            + """"),
    props: """
            + json.dumps(props, cls=encoder(precision=precision))
            + """
  });
  </script>
</body>
</html>"""
        )
    return {
        "html_path": html_path,
        "js_path": js_path if svelte_to_js else None,
        "title": title,
        "div_id": div_id,
        "js_name": js_name,
        "command_output": command_output,
    }
