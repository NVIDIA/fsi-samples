import os
import cloudpickle
import base64
import pathlib


def get_file_path(path: str) -> str:
    """
    @path: the relative or absolute file path
    returns: absolute file path
    """
    if path.startswith('/'):
        return path
    if 'GREENFLOWROOT' in os.environ:
        ROOT = pathlib.Path(os.environ['GREENFLOWROOT'])
    else:
        ROOT = pathlib.Path(os.getcwd())
    if os.path.exists(path):
        return path
    path = pathlib.Path(path)
    if (ROOT/path).absolute().parent.exists():
        return str(ROOT/path)
    else:
        print('current path', os.getcwd())
        print('input path', path)
        print('cannot find the file')
        raise FileNotFoundError("File path cannnot be found")


def get_encoded_class(classObj):
    pickled = cloudpickle.dumps(classObj)
    encoding = base64.b64encode(pickled).decode()
    return encoding
