import os


def get_file_path(path):
    if path.startswith('/'):
        return path
    if 'GQUANTROOT' in os.environ:
        ROOT = os.environ['GQUANTROOT']
    else:
        ROOT = os.getcwd()
    if os.path.exists(path):
        return path
    elif os.path.exists(ROOT+'/'+path):
        return ROOT+'/'+path
    else:
        print('current path', os.getcwd())
        print('input path', path)
        print('cannot find the file')
        raise Exception("File cannnot be found")
