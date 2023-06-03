import os


def get_files(dirs: list[list[str]], path=['data'], ext=''):
    files = []
    for f in os.listdir(os.path.join(*path)):
        p = os.path.join(*path, f)
        if os.path.isfile(p) and p.endswith(ext):
            files.append(p)

    if len(dirs) == 0:
        return files

    for d in dirs[0]:
        path.append(d)
        p = os.path.join(*path)
        if os.path.isdir(p):
            files.extend(get_files(dirs[1:], path=path, ext=ext))
        path.pop()

    return files


def expand_dir_tree(dirs: list[list[str]], path: str | list[str] = 'data'):
    if not isinstance(path, list):
        path = list([path])
    paths = []

    if len(dirs) != 0:
        for d in dirs[0]:
            path.append(d)
            paths.extend(expand_dir_tree(dirs[1:], path))
            path.pop()
    else:
        paths.append(list(path))

    return paths


if __name__ == '__main__':
    options = [['automatic', 'manual'],
               ['luigi-circuit', 'moo-moo-meadows'],
               ['mario'],
               ['karts', 'bikes'],
               ['classic']]
    f = expand_dir_tree(options)
    print(f)
