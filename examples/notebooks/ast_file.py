import ast
from collections import namedtuple
from pathlib import Path

Import = namedtuple("Import", ["module", "name", "alias"])


def get_imports(path):
    with open(path, "rb") as fh:
        root = ast.parse(fh.read(), path)
    # root = ast.parse(path.open().read())
    change = False
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            continue
        elif isinstance(node, ast.ImportFrom):
            for node_val in node.module.split("."):
                if node_val == "charmory":
                    # print(node_val)
                    node.module = node.module.replace("charmory", "armory")
                    change = True
        else:
            continue

    return change, ast.unparse(root)


filename = "/home/chris/armory-library/examples/src/armory/examples/image_classification/imagenet1k_resnet34_pgd.py"
# for imp in get_imports(filename): print(imp)


path = Path("/home/chris/armory-library/")
for p in path.rglob("*"):
    if p.suffix == ".py" and p.name != "ast_file.py" and p.name != "version.py":
        # print(p.name)
        # print(p)
        change, val = get_imports(p)
        if change:
            f = open(p, "w")
            f.write(val)
            f.close()


"""with open(filename) as f:
    tree = ast.parse(f.read(), filename=filename)
    for val in ast.iter_child_nodes(tree):
        if type(val) == ast.Import:
            print(val)
        """
