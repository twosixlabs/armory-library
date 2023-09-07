"""boilerplate experimental charmory feature for demo purposes"""


def a_little_teapot(n: int) -> str:
    return "short and stou" + ("t" * n)


def kookaburra() -> str:
    import inspect

    function_name = inspect.currentframe().f_code.co_name
    return f"{function_name} sits in the old gum tree"
