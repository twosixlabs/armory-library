"""boilerplate experimental charmory feature for demo purposes"""


def a_little_teapot(n: int) -> str:
    return "short and stou" + ("t" * n)


def kookaburra() -> str:
    import inspect

    frame = inspect.currentframe()
    function_name = frame.f_code.co_name if frame is not None else ""
    return f"{function_name} sits in the old gum tree"
