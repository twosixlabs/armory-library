from pathlib import Path
import tempfile
from unittest.mock import Mock, call

import mlflow
import pytest

import charmory.track as track

# Not _technically_ a unit test because we're using the real MLFlow API and
# performing real filesystem I/O, but we're using a temp directory that is
# deleted after each test so tests are still free of side-effects, isolated,
# and fast.
pytestmark = pytest.mark.unit


###
# Fixtures
###


@pytest.fixture(autouse=True)
def temp_mlflow_uri():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(Path(tmpdir))
        yield
        mlflow.set_tracking_uri(orig_tracking_uri)


@pytest.fixture
def experiment_id():
    return mlflow.create_experiment("test")


###
# Tests
###


###
# track_params
###


def test_track_params_when_inline(experiment_id):
    def func(a, b):
        assert a == "hello world"
        assert b == 42

    with mlflow.start_run(experiment_id=experiment_id):
        track.track_params(func)(a="hello world", b=42)

    run = mlflow.last_active_run()
    assert run
    assert "func._func" in run.data.params
    assert run.data.params["func.a"] == "hello world"
    assert run.data.params["func.b"] == "42"


def test_track_params_when_decorator(experiment_id):
    @track.track_params
    def funct(a, b):
        assert a == 314159
        assert b == "hi"

    with mlflow.start_run(experiment_id=experiment_id):
        funct(a=314159, b="hi")

    run = mlflow.last_active_run()
    assert run
    assert "funct._func" in run.data.params
    assert run.data.params["funct.a"] == "314159"
    assert run.data.params["funct.b"] == "hi"


def test_track_params_with_prefix(experiment_id):
    @track.track_params(prefix="the_func")
    def func(a, b):
        assert a == "jenny"
        assert b == 8675309

    with mlflow.start_run(experiment_id=experiment_id):
        func(a="jenny", b=8675309)

    run = mlflow.last_active_run()
    assert run
    assert "the_func._func" in run.data.params
    assert run.data.params["the_func.a"] == "jenny"
    assert run.data.params["the_func.b"] == "8675309"


def test_track_params_with_ignore(experiment_id):
    def func(a, b, c):
        assert a == "arg"
        assert b == 7
        assert c == "kwarg"

    with mlflow.start_run(experiment_id=experiment_id):
        track.track_params(func, ignore=["c"])("arg", b=7, c="kwarg")

    run = mlflow.last_active_run()
    assert run
    assert "func._func" in run.data.params
    assert "func.a" not in run.data.params
    assert run.data.params["func.b"] == "7"
    assert "func.c" not in run.data.params


def test_track_params_when_no_active_run():
    func = Mock()

    track.track_params(func)(a="hello", b="world")

    func.assert_called_once_with(a="hello", b="world")
    try:
        assert not mlflow.last_active_run()
    except mlflow.exceptions.MlflowException:
        # Exception is raised when it has a last-run ID but the run doesn't exist
        # (because we delete the directory after each test)
        pass


def test_track_params_when_multiple_calls(experiment_id):
    mock = Mock()

    @track.track_params
    def func(**kwargs):
        mock(**kwargs)

    with mlflow.start_run(experiment_id=experiment_id):
        func(name="John Doe")
        func(name="Jane Doe")

    mock.assert_has_calls([call(name="John Doe"), call(name="Jane Doe")])

    run = mlflow.last_active_run()
    assert run
    assert "func._func" in run.data.params
    assert run.data.params["func.name"] == "John Doe"
    assert "func.1._func" in run.data.params
    assert run.data.params["func.1.name"] == "Jane Doe"


###
# track_init_params
###


def test_track_init_params_when_inline(experiment_id):
    class TestClass:
        def __init__(self, a, b):
            assert a == "hello world"
            assert b == 42

    with mlflow.start_run(experiment_id=experiment_id):
        track.track_init_params(TestClass)(a="hello world", b=42)

    run = mlflow.last_active_run()
    assert run
    assert "TestClass._func" in run.data.params
    assert run.data.params["TestClass.a"] == "hello world"
    assert run.data.params["TestClass.b"] == "42"


def test_track_init_params_when_decorator(experiment_id):
    @track.track_init_params
    class ClassType:
        def __init__(self, a, b):
            assert a == 314159
            assert b == "hi"

    with mlflow.start_run(experiment_id=experiment_id):
        ClassType(a=314159, b="hi")

    run = mlflow.last_active_run()
    assert run
    assert "ClassType._func" in run.data.params
    assert run.data.params["ClassType.a"] == "314159"
    assert run.data.params["ClassType.b"] == "hi"


def test_track_init_params_with_prefix(experiment_id):
    @track.track_init_params(prefix="the_class")
    class TestClass:
        def __init__(self, a, b):
            assert a == "jenny"
            assert b == 8675309

    with mlflow.start_run(experiment_id=experiment_id):
        TestClass(a="jenny", b=8675309)

    run = mlflow.last_active_run()
    assert run
    assert "the_class._func" in run.data.params
    assert run.data.params["the_class.a"] == "jenny"
    assert run.data.params["the_class.b"] == "8675309"


def test_track_init_params_with_ignore(experiment_id):
    class TestClass:
        def __init__(self, a, b, c):
            assert a == "arg"
            assert b == 7
            assert c == "kwarg"

    with mlflow.start_run(experiment_id=experiment_id):
        track.track_init_params(TestClass, ignore=["c"])("arg", b=7, c="kwarg")

    run = mlflow.last_active_run()
    assert run
    assert "TestClass._func" in run.data.params
    assert "TestClass.a" not in run.data.params
    assert run.data.params["TestClass.b"] == "7"
    assert "TestClass.c" not in run.data.params


def test_track_init_params_when_no_active_run():
    mock = Mock()

    class TestClass:
        def __init__(self, a, b):
            mock(a, b)

    track.track_init_params(TestClass)(a="hello", b="world")

    mock.assert_called_once_with("hello", "world")
    try:
        assert not mlflow.last_active_run()
    except mlflow.exceptions.MlflowException:
        # Exception is raised when it has a last-run ID but the run doesn't exist
        # (because we delete the directory after each test)
        pass


def test_track_init_params_when_multiple_calls(experiment_id):
    mock = Mock()

    @track.track_init_params
    class TestClass:
        def __init__(self, **kwargs):
            mock(**kwargs)

    with mlflow.start_run(experiment_id=experiment_id):
        TestClass(name="John Doe")
        TestClass(name="Jane Doe")

    mock.assert_has_calls([call(name="John Doe"), call(name="Jane Doe")])

    run = mlflow.last_active_run()
    assert run
    assert "TestClass._func" in run.data.params
    assert run.data.params["TestClass.name"] == "John Doe"
    assert "TestClass.1._func" in run.data.params
    assert run.data.params["TestClass.1.name"] == "Jane Doe"
