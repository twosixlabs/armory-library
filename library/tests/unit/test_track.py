from pathlib import Path
import tempfile
from unittest.mock import Mock, call, patch

import mlflow
import pytest

import armory.track as track

# Not _technically_ a unit test because we're using the real MLFlow API and
# performing real filesystem I/O, but we're using a temp directory that is
# deleted after each test so tests are still free of side-effects, isolated,
# and fast.
pytestmark = pytest.mark.unit


###
# Fixtures
###


@pytest.fixture(autouse=True)
def reset_params():
    track.reset_params()
    yield
    track.reset_params()


@pytest.fixture(autouse=True)
def temp_mlflow_uri():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(Path(tmpdir))
        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": Path(tmpdir).as_uri()}):
            yield
        mlflow.set_tracking_uri(orig_tracking_uri)


###
# Tests
###


###
# tracking_context
###


def test_tracking_context_isolates_parameters():
    track.track_param("key", "global")

    with track.tracking_context():
        track.track_param("key", "child")
        assert track.get_current_params() == {"key": "child"}

    assert track.get_current_params() == {"key": "global"}


def test_tracking_context_when_nested():
    track.track_param("key1", "global")

    with track.tracking_context(nested=True):
        track.track_param("key2", "child")
        assert track.get_current_params() == {"key1": "global", "key2": "child"}


###
# track_params
###


def test_track_params_when_inline():
    def func(a, b):
        assert a == "hello world"
        assert b == 42

    track.track_params(func)(a="hello world", b=42)

    params = track.get_current_params()
    params.pop("func._func")
    assert params == {"func.a": "hello world", "func.b": 42}


def test_track_params_when_decorator():
    @track.track_params
    def funct(a, b):
        assert a == 314159
        assert b == "hi"

    funct(a=314159, b="hi")

    params = track.get_current_params()
    params.pop("funct._func")
    assert params == {"funct.a": 314159, "funct.b": "hi"}


def test_track_params_with_prefix():
    @track.track_params(prefix="the_func")
    def func(a, b):
        assert a == "jenny"
        assert b == 8675309

    func(a="jenny", b=8675309)

    params = track.get_current_params()
    params.pop("the_func._func")
    assert params == {"the_func.a": "jenny", "the_func.b": 8675309}


def test_track_params_with_ignore():
    def func(a, b, c):
        assert a == "arg"
        assert b == 7
        assert c == "kwarg"

    track.track_params(func, ignore=["c"])("arg", b=7, c="kwarg")

    params = track.get_current_params()
    params.pop("func._func")
    assert params == {"func.b": 7}


def test_track_params_when_multiple_calls():
    mock = Mock()

    @track.track_params
    def func(**kwargs):
        mock(**kwargs)

    func(name="John Doe")
    func(name="Jane Doe")

    mock.assert_has_calls([call(name="John Doe"), call(name="Jane Doe")])

    params = track.get_current_params()
    params.pop("func._func")
    assert params == {"func.name": "Jane Doe"}


###
# track_init_params
###


def test_track_init_params_when_inline():
    class TestClass:
        def __init__(self, a, b):
            assert a == "hello world"
            assert b == 42

    track.track_init_params(TestClass)(a="hello world", b=42)

    params = track.get_current_params()
    params.pop("TestClass._func")
    assert params == {"TestClass.a": "hello world", "TestClass.b": 42}


def test_track_init_params_when_decorator():
    @track.track_init_params
    class ClassType:
        def __init__(self, a, b):
            assert a == 314159
            assert b == "hi"

    ClassType(a=314159, b="hi")

    params = track.get_current_params()
    params.pop("ClassType._func")
    assert params == {"ClassType.a": 314159, "ClassType.b": "hi"}


def test_track_init_params_with_prefix():
    @track.track_init_params(prefix="the_class")
    class TestClass:
        def __init__(self, a, b):
            assert a == "jenny"
            assert b == 8675309

    TestClass(a="jenny", b=8675309)

    params = track.get_current_params()
    params.pop("the_class._func")
    assert params == {"the_class.a": "jenny", "the_class.b": 8675309}


def test_track_init_params_with_ignore():
    class TestClass:
        def __init__(self, a, b, c):
            assert a == "arg"
            assert b == 7
            assert c == "kwarg"

    track.track_init_params(TestClass, ignore=["c"])("arg", b=7, c="kwarg")

    params = track.get_current_params()
    params.pop("TestClass._func")
    assert params == {"TestClass.b": 7}


def test_track_init_params_when_multiple_calls():
    mock = Mock()

    @track.track_init_params
    class TestClass:
        def __init__(self, **kwargs):
            mock(**kwargs)

    TestClass(name="John Doe")
    TestClass(name="Jane Doe")

    mock.assert_has_calls([call(name="John Doe"), call(name="Jane Doe")])

    params = track.get_current_params()
    params.pop("TestClass._func")
    assert params == {"TestClass.name": "Jane Doe"}


###
# track_evaluation
###


def test_track_evaluation_logs_params_with_mlflow():
    track.track_param("strparam", "str")
    track.track_param("intparam", 42)
    with track.track_evaluation("test"):
        pass

    run = mlflow.last_active_run()
    assert run
    assert run.data.params["strparam"] == "str"
    assert run.data.params["intparam"] == "42"


###
# trackable_context/Trackable
###


def test_trackable_context_records_on_trackables():
    class TestTrackable(track.Trackable):
        pass

    with track.trackable_context():
        track.track_param("key1", "value1")
        obj = TestTrackable()
        track.track_param("key2", "value2")

    assert track.get_current_params() == {}
    assert obj.tracked_params == {"key1": "value1", "key2": "value2"}


def test_trackable_context_and_decorated_trackables():
    @track.track_init_params
    class DecoratedTrackable(track.Trackable):
        def __init__(self, a, b):
            super().__init__()
            assert a == 314159
            assert b == "hi"

    with track.trackable_context():
        obj = DecoratedTrackable(a=314159, b="hi")

    obj.tracked_params.pop("DecoratedTrackable._func")
    assert obj.tracked_params == {
        "DecoratedTrackable.a": 314159,
        "DecoratedTrackable.b": "hi",
    }


def test_trackable_context_when_nested():
    track.track_param("key1", "global")

    class TestTrackable(track.Trackable):
        pass

    with track.trackable_context(nested=True):
        track.track_param("key2", "child")
        obj = TestTrackable()

    assert obj.tracked_params == {"key1": "global", "key2": "child"}
