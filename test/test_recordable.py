
from pathlib import Path
from typing import cast
from codenets.recordable import DictRecordable
import os
import shutil
import pytest
from pyhocon import ConfigFactory

from codenets.codesearchnet.training_ctx import CodeSearchTrainingContext
from codenets.codesearchnet.query_code_siamese.training_ctx import QueryCodeSiameseCtx

test_dir = Path("./tmp-test")
cfg = Path("./test/conf/test.conf")


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want
    os.mkdir(test_dir)

    yield # this is where the testing happens

    # Teardown : fill with any logic you want
    shutil.rmtree(test_dir)


def test_dict_recordable():
    d = DictRecordable({
        'toto': 1,
        'tata': "titi",
        "tutu": 1.2345
    })

    assert d.save(test_dir / "d")
    d2 = DictRecordable.load(test_dir / "d")
    assert d == d2


def test_fullconf_recordable():
    training_ctx = CodeSearchTrainingContext.build_context_from_hocon(ConfigFactory.parse_file(cfg))
    assert training_ctx.save(test_dir / "f")

    training_ctx_2 = QueryCodeSiameseCtx.load(test_dir / "f")
    print("keys", training_ctx.keys(), training_ctx_2.keys())
    assert training_ctx.keys() == training_ctx_2.keys()




