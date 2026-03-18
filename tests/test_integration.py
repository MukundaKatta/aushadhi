"""Integration tests for Aushadhi."""
from src.core import Aushadhi

class TestAushadhi:
    def setup_method(self):
        self.c = Aushadhi()
    def test_10_ops(self):
        for i in range(10): self.c.track(i=i)
        assert self.c.get_stats()["ops"] == 10
    def test_service_name(self):
        assert self.c.track()["service"] == "aushadhi"
    def test_different_inputs(self):
        self.c.track(type="a"); self.c.track(type="b")
        assert self.c.get_stats()["ops"] == 2
    def test_config(self):
        c = Aushadhi(config={"debug": True})
        assert c.config["debug"] is True
    def test_empty_call(self):
        assert self.c.track()["ok"] is True
    def test_large_batch(self):
        for _ in range(100): self.c.track()
        assert self.c.get_stats()["ops"] == 100
