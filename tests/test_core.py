"""Tests for Aushadhi."""
from src.core import Aushadhi
def test_init(): assert Aushadhi().get_stats()["ops"] == 0
def test_op(): c = Aushadhi(); c.track(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Aushadhi(); [c.track() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Aushadhi(); c.track(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Aushadhi(); r = c.track(); assert r["service"] == "aushadhi"
