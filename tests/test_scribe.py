import tempfile
import unittest
from pathlib import Path

from box import scribe


class ScribeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = scribe.connect(Path(self.tmp.name) / "s.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_register_and_find(self):
        scribe.register(self.conn, "Maria Rodriguez, Luis Rodriguez",
                        medical="insulin", phone="801-555-0101")
        hits = scribe.find_person(self.conn, "rodriguez")
        self.assertEqual(len(hits), 1)
        self.assertIn("insulin", hits[0]["medical"])

    def test_headcount_counts_members(self):
        scribe.register(self.conn, "A One, B Two, C Three")
        scribe.register(self.conn, "D Four")
        self.assertEqual(scribe.headcount(self.conn), 4)

    def test_supply_ledger_and_stock(self):
        scribe.supply(self.conn, "Water", 100, "litres")
        scribe.supply(self.conn, "water", -25, "litres")
        self.assertEqual(scribe.stock(self.conn)["water"], 75)

    def test_water_days_uses_sphere_standard(self):
        scribe.register(self.conn, "A, B")             # 2 people
        scribe.supply(self.conn, "water", 90, "litres")
        days = scribe.water_days_remaining(self.conn)  # 90/(15*2)
        self.assertAlmostEqual(days, 3.0)

    def test_briefing_includes_log_and_inventory(self):
        scribe.register(self.conn, "A")
        scribe.supply(self.conn, "blankets", 40)
        text = scribe.shift_briefing_text(self.conn)
        self.assertIn("Registered household: A", text)
        self.assertIn("40 blankets", text)
        self.assertIn("REGISTERED PEOPLE: 1", text)


if __name__ == "__main__":
    unittest.main()
