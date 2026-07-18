"""Spoken supply ledger — parsing and ledger effects."""
import unittest

from box import quartermaster, scribe


def fresh():
    conn = scribe.connect(":memory:")
    scribe.supply(conn, "water", 1000, "L")
    scribe.supply(conn, "MRE meals", 100, "")
    scribe.register(conn, "One Person")
    return conn


class TransactionTest(unittest.TestCase):
    def test_give_out_water_deducts(self):
        conn = fresh()
        r = quartermaster.maybe_answer(
            "We gave out 40 liters of water.", conn)
        self.assertIn("Distributed 40", r)
        self.assertEqual(scribe.stock(conn)["water"], 960)
        self.assertIn("days at Sphere rates", r)

    def test_received_meals_adds_to_existing_line(self):
        conn = fresh()
        r = quartermaster.maybe_answer("We received 200 meals.", conn)
        self.assertIn("Received 200", r)
        self.assertEqual(scribe.stock(conn)["mre meals"], 300)

    def test_gallons_convert_for_water(self):
        conn = fresh()
        quartermaster.maybe_answer("we got 10 gallons of water", conn)
        self.assertAlmostEqual(scribe.stock(conn)["water"], 1037.9, 1)

    def test_filler_word_before_gallons_still_converts(self):
        # verbatim live failure: 'more' broke unit capture and 10 gallons
        # was ledgered as 10 liters
        conn = fresh()
        r = quartermaster.maybe_answer(
            "we just received 10 more gallons of water", conn)
        self.assertAlmostEqual(scribe.stock(conn)["water"], 1037.9, 1)
        self.assertIn("10 gallons", r)
        self.assertIn("37.9 liters", r)

    def test_spoken_numbers(self):
        conn = fresh()
        quartermaster.maybe_answer("we gave out twenty blankets", conn)
        self.assertEqual(scribe.stock(conn)["blankets"], -20)

    def test_cpr_three_times_is_not_a_supply(self):
        conn = fresh()
        self.assertIsNone(quartermaster.maybe_answer(
            "I gave him CPR 3 times", conn))


class QueryTest(unittest.TestCase):
    def test_how_much_water_left(self):
        conn = fresh()
        r = quartermaster.maybe_answer(
            "How much water do we have left?", conn)
        self.assertIn("1000", r)
        self.assertIn("days at Sphere rates", r)

    def test_need_questions_fall_through_to_rag(self):
        conn = fresh()
        # (85-people need-questions now belong to the deterministic
        # planner — see PlanningMathTest)
        self.assertIsNone(quartermaster.maybe_answer(
            "How do I purify creek water?", conn))


if __name__ == "__main__":
    unittest.main()


class PlanningMathTest(unittest.TestCase):
    # demo beat #5: the LLM computed 3,825 / 1,275 / 255 across runs —
    # planning math is deterministic now
    def test_sphere_math_exact_every_time(self):
        conn = fresh()
        for _ in range(3):
            r = quartermaster.maybe_answer(
                "How much water do 85 people need for three days?", conn)
            self.assertIn("3,825 liters", r)
            self.assertIn("Sphere", r)

    def test_shortfall_fusion_with_ledger(self):
        conn = fresh()   # 1000 L on hand
        r = quartermaster.maybe_answer(
            "how much water do 85 people need for 3 days", conn)
        self.assertIn("2,825 liters short", r)

    def test_transactions_not_hijacked_by_planner(self):
        conn = fresh()
        r = quartermaster.maybe_answer(
            "we gave out 40 liters of water", conn)
        self.assertIn("Distributed 40", r)
