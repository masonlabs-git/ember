"""Offline places: distance math, bearings, and voice-intent parsing."""
import unittest

from box.nav import bearing_word, haversine_miles, parse_kind


class DistanceTest(unittest.TestCase):
    def test_lehi_to_slc_about_25_miles(self):
        # Lehi (40.3916,-111.8508) -> SLC Temple Square (40.7704,-111.8919)
        d = haversine_miles(40.3916, -111.8508, 40.7704, -111.8919)
        self.assertAlmostEqual(d, 26.2, delta=1.0)

    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_miles(40, -111, 40, -111), 0.0)


class BearingTest(unittest.TestCase):
    def test_cardinals(self):
        self.assertEqual(bearing_word(40, -111, 41, -111), "north")
        self.assertEqual(bearing_word(40, -111, 39, -111), "south")
        self.assertEqual(bearing_word(40, -111, 40, -110), "east")
        self.assertEqual(bearing_word(40, -111, 40, -112), "west")


class IntentTest(unittest.TestCase):
    def test_places_questions_parse(self):
        self.assertEqual(parse_kind("How far is the nearest hospital?"),
                         "hospital")
        self.assertEqual(parse_kind("where is the closest gas station"),
                         "gas station")
        self.assertEqual(parse_kind("Where's the nearest ER?"), "hospital")
        self.assertEqual(parse_kind("distance to a pharmacy?"), "pharmacy")

    def test_non_places_questions_pass_through(self):
        # these must fall through to RAG, not the places engine
        self.assertIsNone(parse_kind("How do I purify creek water?"))
        self.assertIsNone(parse_kind("My friend is bleeding, help me."))
        self.assertIsNone(parse_kind("How much water do 85 people need?"))

    def test_water_the_supply_vs_water_the_place(self):
        # "nearest drinking water" is a place; plain "water" questions
        # are supplies/RAG territory
        self.assertEqual(parse_kind("where is the nearest drinking water"),
                         "water")
        self.assertIsNone(parse_kind("how much water do we need per day"))


if __name__ == "__main__":
    unittest.main()
