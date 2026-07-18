import unittest

from box import persona
from box.tts import apply_shim, pick_voice, sentences
from box import config


class ShimTest(unittest.TestCase):
    def test_bugout_becomes_bug_out(self):
        self.assertEqual(apply_shim("The Bugout box is ready"),
                         "The bug-out box is ready")

    def test_case_insensitive(self):
        self.assertIn("bug-out", apply_shim("BUGOUT"))

    def test_no_partial_word_mangling(self):
        self.assertEqual(apply_shim("debugout stream"), "debugout stream")


class SentenceStreamTest(unittest.TestCase):
    def test_reassembles_fragments(self):
        frags = ["Boil wat", "er for one minute. Then co", "ol it. Drink."]
        self.assertEqual(list(sentences(frags)),
                         ["Boil water for one minute.", "Then cool it.",
                          "Drink."])

    def test_flushes_tail_without_period(self):
        self.assertEqual(list(sentences(["No period here"])),
                         ["No period here"])

    def test_empty_stream(self):
        self.assertEqual(list(sentences([])), [])


class VoicePickTest(unittest.TestCase):
    def test_spanish_reply_uses_spanish_voice(self):
        text = "Hierva el agua durante un minuto para que sea segura."
        self.assertEqual(pick_voice(text), config.VOICE_ES)

    def test_english_reply_uses_english_voice(self):
        self.assertEqual(pick_voice("Boil the water for one minute."),
                         config.VOICE_EN)


class PersonaTest(unittest.TestCase):
    def test_prompt_includes_sources_and_question(self):
        p = persona.build_prompt("how much water?", "[1] (Sphere) 15 litres")
        self.assertIn("SOURCES:", p)
        self.assertIn("QUESTION: how much water?", p)

    def test_prompt_without_context(self):
        p = persona.build_prompt("hello", "")
        self.assertNotIn("SOURCES:", p)

    def test_personas_demand_citations_and_safety(self):
        for s in (persona.ANSWER, persona.COACH, persona.INTERVIEW):
            self.assertIn("offline", s)
            self.assertIn("Safety first", s)


if __name__ == "__main__":
    unittest.main()


class ModeRoutingTest(unittest.TestCase):
    def test_bleeding_routes_to_coach(self):
        from box.brain import pick_persona
        from box import persona
        self.assertIs(pick_persona("my arm is bleeding badly"),
                      persona.COACH)

    def test_checkin_routes_to_interview(self):
        from box.brain import pick_persona
        from box import persona
        self.assertIs(pick_persona("we just arrived, can you check us in"),
                      persona.INTERVIEW)

    def test_general_question_stays_answer(self):
        from box.brain import pick_persona
        from box import persona
        self.assertIs(pick_persona("how much water do we need"),
                      persona.ANSWER)
