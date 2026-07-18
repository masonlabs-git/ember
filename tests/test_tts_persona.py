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

    def test_personas_ground_and_never_refuse(self):
        for s in (persona.ANSWER, persona.COACH, persona.INTERVIEW):
            self.assertIn("offline", s)
            self.assertIn("Never refuse", s)
            self.assertIn("SOURCES", s)


if __name__ == "__main__":
    unittest.main()


class ModeRoutingTest(unittest.TestCase):
    # pick_persona returns (mode, system). Answer and coach share ONE
    # system prompt (persona.MAIN) so ollama's KV prefix stays cached
    # across modes; only the mode label differs.
    def test_bleeding_routes_to_coach(self):
        from box.brain import pick_persona
        from box import persona
        mode, system = pick_persona("my arm is bleeding badly")
        self.assertEqual(mode, "coach")
        self.assertIs(system, persona.MAIN)

    def test_checkin_routes_to_interview(self):
        from box.brain import pick_persona
        from box import persona
        mode, system = pick_persona("we just arrived, can you check us in")
        self.assertEqual(mode, "interview")
        self.assertIs(system, persona.INTERVIEW)

    def test_general_question_stays_answer(self):
        from box.brain import pick_persona
        from box import persona
        mode, system = pick_persona("how much water do we need")
        self.assertEqual(mode, "answer")
        self.assertIs(system, persona.MAIN)

    def test_answer_and_coach_share_one_kv_prefix(self):
        from box import persona
        self.assertIs(persona.ANSWER, persona.COACH)


class SpeakableTest(unittest.TestCase):
    # a token cap can cut generation mid-citation; the flushed tail
    # ('[2.') strips to nothing and must never reach the synthesizer
    def test_normal_sentence_is_speakable(self):
        from box.tts import speakable
        self.assertTrue(speakable("Boil the water for one minute [1]."))

    def test_citation_only_tail_is_not(self):
        from box.tts import speakable
        self.assertFalse(speakable(" [2]."))       # strips to bare '.'
        self.assertFalse(speakable("."))
        self.assertFalse(speakable("  "))

    def test_partial_citation_still_speaks(self):
        # '[1' doesn't match the strip pattern; it has a digit, so piper
        # can voice it — awkward but crash-free
        from box.tts import speakable
        self.assertTrue(speakable("[1"))


class CitationStripTest(unittest.TestCase):
    def test_strips_single_and_multi(self):
        from box.tts import strip_citations
        self.assertEqual(strip_citations("Boil water [1]."), "Boil water.")
        self.assertEqual(strip_citations("Use bleach [1, 2]."), "Use bleach.")
        self.assertEqual(strip_citations("A [1] and B [2,3] end."),
                         "A and B end.")

    def test_apply_shim_also_strips(self):
        from box.tts import apply_shim
        self.assertNotIn("[", apply_shim("The bugout box says [1]."))
