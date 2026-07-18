"""Wake-word routing — the demo-critical path. route() is pure so every
whisper-ism and stray-speech case is pinned down here."""
import unittest

from box.brain import route


class WakeRoutingTest(unittest.TestCase):
    def test_wake_plus_question_answers_immediately(self):
        action, q = route("Hey Ember, how do I purify creek water?", False)
        self.assertEqual(action, "answer")
        self.assertEqual(q, "how do I purify creek water")

    def test_whisper_hears_amber(self):
        action, q = route("Hey Amber, how much water do we need?", False)
        self.assertEqual(action, "answer")
        self.assertIn("water", q)

    def test_bare_wake_acknowledges(self):
        self.assertEqual(route("Hey Ember.", False), ("wake", ""))
        self.assertEqual(route("Ember?", False), ("wake", ""))

    def test_stray_speech_ignored_when_asleep(self):
        self.assertEqual(route("how much water do we need", False),
                         ("ignore", ""))

    def test_followup_window_answers_without_wake(self):
        action, q = route("how much water do we need", True)
        self.assertEqual(action, "answer")
        self.assertEqual(q, "how much water do we need")

    def test_coach_single_word_turns_flow(self):
        # the coach loop runs on one-word turns — they must get through
        for w in ("Done.", "next", "Repeat!", "okay"):
            action, q = route(w, True)
            self.assertEqual(action, "answer", f"{w!r} was dropped")

    def test_single_word_noise_still_ignored_in_window(self):
        self.assertEqual(route("you", True), ("ignore", ""))
        self.assertEqual(route("Bye.", True), ("ignore", ""))

    def test_remember_and_november_do_not_wake(self):
        self.assertEqual(route("Remember to bring the water", False),
                         ("ignore", ""))
        self.assertEqual(route("See you in November", False),
                         ("ignore", ""))

    def test_whisper_noise_hallucinations_ignored(self):
        for noise in ("Thanks for watching!", "you", "Thank you."):
            self.assertEqual(route(noise, False), ("ignore", ""),
                             f"hallucination not ignored: {noise!r}")


if __name__ == "__main__":
    unittest.main()
