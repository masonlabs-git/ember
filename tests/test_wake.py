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

    def test_bracket_garbage_ignored_even_in_window(self):
        # seen live: '[ [ [ [ [ [] ].].' passed the two-word floor and
        # got answered with a fabricated citation
        for garbage in ("[ [ [ [ [ [] ].].", "10,000.", "... ... ..."):
            self.assertEqual(route(garbage, True), ("ignore", ""),
                             f"garbage not ignored: {garbage!r}")


if __name__ == "__main__":
    unittest.main()


class StoryRoutingTest(unittest.TestCase):
    def test_story_requests_detected(self):
        from box.brain import is_story
        for q in ("Awesome, can you read a bedtime story",
                  "tell me a story about a brave robot",
                  "can you tell the kids a story",
                  "make up a story for Ana"):
            self.assertTrue(is_story(q), q)

    def test_normal_questions_are_not_stories(self):
        from box.brain import is_story
        for q in ("How do I purify creek water?",
                  "what's the story with the water supply",
                  "we gave out 20 blankets"):
            self.assertFalse(is_story(q), q)


class DirectedFollowupTest(unittest.TestCase):
    # verbatim from the failed kitchen demo: the box answered fragments
    # of two people talking to each other inside the follow-up window
    def test_room_chatter_ignored_in_window(self):
        for chatter in (
            "Oh, you've, got the clue, saying that tips randomly talking. "
            "Yes. Oh, but it's a like, disaster recovery effort. - Edge "
            "device. - I'm calling it. So think about like a couple "
            "things that it does. One",
            "FimoWorker has a device, right?",
            "the embers.",
            "Well, I'm putting this right here just inside the box.",
        ):
            self.assertEqual(route(chatter, True), ("ignore", ""),
                             f"chatter answered: {chatter[:40]!r}")

    def test_real_followups_still_flow(self):
        for q in ("how do I purify creek water",
                  "what about for a burn",
                  "we just received 10 more gallons of water",
                  "can you read peter rabbit",
                  "done", "next"):
            action, _ = route(q, True)
            self.assertEqual(action, "answer", f"{q!r} was dropped")
