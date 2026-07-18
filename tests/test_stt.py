import unittest

from box.stt import _parse


class SttParseTest(unittest.TestCase):
    OUT = """Architecture: hailo10h
Variant: Whisper base
Initializing Whisper pipeline...
✓ Ready (chunk length: 10s)
Loading: /tmp/stt_in.wav
✓ Loaded (2.8s)
Transcribing (1 chunk(s), gain 1.00→0.90)...
--------------------------------------------------
How do I purify creek water to make it safe.
--------------------------------------------------
(1.2s)
Done."""

    def test_parses_transcript_between_rules(self):
        self.assertEqual(_parse(self.OUT),
                         "How do I purify creek water to make it safe.")

    def test_empty_output(self):
        self.assertEqual(_parse(""), "")

    def test_multiline_transcript(self):
        out = "x\n----------\nline one\nline two\n----------\n(1s)"
        self.assertEqual(_parse(out), "line one line two")


if __name__ == "__main__":
    unittest.main()
