import tempfile
import unittest
from pathlib import Path

from box.retrieval import (build_authority, chunk_text, connect,
                           context_block, ingest_txt, search)

WATER = """Water Purification

Boil water vigorously for at least one minute to kill pathogens. At
altitudes above 6,500 feet, boil for three minutes.

If boiling is impossible, add eight drops of unscented household bleach
per gallon of water, stir, and wait thirty minutes before drinking.

Shelter Basics

A shelter site should be dry, level, and away from natural hazards such
as dead trees and drainage channels.
"""


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "index.db"
        self.conn = connect(self.db)
        src = Path(self.tmp.name) / "water.txt"
        src.write_text(WATER)
        ingest_txt(self.conn, src, "Test Manual")
        build_authority(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_search_finds_relevant_chunk(self):
        hits = search(self.conn, "how do I purify creek water?", top_k=2)
        self.assertTrue(hits)
        self.assertIn("bleach", hits[0].text.lower())

    def test_stemming_matches_variants(self):
        hits = search(self.conn, "boiling", top_k=2)
        self.assertTrue(any("boil" in h.text.lower() for h in hits))

    def test_citation_present(self):
        hits = search(self.conn, "shelter site", top_k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0].source, "Test Manual")
        self.assertIn("Test Manual", hits[0].citation)

    def test_operator_injection_is_neutralized(self):
        # FTS5 syntax in user input must not raise
        hits = search(self.conn, 'water AND (NEAR "bleach*" OR NOT)')
        self.assertIsInstance(hits, list)

    def test_empty_query(self):
        self.assertEqual(search(self.conn, "!!!"), [])

    def test_context_block_budget(self):
        hits = search(self.conn, "water bleach boil shelter", top_k=3)
        block = context_block(hits, budget_chars=200)
        self.assertLessEqual(len(block), 320)  # budget + headers
        self.assertIn("[1]", block)


class ChunkerTest(unittest.TestCase):
    def test_small_text_single_chunk(self):
        self.assertEqual(list(chunk_text("hello world", size=100)),
                         ["hello world"])

    def test_long_text_splits_on_paragraphs(self):
        text = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(10))
        chunks = list(chunk_text(text, size=800, overlap=100))
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))

    def test_pathological_paragraph_hard_splits(self):
        chunks = list(chunk_text("y" * 5000, size=800, overlap=100))
        self.assertGreater(len(chunks), 4)
        self.assertTrue(all(len(c) <= 800 for c in chunks))


if __name__ == "__main__":
    unittest.main()


class TierAndPinTest(unittest.TestCase):
    def setUp(self):
        import box.retrieval as R
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "i.db")
        rows = [("Wikipedia Medicine", "Triage trivia",
                 "triage history and water trivia " * 30)] * 5
        rows += [("START Triage Protocol", "",
                  "Assess respiration perfusion mental status for triage.")]
        rows += [("FM 21-76 Survival", "", "Boil water to purify it fully.")]
        self.conn.executemany(
            "INSERT INTO chunks(source,title,text) VALUES (?,?,?)", rows)
        self.conn.commit()
        build_authority(self.conn)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_authority_beats_bulk(self):
        # "water" alone matches the Wikipedia trivia rows too, but those live
        # only in the bulk table which is never touched when authority hits.
        hits = search(self.conn, "purify water", top_k=2)
        self.assertEqual(hits[0].source, "FM 21-76 Survival")
        self.assertTrue(all(h.source != "Wikipedia Medicine" for h in hits))

    def test_protocol_pin_guarantees_start(self):
        hits = search(self.conn, "help me triage the injured", top_k=3)
        self.assertIn("START Triage Protocol", [h.source for h in hits])
