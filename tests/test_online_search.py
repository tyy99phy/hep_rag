from __future__ import annotations

import unittest
from unittest import mock

from hep_rag_v2.config import default_config
from hep_rag_v2.pipeline import _search_online_hits


def _hit(
    control_number: int,
    title: str,
    *,
    doc_types: list[str] | None = None,
    year: int | None = None,
    collaborations: list[str] | None = None,
    arxiv_id: str | None = None,
    doi: str | None = None,
) -> dict:
    metadata = {
        "control_number": control_number,
        "titles": [{"title": title}],
        "document_type": list(doc_types or []),
    }
    if year is not None:
        metadata["earliest_date"] = f"{year}-01-01"
    if collaborations:
        metadata["collaborations"] = [{"value": value} for value in collaborations]
    if arxiv_id:
        metadata["arxiv_eprints"] = [{"value": arxiv_id}]
    if doi:
        metadata["dois"] = [{"value": doi}]
    return {"metadata": metadata}


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, **_: object) -> dict[str, str]:
        return {
            "model": "gpt-5.4",
            "content": self.content,
        }


class OnlineSearchTests(unittest.TestCase):
    def test_search_online_hits_uses_query_rewrite_and_rank_fusion(self) -> None:
        config = default_config()
        config["llm"]["enabled"] = True
        config["llm"]["backend"] = "openai_compatible"
        config["llm"]["api_base"] = "http://127.0.0.1:8317/v1"
        config["llm"]["api_key"] = "sk-test"
        config["llm"]["model"] = "gpt-5.4"
        config["query_rewrite"]["max_queries"] = 4

        seed_query = (
            'collaboration:CMS and '
            '(title:"same-sign WW" or abstract:"same-sign WW" or '
            'title:"same-sign W boson" or abstract:"same-sign W boson" or '
            'title:"same-sign W boson pairs" or abstract:"same-sign W boson pairs") and '
            '(title:"vector boson scattering" or abstract:"vector boson scattering" or '
            'title:"electroweak production" or abstract:"electroweak production" or '
            'title:"in association with two jets" or abstract:"in association with two jets" or '
            'title:"two jets" or abstract:"two jets" or title:"dijet" or abstract:"dijet")'
        )

        search_results = {
            "CMS VBS SSWW": [
                _hit(2758816, "Measurement of same sign WW VBS processes at CMS with one hadronic tau in the final state", doc_types=["article"]),
                _hit(1847777, "Vector-Boson scattering at the LHC: Unraveling the electroweak sector", doc_types=["review"]),
            ],
            seed_query: [
                _hit(1789135, "Measurements of production cross sections of WZ and same-sign WW boson pairs in association with two jets in proton-proton collisions at 13 TeV", doc_types=["note"], year=2020, collaborations=["CMS"]),
                _hit(1794169, "Measurements of production cross sections of WZ and same-sign WW boson pairs in association with two jets in proton-proton collisions at 13 TeV", doc_types=["article"], year=2020, collaborations=["CMS"], arxiv_id="2005.01173", doi="10.1016/j.physletb.2020.135710"),
            ],
            "CMS same-sign WW WZ jets 13 TeV": [
                _hit(1794169, "Measurements of production cross sections of WZ and same-sign WW boson pairs in association with two jets in proton-proton collisions at 13 TeV", doc_types=["article"], year=2020, collaborations=["CMS"], arxiv_id="2005.01173", doi="10.1016/j.physletb.2020.135710"),
                _hit(2758816, "Measurement of same sign WW VBS processes at CMS with one hadronic tau in the final state", doc_types=["article"]),
            ],
            'collaboration CMS and (title:"same-sign" or abstract:"same sign") and (title:WZ or abstract:WZ) and (title:jets or abstract:jets)': [
                _hit(1794169, "Measurements of production cross sections of WZ and same-sign WW boson pairs in association with two jets in proton-proton collisions at 13 TeV", doc_types=["article"], year=2020, collaborations=["CMS"], arxiv_id="2005.01173", doi="10.1016/j.physletb.2020.135710"),
                _hit(1624170, "Observation of electroweak production of same-sign W boson pairs in the two jet and two same-sign lepton final state in proton-proton collisions at 13 TeV", doc_types=["article"]),
            ],
        }

        with (
            mock.patch("hep_rag_v2.pipeline._build_llm_client", return_value=_FakeClient(
                '["CMS same-sign WW WZ jets 13 TeV", "collaboration CMS and (title:\\"same-sign\\" or abstract:\\"same sign\\") and (title:WZ or abstract:WZ) and (title:jets or abstract:jets)"]'
            )),
            mock.patch("hep_rag_v2.pipeline.search_literature", side_effect=lambda query, **_: search_results[query]),
        ):
            hits, search_plan = _search_online_hits(config, query="CMS VBS SSWW", limit=3)

        self.assertTrue(search_plan["rewrite_used"])
        self.assertTrue(search_plan["seed_used"])
        self.assertEqual(
            search_plan["queries"],
            [
                "CMS VBS SSWW",
                seed_query,
                "CMS same-sign WW WZ jets 13 TeV",
                'collaboration CMS and (title:"same-sign" or abstract:"same sign") and (title:WZ or abstract:WZ) and (title:jets or abstract:jets)',
            ],
        )
        hit_ids = [hit["metadata"]["control_number"] for hit in hits]
        self.assertEqual(hit_ids[0], 1794169)
        self.assertIn(2758816, hit_ids)
        self.assertNotIn(1789135, hit_ids)

    def test_search_online_hits_uses_hep_seed_query_without_llm(self) -> None:
        config = default_config()
        config["llm"]["enabled"] = False

        seed_query = (
            'collaboration:CMS and '
            '(title:"same-sign WW" or abstract:"same-sign WW" or '
            'title:"same-sign W boson" or abstract:"same-sign W boson" or '
            'title:"same-sign W boson pairs" or abstract:"same-sign W boson pairs") and '
            '(title:"vector boson scattering" or abstract:"vector boson scattering" or '
            'title:"electroweak production" or abstract:"electroweak production" or '
            'title:"in association with two jets" or abstract:"in association with two jets" or '
            'title:"two jets" or abstract:"two jets" or title:"dijet" or abstract:"dijet")'
        )

        with mock.patch(
            "hep_rag_v2.pipeline.search_literature",
            side_effect=lambda query, **_: {
                "CMS VBS SSWW": [_hit(1847777, "Vector-Boson scattering at the LHC: Unraveling the electroweak sector", doc_types=["review"])],
                seed_query: [
                    _hit(1789135, "Measurements of production cross sections of same-sign WW and WZ boson pairs in association with two jets in proton-proton collisions at 13 TeV", doc_types=["note"], year=2020, collaborations=["CMS"]),
                    _hit(1794169, "Measurements of production cross sections of WZ and same-sign WW boson pairs in association with two jets in proton-proton collisions at 13 TeV", doc_types=["article"], year=2020, collaborations=["CMS"], arxiv_id="2005.01173", doi="10.1016/j.physletb.2020.135710"),
                ],
            }[query],
        ):
            hits, search_plan = _search_online_hits(config, query="CMS VBS SSWW", limit=3)

        self.assertFalse(search_plan["rewrite_used"])
        self.assertTrue(search_plan["seed_used"])
        self.assertEqual(search_plan["queries"], ["CMS VBS SSWW", seed_query])
        self.assertEqual(hits[0]["metadata"]["control_number"], 1794169)
        self.assertEqual(search_plan["dedupe_removed"], 1)

    def test_search_online_hits_dedupes_note_article_variants_before_limit(self) -> None:
        config = default_config()
        config["llm"]["enabled"] = False

        with mock.patch(
            "hep_rag_v2.pipeline.search_literature",
            return_value=[
                _hit(
                    1808672,
                    "Measurements of production cross sections of polarized same-sign W boson pairs in association with two jets in proton-proton collisions at 13 TeV",
                    doc_types=["note"],
                    year=2020,
                    collaborations=["CMS"],
                ),
                _hit(
                    1818160,
                    "Measurements of production cross sections of polarized same-sign W boson pairs in association with two jets in proton-proton collisions at 13 TeV",
                    doc_types=["article"],
                    year=2020,
                    collaborations=["CMS"],
                    arxiv_id="2009.09429",
                    doi="10.1016/j.physletb.2020.136018",
                ),
                _hit(1794169, "CMS article two", doc_types=["article"], year=2020, collaborations=["CMS"]),
                _hit(1847777, "Vector-Boson scattering at the LHC: Unraveling the electroweak sector", doc_types=["review"], year=2021),
            ],
        ):
            hits, search_plan = _search_online_hits(config, query="CMS same-sign WW", limit=3)

        hit_ids = [hit["metadata"]["control_number"] for hit in hits]
        self.assertEqual(hit_ids, [1818160, 1794169, 1847777])
        self.assertEqual(search_plan["dedupe_removed"], 1)

    def test_search_online_hits_skips_query_rewrite_for_structured_queries(self) -> None:
        config = default_config()
        config["llm"]["enabled"] = True
        structured_query = 'collaboration:"CMS" and collection:Published'

        with (
            mock.patch("hep_rag_v2.pipeline._build_llm_client") as build_client,
            mock.patch("hep_rag_v2.pipeline.search_literature", return_value=[_hit(1794169, "CMS paper")]) as search,
        ):
            hits, search_plan = _search_online_hits(config, query=structured_query, limit=5)

        build_client.assert_not_called()
        search.assert_called_once()
        self.assertFalse(search_plan["rewrite_used"])
        self.assertFalse(search_plan["seed_used"])
        self.assertEqual(search_plan["queries"], [structured_query])
        self.assertEqual(hits[0]["metadata"]["control_number"], 1794169)


if __name__ == "__main__":
    unittest.main()
