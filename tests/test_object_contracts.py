from __future__ import annotations

import unittest

from hep_rag_v2.object_contracts import (
    CURRENT_OBJECT_CONTRACT_VERSION,
    EvidenceBundle,
    MethodSignature,
    ResultSignature,
    TraceStep,
    WorkCapsule,
    export_object_contract_schemas,
)


class ObjectContractTests(unittest.TestCase):
    def test_contracts_round_trip_with_v1_wire_format(self) -> None:
        capsule = WorkCapsule(
            object_id="work:42",
            work_id=42,
            title="Rare decay measurement",
            abstract="We measure a branching fraction.",
            status="ready",
            source_kind="retrieval",
            source_refs=("works:42",),
            derivation="normalized",
            collection_name="default",
            chunk_refs=("chunk:101",),
            has_fulltext_support=True,
        )
        result = ResultSignature(
            object_id="result:42:default",
            work_id=42,
            label="branching fraction",
            result_kind="measurement",
            summary_text="Rare decay measurement | result signatures: branching fraction | source=chunks",
            status="ready",
            source_kind="extraction",
            source_refs=("works:42", "result_objects:77"),
            derivation="normalized",
            confidence=0.91,
            work_capsule_id="work:42",
            value_items=({"numeric_value": 1.2, "unit": "arb"},),
            evidence_items=({"citation_id": "E1", "text": "We measure the branching fraction."},),
        )
        method = MethodSignature(
            object_id="method:42:default",
            work_id=42,
            label="profile likelihood",
            method_kind="statistical_fit",
            summary_text="Rare decay measurement | method signatures: profile likelihood | source=chunks",
            status="ready",
            source_kind="extraction",
            source_refs=("works:42", "method_objects:12"),
            derivation="normalized",
            normalized_text="profile likelihood",
            confidence=0.88,
            work_capsule_id="work:42",
        )
        evidence = EvidenceBundle(
            object_id="evidence:run-1",
            status="ready",
            source_kind="service",
            source_refs=("service:run-1",),
            derivation="aggregated",
            subject_refs=(result.object_id, method.object_id),
            items=(
                {
                    "citation_id": "E1",
                    "evidence_key": result.object_id,
                    "object_type": "result_signature",
                    "object_id": result.object_id,
                    "title": "Rare decay measurement",
                    "occurrences": 1,
                },
            ),
            work_capsule_refs=(capsule.object_id,),
        )
        trace = TraceStep(
            object_id="trace:run-1:step-1",
            step_type="retrieve",
            summary="retrieved 1 evidence item",
            status="ready",
            source_kind="service",
            source_refs=("service:run-1",),
            derivation="summarized",
            step_index=1,
            output_refs=(evidence.object_id,),
        )

        payloads = [item.to_payload() for item in (capsule, result, method, evidence, trace)]
        restored = [
            WorkCapsule.from_payload(payloads[0]),
            ResultSignature.from_payload(payloads[1]),
            MethodSignature.from_payload(payloads[2]),
            EvidenceBundle.from_payload(payloads[3]),
            TraceStep.from_payload(payloads[4]),
        ]

        self.assertEqual(restored, [capsule, result, method, evidence, trace])
        self.assertEqual(payloads[0]["contract_version"], CURRENT_OBJECT_CONTRACT_VERSION)
        self.assertEqual(payloads[0]["object_type"], "work_capsule")
        self.assertEqual(payloads[1]["object_type"], "result_signature")
        self.assertEqual(payloads[2]["object_type"], "method_signature")
        self.assertEqual(payloads[3]["object_type"], "evidence_bundle")
        self.assertEqual(payloads[4]["object_type"], "trace_step")

    def test_validation_rejects_invalid_envelope_fields(self) -> None:
        with self.assertRaises(ValueError):
            WorkCapsule(
                object_id="work:42",
                work_id=42,
                title="Title",
                status="draft",
                source_kind="retrieval",
                source_refs=("works:42",),
                derivation="normalized",
            )
        with self.assertRaises(ValueError):
            ResultSignature.from_payload(
                {
                    "contract_version": "v2",
                    "object_type": "result_signature",
                    "object_id": "result:42:default",
                    "work_id": 42,
                    "label": "branching fraction",
                    "result_kind": "measurement",
                    "summary_text": "summary",
                    "status": "ready",
                    "source_kind": "extraction",
                    "source_refs": ["works:42"],
                    "derivation": "normalized",
                }
            )
        with self.assertRaises(ValueError):
            EvidenceBundle(
                object_id="evidence:run-1",
                status="ready",
                source_kind="service",
                source_refs=("service:run-1",),
                derivation="aggregated",
                subject_refs=("result:42:default",),
                items=({"title": "missing stable refs"},),
            )
        with self.assertRaises(ValueError):
            TraceStep(
                object_id="trace:run-1:step-1",
                step_type="",
                summary="summary",
                status="ready",
                source_kind="service",
                source_refs=("service:run-1",),
                derivation="summarized",
            )

    def test_schema_export_lists_required_v1_wire_fields(self) -> None:
        schemas = export_object_contract_schemas()

        self.assertEqual(schemas["version"], CURRENT_OBJECT_CONTRACT_VERSION)
        self.assertEqual(set(schemas["objects"]), {
            "WorkCapsule",
            "ResultSignature",
            "MethodSignature",
            "EvidenceBundle",
            "TraceStep",
        })
        for schema in schemas["objects"].values():
            self.assertIn("contract_version", schema["required"])
            self.assertIn("object_type", schema["required"])
            self.assertIn("object_id", schema["required"])
            self.assertIn("source_kind", schema["required"])
            self.assertIn("status", schema["required"])
            self.assertIn("source_refs", schema["required"])
            self.assertIn("derivation", schema["required"])
        self.assertEqual(schemas["objects"]["WorkCapsule"]["properties"]["object_type"]["const"], "work_capsule")
        self.assertEqual(schemas["objects"]["TraceStep"]["properties"]["object_type"]["const"], "trace_step")


if __name__ == "__main__":
    unittest.main()
