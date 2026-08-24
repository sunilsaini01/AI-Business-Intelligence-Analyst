from report_view import build_report_sections


def _full_report(**overrides) -> dict:
    base = {
        "executive_summary": "Revenue fell 6.7% in July.",
        "key_findings": ["June: 161445.80", "July: 150633.02"],
        "evidence": [{"query": "SELECT ...", "row_count": 2}],
        "recommendations": ["Review Enterprise segment retention."],
        "confidence": "Medium",
        "limitations": "Root cause could not be fully confirmed.",
        "verified_claims": ["June: 161445.80"],
        "analysis_explanation": "Revenue moved from 161,445.80 to 150,633.02.",
        "visualizations": [{"chart_type": "bar", "title": "Revenue"}],
        "technical_details": {"critic_status": "PASS", "critic_score": 1.0},
        "narrative": None,
    }
    base.update(overrides)
    return base


def _titles(report: dict) -> list[str]:
    return [s.title for s in build_report_sections(report)]


def test_full_report_includes_all_expected_sections_in_order():
    titles = _titles(_full_report())
    assert titles == [
        "Executive Summary",
        "Key Findings",
        "Evidence",
        "Verified Claims",
        "Analysis",
        "Confidence",
        "Limitations",
        "Recommendations",
        "Technical Details",
    ]


def test_missing_narrative_omits_the_narrative_section_entirely():
    titles = _titles(_full_report(narrative=None))
    assert "Narrative" not in titles


def test_present_narrative_is_shown():
    titles = _titles(_full_report(narrative="Revenue declined in July, led by Enterprise."))
    assert "Narrative" in titles


def test_empty_narrative_string_is_also_omitted():
    titles = _titles(_full_report(narrative=""))
    assert "Narrative" not in titles


def test_confidence_is_always_shown_verbatim_never_upgraded():
    sections = build_report_sections(_full_report(confidence="Low"))
    confidence_section = next(s for s in sections if s.title == "Confidence")
    assert confidence_section.content == "Low"  # never silently upgraded to Medium/High


def test_limitations_are_always_preserved_when_present():
    sections = build_report_sections(_full_report(confidence="Low", limitations="Verification did not fully pass."))
    limitations_section = next(s for s in sections if s.title == "Limitations")
    assert limitations_section.kind == "warning"
    assert limitations_section.content == "Verification did not fully pass."


def test_empty_limitations_omits_the_section_not_an_empty_warning():
    titles = _titles(_full_report(limitations=""))
    assert "Limitations" not in titles


def test_verified_claims_shown_when_present_omitted_when_empty():
    assert "Verified Claims" in _titles(_full_report(verified_claims=["x"]))
    assert "Verified Claims" not in _titles(_full_report(verified_claims=[]))


def test_analysis_explanation_shown_when_present_omitted_when_empty():
    assert "Analysis" in _titles(_full_report(analysis_explanation="text"))
    assert "Analysis" not in _titles(_full_report(analysis_explanation=""))


def test_evidence_shown_when_present_omitted_when_empty():
    assert "Evidence" in _titles(_full_report(evidence=[{"query": "x", "row_count": 1}]))
    assert "Evidence" not in _titles(_full_report(evidence=[]))


def test_recommendations_shown_when_present_omitted_when_empty():
    assert "Recommendations" in _titles(_full_report(recommendations=["do X"]))
    assert "Recommendations" not in _titles(_full_report(recommendations=[]))


def test_technical_details_shown_when_present_omitted_when_empty():
    assert "Technical Details" in _titles(_full_report(technical_details={"critic_status": "PASS"}))
    assert "Technical Details" not in _titles(_full_report(technical_details={}))


def test_fail_exhausted_style_report_keeps_low_confidence_and_limitations_visible():
    """A degraded (Critic FAIL-exhausted) report — confidence must remain
    Low and limitations must remain visible, never hidden or reworded."""
    degraded = _full_report(
        confidence="Low",
        limitations="Automated review found unresolved issues: unsupported number",
        verified_claims=[],
    )
    sections = build_report_sections(degraded)
    titles = [s.title for s in sections]
    assert "Limitations" in titles
    assert next(s for s in sections if s.title == "Confidence").content == "Low"
    assert "Verified Claims" not in titles  # honestly empty, not fabricated


def test_never_invents_a_value_for_a_missing_field():
    minimal = {"executive_summary": "x", "confidence": "Low"}
    sections = build_report_sections(minimal)
    titles = [s.title for s in sections]
    assert titles == ["Executive Summary", "Confidence"]  # only what's actually present
    for section in sections:
        assert section.content not in (None,)  # never a fabricated placeholder value
