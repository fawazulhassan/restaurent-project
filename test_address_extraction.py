"""Offline tests of verbatim extraction and the enabled address flow."""

from unittest.mock import patch

import pytest

from app import llm
from app.dialog import chat_turn, build_system_prompt
from app.dialog_state import DialogStage, DialogState
from app.order import Order, add_item_to_order
from app.replies import ASK_ADDRESS_URDU, HUMAN_HANDOFF_URDU, build_confirm_address_urdu


def span_result(start, end, *, label="address", confidence=0.99):
    return llm.AssistantResult(tool_calls=[llm.ToolCall(
        "1", "extract_address_span", {
            "label": label, "start": start, "end": end, "confidence": confidence,
        },
    )])


@pytest.mark.parametrize("text,start,end,expected", [
    ("nhi mera address ye hai Purana Lari Adda Kasur", 5, 9, "Purana Lari Adda Kasur"),
    ("نہیں میرا اٹریس ہے، پڑھانا لاری، اٹڑا قصور۔", 4, 8, "پڑھانا لاری اٹڑا قصور"),
    ("میرا ڈریوڈری کا اڈریکس ہے پنانا، لڑی، اڈا، قصور۔", 5, 9, "پنانا لڑی اڈا قصور"),
    ("نئی میرا ڈریسہ کوک، شامی شہید، نیر، موڈل پیزار، قصور۔", 3, 10, "کوک شامی شہید نیر موڈل پیزار قصور"),
    ("نہیں پرانا لڑی اڈا کصور ہے بس۔", 1, 5, "پرانا لڑی اڈا کصور"),
    ("جی ٹی روڈ House no ۵", 0, 6, "جی ٹی روڈ House no ۵"),
    ("address is Nahi Chowk near Nayi Abadi", 2, 7, "Nahi Chowk near Nayi Abadi"),
])
def test_extractor_copies_original_span_without_fixing_place_names(text, start, end, expected):
    with patch("app.llm._call_openai", return_value=span_result(start, end)) as call:
        assert llm.extract_address_with_llm(text) == expected
    assert call.call_args.kwargs["timeout_seconds"] <= 4
    assert call.call_args.kwargs["tool_choice"]["name"] == "extract_address_span"


@pytest.mark.parametrize("start,end", [(-1, 2), (0, 99), (3, 2), (0, 0), (True, 2), (0, "2")])
def test_invalid_spans_cannot_invent_or_rewrite_an_address(start, end):
    with patch("app.llm._call_openai", return_value=span_result(start, end)):
        with pytest.raises(llm.LLMMalformedResponseError):
            llm.extract_address_with_llm("House 5 Kasur")


@pytest.mark.parametrize("result", [
    span_result(0, 0, label="unclear"),
    span_result(0, 3, confidence=0.5),
])
def test_uncertain_extraction_returns_no_address(result):
    with patch("app.llm._call_openai", return_value=result):
        assert llm.extract_address_with_llm("House 5 Kasur") is None


@pytest.mark.parametrize("confidence", [0.6, 0.78, 0.84])
def test_reported_confidence_scores_produce_readback_candidates(confidence):
    diagnostics = {}
    text = "کڑ، شامی، شہید، نیر، موڈل، بازار، قصور۔"
    with patch("app.llm._call_openai", return_value=span_result(0, 7, confidence=confidence)):
        candidate = llm.extract_address_with_llm(text, diagnostics=diagnostics)
    assert candidate == "کڑ شامی شہید نیر موڈل بازار قصور"
    assert diagnostics == {"status": "address", "confidence": confidence}


def session():
    order = Order()
    add_item_to_order(order, "plain-naan", None, 1)
    state = DialogState()
    state.sync_stage(order)
    return order, [{"role": "system", "content": build_system_prompt()}], state


def test_initial_address_and_correction_use_llm_but_confirmations_stay_local():
    order, messages, state = session()
    initial = "میرا ڈریوڈری کا اڈریکس ہے پنانا، لڑی، اڈا، قصور۔"
    correction = "نہیں میرا اٹریس ہے، پڑھانا لاری، اٹڑا قصور۔"
    with (
        patch("app.llm._call_openai", side_effect=[span_result(5, 9), span_result(4, 8)]) as api,
        patch("app.dialog.resolve_ambiguous_intent") as general,
    ):
        for text, expected in (
            (initial, "پنانا لڑی اڈا قصور"),
            (correction, "پڑھانا لاری اٹڑا قصور"),
        ):
            reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
            assert not complete
            assert order.delivery_address is None
            assert state.pending_address == expected
            assert reply == build_confirm_address_urdu(expected)
            assert state.last_path == "openai_address_extraction"
        assert api.call_count == 2
        _, order, messages, complete = chat_turn("ہنجی یہی ہے۔", order, messages, dialog_state=state)
        assert api.call_count == 2
        assert state.last_path == "deterministic"
        assert order.delivery_address == "پڑھانا لاری اٹڑا قصور"
        assert not complete  # final order still needs its own confirmation
        assert state.stage == DialogStage.AWAITING_CONFIRMATION
        general.assert_not_called()


@pytest.mark.parametrize("text", ["نہیں", "نی.", "نہیں، میرا ڈریس یہ نہیں ہے۔"])
def test_rejection_is_local_and_clears_only_pending_address(text):
    order, messages, state = session()
    state.await_address_confirmation("House 5")
    with patch("app.llm._call_openai") as api:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    api.assert_not_called()
    assert reply == ASK_ADDRESS_URDU
    assert state.pending_address is None
    assert order.delivery_address is None
    assert not complete


@pytest.mark.parametrize("text", [
    "ہاں ڈی", "ہاں، ڈی۔", "ہان جی", "جی ہان", "g hn", "hn g",
    "ہنڈی۔", "ہنڈی، ہنڈی", "ڈی، ہاں۔", "hnjee", "geehaan", "جی ہن",
])
def test_short_yes_variants_save_existing_pending_address_without_llm(text):
    order, messages, state = session()
    state.await_address_confirmation("کوٹ شامی شاہی")
    with patch("app.dialog.extract_address_with_llm") as extract:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    extract.assert_not_called()
    assert state.last_path == "deterministic"
    assert order.delivery_address == "کوٹ شامی شاہی"
    assert state.pending_address is None
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert state.summary_presented
    assert not complete


def test_yes_with_different_location_still_requires_new_address_confirmation():
    order, messages, state = session()
    state.await_address_confirmation("کوٹ شامی شاہی")
    text = "میں ہان جی، میرا پتہ یہ ہی ہے کوچ آمی شہید۔"
    with patch("app.dialog.extract_address_with_llm", return_value="کوچ آمی شہید") as extract:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    extract.assert_called_once()
    assert state.pending_address == "کوچ آمی شہید"
    assert reply == build_confirm_address_urdu("کوچ آمی شہید")
    assert order.delivery_address is None
    assert not complete


@pytest.mark.parametrize("text", ["nhi g", "g nhi", "nhinjee", "نھیں جی", "نئیں جی", "ہنڈی نہیں"])
def test_extended_no_vocabulary_rejects_address_without_llm(text):
    order, messages, state = session()
    state.await_address_confirmation("کوٹ شامی شہید")
    with patch("app.dialog.extract_address_with_llm") as extract:
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    extract.assert_not_called()
    assert reply == ASK_ADDRESS_URDU
    assert order.delivery_address is None
    assert state.pending_address is None
    assert state.stage == DialogStage.AWAITING_ADDRESS
    assert not complete


def test_quantity_alone_never_confirms_pending_address():
    order, messages, state = session()
    state.await_address_confirmation("کوٹ شامی شہید")
    with patch("app.dialog.extract_address_with_llm", return_value=None):
        chat_turn("ڈیڑھ", order, messages, dialog_state=state)
    assert order.delivery_address is None
    assert state.pending_address == "کوٹ شامی شہید"


@pytest.mark.parametrize("failure", [None, llm.LLMServiceError("timeout"), llm.LLMMalformedResponseError("bad span")])
def test_failed_extraction_preserves_previous_address_and_cannot_confirm(failure):
    order, messages, state = session()
    order.delivery_address = "Existing confirmed location"
    state.await_address_confirmation("Previous pending location")
    kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
    with patch("app.dialog.extract_address_with_llm", **kwargs) as extract:
        for _ in range(3):
            reply, order, messages, complete = chat_turn("uncertain correction", order, messages, dialog_state=state)
            assert order.delivery_address == "Existing confirmed location"
            assert not complete
        assert extract.call_count == 3
    assert reply == HUMAN_HANDOFF_URDU
    assert state.stage == DialogStage.HANDOFF


def test_corrected_confirmed_address_is_not_overwritten_before_local_yes():
    order, messages, state = session()
    order.delivery_address = "House 1"
    state.await_address_confirmation("House 2")
    with patch("app.dialog.extract_address_with_llm", return_value="House 3") as extract:
        _, order, messages, _ = chat_turn("nhi mera address ye hai House 3", order, messages, dialog_state=state)
        assert order.delivery_address == "House 1"
        assert state.pending_address == "House 3"
        chat_turn("yes", order, messages, dialog_state=state)
        extract.assert_called_once()
    assert order.delivery_address == "House 3"


def test_no_address_does_not_turn_yes_into_an_address():
    order, messages, state = session()
    with patch("app.llm._call_openai") as api:
        _, order, messages, complete = chat_turn("yes", order, messages, dialog_state=state)
    api.assert_not_called()
    assert order.delivery_address is None
    assert state.pending_address is None
    assert not complete


def test_address_extraction_preloads_openai_even_if_general_fallback_is_disabled(monkeypatch):
    import config
    monkeypatch.setattr(config, "HYBRID_INTENT_ENABLED", True)
    monkeypatch.setattr(config, "HYBRID_LLM_FALLBACK_ENABLED", False)
    with patch("app.llm.check_llm_service") as check:
        llm.preload_llm()
    check.assert_called_once()


def test_address_with_menu_words_cannot_add_food_while_waiting_for_address():
    order, messages, state = session()
    before = [item.model_dump() for item in order.items]
    text = "میرا ڈلیوری کا ڈریس ہے، کوک، شامی شہید، نیر، موڈر، بزار، قصور کڑاہی۔"
    with (
        patch("app.llm._call_openai", return_value=span_result(5, 13, confidence=0.78)) as api,
        patch("app.dialog.parse_customer_intent") as parser,
    ):
        reply, order, messages, complete = chat_turn(text, order, messages, dialog_state=state)
    api.assert_called_once()
    parser.assert_called_once()
    assert [item.model_dump() for item in order.items] == before
    assert order.delivery_address is None
    assert state.pending_address == "کوک شامی شہید نیر موڈر بزار قصور کڑاہی"
    assert reply == build_confirm_address_urdu(state.pending_address)
    assert not complete


def test_failed_first_extraction_can_recover_on_next_reply():
    order, messages, state = session()
    first = "کڑ، شامی، شہید، نیت، موڈل، بازار، قصور۔"
    second = "کوٹ، شیمی، شہید، نیر، موڈل، بزار، قصور۔"
    with patch("app.llm._call_openai", side_effect=[
        span_result(0, 0, label="unclear", confidence=0.3),
        span_result(0, 7, confidence=0.84),
    ]) as api:
        chat_turn(first, order, messages, dialog_state=state)
        assert state.address_confirmation_attempts == 1
        assert state.stage == DialogStage.AWAITING_ADDRESS
        reply, order, messages, complete = chat_turn(second, order, messages, dialog_state=state)
        assert state.address_confirmation_attempts == 0
        assert state.pending_address == "کوٹ شیمی شہید نیر موڈل بزار قصور"
        assert order.delivery_address is None
        assert reply == build_confirm_address_urdu(state.pending_address)
        assert not complete
        chat_turn("ہنجی یہی ہے", order, messages, dialog_state=state)
        assert api.call_count == 2
    assert order.delivery_address == "کوٹ شیمی شہید نیر موڈل بزار قصور"
    assert state.stage == DialogStage.AWAITING_CONFIRMATION
    assert not state.last_address_extraction_detail


def test_service_error_is_distinguished_from_uncertain_address():
    order, messages, state = session()
    with patch("app.dialog.extract_address_with_llm", side_effect=llm.LLMTimeoutError("request timed out")):
        reply, order, messages, complete = chat_turn("House 5 Kasur", order, messages, dialog_state=state)
    assert "LLMTimeoutError" in state.last_address_extraction_detail
    assert "request timed out" in state.last_address_extraction_detail
    assert "سروس" in reply
    assert state.pending_address is None
    assert order.delivery_address is None
    assert not complete


def test_explicit_food_add_request_still_works_while_waiting_for_address():
    order, messages, state = session()
    with patch("app.llm._call_openai") as api:
        _, order, messages, complete = chat_turn("ek garlic naan add kar do", order, messages, dialog_state=state)
    api.assert_not_called()
    assert any(item.id == "garlic-naan" for item in order.items)
    assert order.delivery_address is None
    assert not complete


def test_address_containing_alef_madda_is_saved_unchanged():
    order = Order()
    messages = [{"role": "system", "content": build_system_prompt()}]
    state = DialogState()
    state.await_address_confirmation("آرام باغ قصور")
    with patch("app.llm._call_openai") as api:
        chat_turn("ہاں جی", order, messages, dialog_state=state)
    api.assert_not_called()
    assert order.delivery_address == "آرام باغ قصور"
    assert "آ" in order.delivery_address
