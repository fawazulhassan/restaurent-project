"""One tool-calling turn. Code commits state only after the spoken text is valid."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from app.agent_prompt import SYSTEM_PROMPT
from app.agent_tools import (
    AdditionBudget,
    AgentAbort,
    execute_tool,
    load_facts,
    tools_for_stage,
)
from app.dialog_state import DialogStage, DialogState
from app.llm import (
    LLMMalformedResponseError,
    LLMServiceError,
    LLMTimeoutError,
    call_agent_model,
)
from app.numbers_urdu import _UNDER_100, number_to_urdu_words
from app.order import Order, calculate_total
from app.text_fold import compare_tokens, fold_text

_PLACEHOLDER = re.compile(
    r"\{price:([a-z0-9-]+):(null|[a-z0-9-]+)\}"
    r"|\{total\}"
    r"|\{fact:([a-z0-9_]+)\}"
    r"|\{item:([a-z0-9-]+)\}"
)
_CONFIRM_WORDS = ("تصدیق", "کنفرم", "confirm", "saved", "محفوظ", "placed")
_PRICE_WORDS = ("روپے", "ہزار", "سو", "لاکھ", "کروڑ", "rupees", "price")
_SUCCESS_PHRASES = ("شامل کر دیا", "ہٹا دیا", "بدل دیا", "added", "removed")
_NUMBER_WORDS = set(_UNDER_100) | set(_PRICE_WORDS)


@dataclass
class AgentOutcome:
    ok: bool
    reply: str = ""
    order: Order | None = None
    state: DialogState | None = None
    reason: str = ""
    text_only: bool = False
    rounds: int = 0
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    validation: str = ""
    latency: float = 0.0
    applied: bool = False
    basket_detail: str = ""


def _compact_menu() -> list[dict[str, Any]]:
    from app.agent_tools import _menu_items

    rows = []
    for item in _menu_items():
        if not item.get("available", True):
            continue
        rows.append({
            "item_id": item["id"],
            "spoken_name_urdu": item.get("spoken_name_urdu") or "",
            "sizes": [size["id"] for size in item.get("sizes", [])],
        })
    return rows


def _payload(user_text: str, order: Order, state: DialogState, tools: list[dict]) -> list[dict]:
    from app.llm import _history_payload, _pending_size_id, _spoken_basket

    body = {
        "state": {
            "stage": state.stage.value,
            "basket": _spoken_basket(order),
            "pending_item_id": state.pending_item_id,
            "pending_size_id": _pending_size_id(state),
            "pending_choices": list(state.pending_choices),
        },
        "menu": _compact_menu(),
        "untrusted": {
            "history": _history_payload(state),
            "utterance": user_text,
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
    ]


def _phrase_present(tokens: list[str], phrase: str) -> bool:
    wanted = compare_tokens(phrase)
    if not wanted:
        return phrase.casefold() in tokens
    return any(tokens[index:index + len(wanted)] == wanted for index in range(len(tokens) - len(wanted) + 1))


def _find_item(item_id: str) -> dict | None:
    from app.agent_tools import _find_item as find
    return find(item_id)


def validate_agent_text(text: str, *, mutations_failed: bool) -> str:
    """Return 'ok' or an error code. Placeholders are removed before the letter check."""
    if not isinstance(text, str) or not text.strip() or len(text) > 300:
        return "bad_length"
    stripped = _PLACEHOLDER.sub(" ", text)
    if "{" in stripped or "}" in stripped:
        return "bad_placeholder"
    folded = fold_text(stripped)
    if any(character.isdigit() for character in folded):
        return "digit"
    tokens = compare_tokens(stripped)
    blocked = set()
    for word in (*_CONFIRM_WORDS, *_PRICE_WORDS):
        blocked.update(compare_tokens(word))
    blocked.update(compare_tokens(word)[0] for word in _NUMBER_WORDS if compare_tokens(word))
    if any(token in blocked for token in tokens):
        return "blocked_word"
    if mutations_failed and any(_phrase_present(tokens, phrase) for phrase in _SUCCESS_PHRASES):
        return "false_success"
    letters = [character for character in stripped if character.isalpha()]
    if not letters:
        return "not_urdu"
    urdu = sum(1 for character in letters if 0x0600 <= ord(character) <= 0x06FF or 0x0750 <= ord(character) <= 0x077F or 0x08A0 <= ord(character) <= 0x08FF)
    if urdu / len(letters) < 0.8:
        return "not_urdu"
    return "ok"


def fill_placeholders(text: str, order: Order) -> str:
    """Replace placeholders from the trial order, the menu, and facts. Unknowns fail."""

    def replace(match: re.Match) -> str:
        item_id, size_id, fact_key, named_item = match.groups()
        if match.group(0) == "{total}":
            return f"{number_to_urdu_words(calculate_total(order))} روپے"
        if fact_key is not None:
            from app.agent_prompt import FACT_KEYS
            from app.replies import FACT_UNKNOWN_URDU

            facts = load_facts()
            if fact_key not in FACT_KEYS:
                raise AgentAbort("unknown_fact")
            value = facts.get(fact_key, "")
            if not str(value).strip():
                return FACT_UNKNOWN_URDU
            return str(value)
        target = item_id or named_item
        item = _find_item(target or "")
        if item is None:
            raise AgentAbort("unknown_item")
        if named_item is not None:
            if not item.get("available", True):
                raise AgentAbort("item_unavailable")
            return item.get("spoken_name_urdu") or item.get("name_urdu") or ""
        size_key = None if size_id == "null" else size_id
        if size_key:
            price = next((size["price_pkr"] for size in item.get("sizes", []) if size["id"] == size_key), None)
            if price is None:
                raise AgentAbort("unknown_size")
        else:
            price = item.get("price_pkr")
            if price is None and item.get("sizes"):
                raise AgentAbort("size_required")
        return f"{number_to_urdu_words(int(price))} روپے"

    return _PLACEHOLDER.sub(replace, text)


def _model_already_asks_more(model_sentence: str) -> bool:
    from app.replies import ASK_MORE_ITEMS_URDU

    folded = fold_text(model_sentence)
    return "شامل کرنا چاہ" in folded or fold_text(ASK_MORE_ITEMS_URDU) in folded


def _append_more_items(reply: str, model_sentence: str) -> str:
    from app.replies import ASK_MORE_ITEMS_URDU

    if reply.endswith(ASK_MORE_ITEMS_URDU) or _model_already_asks_more(model_sentence):
        return reply
    return f"{reply} {ASK_MORE_ITEMS_URDU}"


def _close_for_mixed_no(user_text: str, order: Order) -> bool:
    from app.dialog import _mixed_no_close

    return _mixed_no_close(user_text, order)


def _with_total_and_close(
    reply: str,
    model_sentence: str,
    user_text: str,
    order: Order,
    state: DialogState,
) -> str:
    from app.dialog import _close_more_items, _has_total_word
    from app.replies import ASK_MORE_ITEMS_URDU

    spoken = reply
    if _has_total_word(user_text) and "{total}" not in model_sentence:
        spoken = f"{spoken} {number_to_urdu_words(calculate_total(order))} روپے".strip()
    close_reply, _complete = _close_more_items(order, state)
    if close_reply == ASK_MORE_ITEMS_URDU and spoken.endswith(ASK_MORE_ITEMS_URDU):
        return spoken
    return f"{spoken} {close_reply}".strip()


def _finish_items_speech(tool_log: list[dict[str, Any]]) -> str:
    for entry in tool_log:
        if entry["name"] == "finish_items" and entry["result"].get("ok"):
            return str(entry["result"].get("speak") or "")
    return ""


def _pending_question(state: DialogState) -> str:
    from app.dialog import _choice_question, _menu_item, build_missing_size_urdu
    from app.replies import ASK_ADDRESS_URDU, ASK_CONFIRM_URDU, build_confirm_address_urdu

    if state.pending_item_id and (item := _menu_item(state.pending_item_id)):
        return build_missing_size_urdu(item)
    if state.pending_choices:
        return _choice_question(state)
    if state.stage == DialogStage.AWAITING_ADDRESS_CONFIRMATION and state.pending_address:
        return build_confirm_address_urdu(state.pending_address)
    if state.stage == DialogStage.AWAITING_ADDRESS:
        return ASK_ADDRESS_URDU
    if state.stage == DialogStage.AWAITING_CONFIRMATION:
        return ASK_CONFIRM_URDU
    return ""


def _append_agent_log(record: dict[str, Any]) -> None:
    directory = Path(__file__).resolve().parent.parent / "logs"
    file_path = directory / "agent_turns.jsonl"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and file_path.stat().st_size >= 1_048_576:
            backup = directory / "agent_turns.jsonl.1"
            if backup.exists():
                backup.unlink()
            file_path.replace(backup)
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return


def record_agent_turn(record: dict[str, Any]) -> None:
    _append_agent_log(record)


def run_agent_turn(
    user_text: str,
    order: Order,
    state: DialogState,
    *,
    shadow: bool = False,
) -> AgentOutcome:
    """Run tools on copies. shadow=True never returns ok."""
    started = time.perf_counter()
    trial_order = order.model_copy(deep=True)
    trial_state = state.model_copy(deep=True)
    started_stage = state.stage
    started_pending = (
        state.pending_item_id,
        state.pending_quantity,
        list(state.pending_choices),
        state.pending_choice_quantity,
        state.size_attempts,
        state.choice_attempts,
        state.order_confirmation_attempts,
        state.order_change_failures,
        state.summary_presented,
        state.pending_address,
    )
    previous_reply = state.last_reply
    budget = AdditionBudget()
    tools = tools_for_stage(state)
    messages = _payload(user_text, trial_order, trial_state, tools)
    tool_log: list[dict[str, Any]] = []
    rounds = 0
    call_count = 0
    mutation_ok = False
    mutation_failed = False
    code_reply = ""
    repeat_requested = False
    reason = ""
    final_text = ""
    validation = ""

    def fail(why: str) -> AgentOutcome:
        detail = trial_state.last_basket_edit_detail if trial_state.last_basket_edit_detail.startswith("rejected=") else ""
        outcome = AgentOutcome(
            ok=False,
            reason=why,
            rounds=rounds,
            tool_log=tool_log,
            final_text=final_text,
            validation=validation or why,
            latency=time.perf_counter() - started,
            basket_detail=detail,
        )
        return outcome

    try:
        while True:
            elapsed = time.perf_counter() - started
            remaining = config.AGENT_TURN_DEADLINE_SECONDS - elapsed
            if remaining <= 0:
                return fail("deadline")
            if rounds >= config.AGENT_MAX_ROUNDS:
                return fail("round_cap")
            rounds += 1
            result = call_agent_model(
                messages,
                tools,
                timeout_seconds=min(config.AGENT_CALL_TIMEOUT_SECONDS, remaining),
                max_completion_tokens=config.OPENAI_FALLBACK_MAX_COMPLETION_TOKENS,
            )
            if not result.tool_calls:
                final_text = result.content or ""
                break
            messages.append({
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in result.tool_calls
                ],
            })
            for call in result.tool_calls:
                if call_count >= config.AGENT_MAX_TOOL_CALLS:
                    raise AgentAbort("tool_budget_exceeded")
                call_count += 1
                tool_result = execute_tool(call.name, call.arguments, trial_order, trial_state, budget)
                tool_log.append({"name": call.name, "arguments": call.arguments, "result": tool_result})
                if tool_result.get("status") in {"added", "edited"} and tool_result.get("ok"):
                    mutation_ok = True
                if tool_result.get("error") in {"plan_rejected", "unknown_item", "item_unavailable", "unknown_size", "bad_quantity"} and call.name in {"add_item", "apply_basket_edits"}:
                    mutation_failed = True
                if tool_result.get("speak"):
                    code_reply = str(tool_result["speak"])
                if call.name == "repeat_last" and tool_result.get("ok"):
                    repeat_requested = True
                messages.append({
                    "role": "tool",
                    "_tool_call_id": call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })
            if code_reply and any(entry["name"] in {"transfer_to_human", "request_cancel"} for entry in tool_log):
                final_text = code_reply
                break
        if code_reply and trial_state.stage == DialogStage.HANDOFF:
            reply = code_reply
            validation = "ok"
        elif code_reply and trial_state.stage == DialogStage.AWAITING_CANCEL_CONFIRM:
            reply = code_reply
            validation = "ok"
        elif code_reply and not trial_order.items and any(entry["name"] == "request_cancel" and not entry["result"].get("ok") for entry in tool_log):
            reply = code_reply
            validation = "ok"
            if shadow:
                return AgentOutcome(
                    ok=False, reply=reply, reason="shadow", rounds=rounds, tool_log=tool_log,
                    final_text=reply, validation="ok", latency=time.perf_counter() - started, applied=False,
                )
            return AgentOutcome(
                ok=True, reply=reply, order=order, state=state, text_only=True, rounds=rounds,
                tool_log=tool_log, final_text=reply, validation="ok",
                latency=time.perf_counter() - started, applied=True,
            )
        elif trial_state.stage == DialogStage.AWAITING_SIZE and trial_state.pending_item_id and started_stage != DialogStage.AWAITING_SIZE:
            from app.dialog import _menu_item, _merge_parsed_additions, build_missing_size_urdu
            _merge_parsed_additions(user_text, trial_order, trial_state)
            item = _menu_item(trial_state.pending_item_id)
            reply = build_missing_size_urdu(item) if item else code_reply
            validation = "ok"
        elif repeat_requested and not mutation_ok:
            if not previous_reply.strip():
                return fail("empty")
            reply = previous_reply
            validation = "ok"
        else:
            finish_speak = _finish_items_speech(tool_log)
            if not final_text.strip():
                if not finish_speak:
                    return fail("empty")
                reply = finish_speak
                validation = "ok"
            else:
                validation = validate_agent_text(final_text, mutations_failed=mutation_failed and not mutation_ok)
                if validation != "ok":
                    return fail(validation)
                reply = fill_placeholders(final_text, trial_order)
                if finish_speak:
                    reply = f"{reply} {finish_speak}".strip()
                else:
                    guarded = started_stage in {
                        DialogStage.AWAITING_SIZE,
                        DialogStage.AWAITING_ADDRESS,
                        DialogStage.AWAITING_ADDRESS_CONFIRMATION,
                        DialogStage.AWAITING_CONFIRMATION,
                    } or bool(started_pending[2])
                    if guarded and not mutation_ok:
                        trial_state.pending_item_id = started_pending[0]
                        trial_state.pending_quantity = started_pending[1]
                        trial_state.pending_choices = list(started_pending[2])
                        trial_state.pending_choice_quantity = started_pending[3]
                        trial_state.size_attempts = started_pending[4]
                        trial_state.choice_attempts = started_pending[5]
                        trial_state.order_confirmation_attempts = started_pending[6]
                        trial_state.order_change_failures = started_pending[7]
                        trial_state.summary_presented = started_pending[8]
                        trial_state.pending_address = started_pending[9]
                        trial_state.stage = started_stage
                        question = _pending_question(trial_state)
                        if question:
                            reply = f"{reply} {question}"
                    elif mutation_ok:
                        from app.dialog import _reread_confirmation
                        from app.replies import build_update_reply_urdu
                        added = build_update_reply_urdu(trial_order, trial_state, "add")
                        reply, _complete = _reread_confirmation(
                            trial_order, trial_state, started_stage, f"{reply} {added}", False,
                        )
                    elif _close_for_mixed_no(user_text, trial_order):
                        reply = _with_total_and_close(reply, final_text, user_text, trial_order, trial_state)
                    elif trial_state.stage == DialogStage.AWAITING_MORE_ITEMS:
                        reply = _append_more_items(reply, final_text)
    except AgentAbort as exc:
        return fail(exc.reason)
    except LLMTimeoutError:
        return fail("timeout")
    except (LLMServiceError, LLMMalformedResponseError):
        return fail("model_error")
    except Exception:
        return fail("model_error")

    if shadow:
        return AgentOutcome(
            ok=False,
            reply=reply,
            reason="shadow",
            rounds=rounds,
            tool_log=tool_log,
            final_text=final_text or reply,
            validation=validation,
            latency=time.perf_counter() - started,
            applied=False,
            basket_detail=trial_state.last_basket_edit_detail,
        )
    opened_size = (
        trial_state.stage == DialogStage.AWAITING_SIZE and started_stage != DialogStage.AWAITING_SIZE
    )
    finished_items = bool(_finish_items_speech(tool_log))
    text_only = not mutation_ok and not finished_items and trial_state.stage not in {
        DialogStage.HANDOFF, DialogStage.AWAITING_CANCEL_CONFIRM,
    } and not opened_size
    return AgentOutcome(
        ok=True,
        reply=reply,
        order=trial_order,
        state=trial_state,
        text_only=text_only and not mutation_ok,
        rounds=rounds,
        tool_log=tool_log,
        final_text=final_text or reply,
        validation=validation,
        latency=time.perf_counter() - started,
        applied=True,
        basket_detail=trial_state.last_basket_edit_detail,
    )
