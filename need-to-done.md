# Need To Done

Ranked implementation plan for improving accuracy, address handling, and speed.

## Critical 1: Fix Address Error And Confirmation Flow

Goal: never save a wrong address directly from unclear speech.

### Task 1.1: Add pending address state

- Add `pending_address` to dialog state.
- When the app is waiting for address, store new address text in `pending_address`, not `order.delivery_address`.
- Do not allow final order confirmation while `pending_address` is unconfirmed.

### Task 1.2: Ask user to confirm address before saving

- After hearing an address, reply with a confirmation question.
- Example:

```text
کیا آپ کا پتہ یہ ہے: پرانا لاری اڈہ قصور؟
```

- Only move `pending_address` into `order.delivery_address` after a clear yes.

### Task 1.3: Handle address correction words

- Detect correction starts like:
  - "نہیں"
  - "no"
  - "not this"
  - "my address is"
  - "میرا ایڈریس ہے"
  - "میرا پتہ ہے"
- Remove correction words before saving the actual address candidate.
- Example input:

```text
نہیں، میرا ادریس ہے پرانا لاری ادہ قصور۔
```

- Expected saved address:

```text
پرانا لاری اڈہ قصور
```

### Task 1.4: Add address rejection flow

- If user says the read-back address is wrong, clear `pending_address`.
- Ask for the address again.
- Do not keep the wrong previous address.

### Task 1.5: Add address-specific tests

- Test that unclear food-like text is not saved as address immediately.
- Test that correction phrases are stripped.
- Test that final confirmation is blocked until address is confirmed.
- Test that rejected address is cleared.

## Critical 2: Add Confidence Gates Before Mutating Orders

Goal: avoid adding items, address, or confirmation from uncertain STT text.

### Task 2.1: Add confidence threshold per intent

- Use stricter confidence for:
  - address
  - final confirmation
  - item deletion
  - quantity changes
- If confidence is low, ask a clarification question instead of updating the order.

### Task 2.2: Add safe fallback for unclear STT

- If parser and OpenAI are both uncertain, reply:

```text
معذرت، میں صحیح سمجھ نہیں سکی۔ براہ کرم دوبارہ بتائیں۔
```

- Do not mutate order state.

### Task 2.3: Require explicit final confirmation

- Final confirmation should only happen after:
  - order has items
  - address is confirmed
  - summary was read back
  - user gives a clear yes/confirm phrase

### Task 2.4: Add mutation safety tests

- Test that unclear text does not add items.
- Test that unclear text does not overwrite address.
- Test that unclear text does not confirm order.

## High 3: Improve OpenAI Validator For Ambiguous Turns

Goal: use OpenAI as a validator/classifier, not just an order parser.

### Task 3.1: Add structured intent validator

- For unclear turns, ask OpenAI to classify the user text as:
  - add item
  - size answer
  - address
  - address correction
  - confirmation
  - cancellation
  - unclear
- Return JSON/tool output only.

### Task 3.2: Include current dialog state in validator prompt

- Include:
  - current stage
  - pending item
  - pending address
  - current order summary
  - whether address is confirmed

### Task 3.3: Add low-confidence behavior

- If OpenAI returns low confidence, ask user to repeat.
- Do not update the order.

### Task 3.4: Add validator tests

- Mock OpenAI responses.
- Verify no mutations happen on low confidence.
- Verify address correction is classified correctly.

## High 4: Improve Speech-To-Text Accuracy

Goal: reduce bad transcripts before they reach the parser.

### Task 4.1: Test larger Whisper model

- Try in `.env`:

```env
WHISPER_MODEL=medium
WHISPER_BEAM_SIZE=7
WHISPER_LANGUAGE=ur
```

- Compare accuracy and latency against:

```env
WHISPER_MODEL=small
WHISPER_BEAM_SIZE=5
```

### Task 4.2: Add STT benchmark script

- Create a small script that runs fixed test audio samples through Whisper.
- Save transcript and timing results.
- Compare models using the same audio.

### Task 4.3: Improve initial prompts

- Add common menu words, address words, city names, and restaurant vocabulary to Whisper prompts.
- Keep prompts generic enough for new customers.

### Task 4.4: Optional cloud STT comparison

- Compare local Whisper with ElevenLabs or OpenAI transcription.
- Measure:
  - accuracy
  - latency
  - cost
  - Urdu/Roman Urdu handling

## High 5: Replace Fixed Recording With Better Capture

Goal: avoid recording silence/noise and make the app feel faster.

### Task 5.1: Add push-to-talk as recommended voice mode

- Prefer `run_mic.py` style interaction for real use.
- User controls when recording starts and stops.

### Task 5.2: Add VAD/silence detection

- Start recording when speech begins.
- Stop when user is silent for a short period.
- Keep maximum recording length as a safety limit.

### Task 5.3: Add recording quality checks

- Detect very low volume.
- Ask user to repeat if audio is too quiet.
- Avoid sending empty/noisy audio to STT.

## Medium 6: Reduce OpenAI Calls

Goal: improve speed and lower cost.

### Task 6.1: Expand deterministic parser coverage

- Handle more common menu/order cases locally.
- Keep it generic by using menu aliases and state, not one-off customer phrases.

### Task 6.2: Call OpenAI only for ambiguous turns

- Clear item add/remove/size/quantity cases should stay deterministic.
- OpenAI should only handle unclear speech, corrections, and complex language.

### Task 6.3: Keep OpenAI prompts compact

- Send only current state and compact menu fields.
- Do not send long conversation history unless needed.

### Task 6.4: Tune OpenAI token limits

- Lower output token limits where safe.
- Keep fallback resolver output short and structured.

## Medium 7: Speed Up TTS

Goal: reduce Piper delay and avoid repeated synthesis.

### Task 7.1: Pre-generate common replies

- Save WAV files for common replies:
  - greeting
  - ask size
  - ask address
  - ask repeat
  - order added
  - confirmation question

### Task 7.2: Add TTS cache

- Cache Piper output by exact reply text.
- Reuse cached WAV when the same reply appears again.

### Task 7.3: Keep Piper server always running

- Do not restart Piper per run.
- Document and enforce two-terminal workflow.

## Medium 8: Improve Conversation Recovery

Goal: help users fix mistakes naturally.

### Task 8.1: Add correction intent

- Support phrases like:
  - "نہیں یہ غلط ہے"
  - "address change karo"
  - "item hata do"
  - "size full kar do"

### Task 8.2: Add undo last change

- Keep previous order snapshot.
- If user says the last update was wrong, restore previous snapshot.

### Task 8.3: Read back changed field only

- For address correction, only read back address.
- For item update, only read back updated item and total.

## Low 9: Framework Or API Layer

Goal: improve integration, not raw speed.

### Task 9.1: Add FastAPI only if needed

- Use FastAPI if this becomes a backend for:
  - web app
  - mobile app
  - restaurant dashboard
  - external ordering system

### Task 9.2: Add API endpoints

- Possible endpoints:
  - `GET /menu`
  - `POST /text-order`
  - `POST /voice-order`
  - `GET /orders`

### Task 9.3: Keep CLI working

- Do not remove `run_text.py`, `run_voice.py`, or `run_mic.py`.
- API layer should reuse the existing app logic.

## Low 10: Observability And Debugging

Goal: make accuracy issues easier to diagnose.

### Task 10.1: Save per-turn debug logs

- Log:
  - raw STT text
  - normalized text
  - detected intent
  - confidence
  - state before and after
  - latency breakdown

### Task 10.2: Add mistake review file

- Save unclear or corrected turns to a review JSONL file.
- Use this file to improve prompts and tests.

### Task 10.3: Add replay tests from real mistakes

- Turn real failed conversations into test cases.
- Keep customer-specific text only as examples, not hardcoded production rules.
