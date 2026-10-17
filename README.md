# Restaurant Voice Agent

Voice ordering demo for a restaurant. The app uses:

- OpenAI for the order-taking LLM
- Faster Whisper for speech-to-text
- Piper HTTP server for Urdu text-to-speech

## Setup

Open PowerShell and go to the project folder:

```powershell
cd C:\Users\Fawaz\Desktop\resturant-project\restaurant-voice-agent
```

Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

Make sure `.env` contains your real OpenAI key:

```env
OPENAI_API_KEY=your_real_openai_key_here
```

## Run The Full Voice App

You need two terminals.

### Terminal 1: Start Piper TTS

Keep this terminal open:

```powershell
(venv) C:\Users\Fawaz\Desktop\resturant-project\restaurant-voice-agent>.\venv\Scripts\python.exe -m piper.http_server -m ur_PK-aegis_female-medium --data-dir ..\paper\voices
```

Optional check:

```powershell
curl http://127.0.0.1:5000/info
```

### Terminal 2: Run The Voice Agent

Run this after Piper is started:

```powershell
(venv) C:\Users\Fawaz\Desktop\resturant-project\restaurant-voice-agent>.\venv\Scripts\python.exe run_voice.py --seconds 5 --latency
```

The app records 5 seconds per turn, transcribes your speech, processes the order with OpenAI, then speaks the reply with Piper.

## Text-Only Mode

Use this if you want to test the ordering flow without microphone or TTS:

```powershell
.\venv\Scripts\python.exe run_text.py
```

## Taking Orders

`BASKET_EDIT_LLM_ENABLED=true` (default) enables semantic editing of an existing
basket. Every basket-content reply in the item-review phase is interpreted by
OpenAI as a plan of `add`, `set`, `reduce`, and `remove` operations. This covers
different phrasing and mixed requests rather than matching whole example sentences.
Pure yes/no replies stay local. This toggle is independent of the general LLM
fallback toggle; set it to false to keep the earlier deterministic edit path.

Examples: `دو نان کم کر دو` subtracts two, `نان ہٹا دو` removes the whole naan
line, `ٹوٹل چار نان` sets the final count to four, and `دو نان اور` adds two.
Python validates items, sizes and counts, performs the arithmetic, and commits
all proposed operations together. A zero final count removes the line; reducing
more than exists or choosing an ambiguous size asks for clarification without
changing the basket. The editor cannot change addresses, prices or order status.
Look for `path=openai_basket_edit` and a `[basket_edit]` diagnostic with the
model status/confidence or failure reason; the request timeout is at most six seconds,
with no retries. Unclear requests retain bounded context for a follow-up answer.

One request can contain several items, for example `ایک نان اور ایک پوری چکن
کڑاہی` or `2 naan and 1 full chicken karahi`. Quantities and sizes are handled
per item. Missing sizes are requested before the staged additions are committed.

After adding items, the app reads back the current basket and total, then asks
`کیا آپ کچھ اور شامل کرنا چاہتے ہیں؟`. Yes asks which item to add; customers can
also name another item directly. No (`نہیں`, `nhi`, `no thanks`, `بس اتنا`)
moves to the delivery-address step. The extracted address and final order still
require their own confirmations.

Quantity corrections such as `نان چار کرنا چاہتا ہوں` or `ٹوٹل نان چار، پانچ
نہیں` use the structured LLM resolver to set the final total, rather than add
that many more items. Explicit additions such as `دو نان اور شامل کریں` still
increment the existing quantity. Unclear numbers ask for the total again;
the customer can answer with a number such as `چار`. Corrections cannot be
applied as an add operation or use a number that the customer explicitly rejected.

## Address Extraction

`ADDRESS_LLM_EXTRACTION_ENABLED=true` is the default. New addresses and address
corrections use one OpenAI call to select the location words from the original
transcript, excluding introductory words such as `nhi mera address ye hai`.
The app copies the selected words directly; the model cannot rewrite place names.
Only whitespace and separator punctuation are normalized. This does not validate
that a location exists or repair STT mistakes in place names.

Clear yes/no replies stay local and make no API call. Every extracted address is
held as `pending_address` and read back. Only a local yes saves it. The final
order summary then needs a separate confirmation.

Confirmation vocabulary includes common Roman Urdu spellings (`hn`, `han`,
`haan`, `nhi`, `nahi`, `nahin`, `nhin`) with `g`/`ji`/`jee`/`gee` in either order,
joined or spaced. Observed Whisper forms `ہنڈی` and `ڈی، ہاں` are handled locally
during address read-back; `ہنڈی` is not a general food/order confirmation.
Bare quantities such as `ڈیڑھ` remain ambiguous. Location words accompanying a
yes/no phrase still go through address extraction and a new read-back.

Extraction uses a four-second request timeout (or the configured timeout if
shorter), no retries, and a confidence threshold of 0.6 for proposing a read-back
candidate. It does not measure whether the location exists. Under `--latency`, look
for `path=openai_address_extraction` and an `[address]` line with the confidence
or service-error reason. Failed/uncertain extraction asks again and
does not overwrite the confirmed address. Three failures end the session with a
request to contact staff; there is no automatic connection to staff.

Set `ADDRESS_LLM_EXTRACTION_ENABLED=false` to use deterministic address parsing.
This setting is separate from the general `HYBRID_LLM_FALLBACK_ENABLED` toggle.

While the app is waiting for an address, location replies reach extraction before
menu matching, so STT words such as coke/karahi cannot accidentally add food.
Explicit commands such as "add garlic naan" still use the normal order flow.

## STT GPU And Benchmarks

CPU remains the safe default. To test NVIDIA GPU STT, set these in `.env`:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

If CUDA cannot initialize, the app exits clearly instead of silently falling back to CPU. Recent Faster Whisper/CTranslate2 builds usually need CUDA 12 and cuDNN 9; older CUDA 11 setups may need `ctranslate2==3.24.0`.

Record benchmark clips:

```powershell
.\venv\Scripts\python.exe run_stt.py --record-clip haan-ji --seconds 3
.\venv\Scripts\python.exe run_stt.py --record-clip haan-ji --seconds 3 --force
```

Transcribe a WAV file:

```powershell
.\venv\Scripts\python.exe run_stt.py --file data\test_audio\haan-ji.wav --latency
```

Run one benchmark per config, then compare:

```powershell
.\venv\Scripts\python.exe benchmark_stt.py --dir data\test_audio --runs 5 --warmup 1
.\venv\Scripts\python.exe benchmark_stt.py --compare benchmark-results-small-cpu-int8.json benchmark-results-small-cuda-float16.json
```

Adopt GPU only when the compare command prints `ADOPT`.

## Tests

Run tests from the project folder:

```powershell
.\venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

Live OpenAI tests are skipped by default. To enable them:

```powershell
$env:RUN_LIVE_OPENAI_TESTS="1"
.\venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

## Troubleshooting

- If voice output does not work, confirm Terminal 1 is still running Piper.
- If OpenAI fails, check `OPENAI_API_KEY` in `.env`.
- If microphone recording fails, check Windows microphone privacy permissions.
