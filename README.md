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
