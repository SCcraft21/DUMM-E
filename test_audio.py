import subprocess
import os

def speak_ps(text: str):
    ps_script = os.path.abspath("_speak.ps1")
    escaped = text.replace("'", "''")
    with open(ps_script, "w", encoding="utf-8") as f:
        f.write(f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Speak('{escaped}')
''')
    res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps_script], capture_output=True, text=True)
    try:
        os.remove(ps_script)
    except Exception:
        pass
    print("PS Speak exit code:", res.returncode, "stdout:", res.stdout, "stderr:", res.stderr)

if __name__ == "__main__":
    speak_ps("Hello! Dumm-E audio output is now working on your computer.")
