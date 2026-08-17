#!/usr/bin/env python3
"""
Harley Inference Server
Wraps llama.cpp with persona injection, vision support, and live camera.
Port 5051 — no admin needed.
"""
import os, sys, json, io, base64, threading, time
from datetime import datetime
from flask import Flask, request, jsonify, Response, redirect
from flask_cors import CORS
import urllib.request

# ── Config ────────────────────────────────────────────────────────────
LLAMA_URL = os.environ.get('LLAMA_URL', 'http://localhost:8080')
MODEL_PATH = os.environ.get('MODEL_PATH', 
    r'C:\HarleysPlace\models\lmstudio-community\Qwen2.5-VL-3B-Instruct-GGUF\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf')

PERSONA_DIR = os.path.join(os.path.dirname(__file__), 'personas')
UI_DIR = os.path.join(os.path.dirname(__file__), 'ui')
PORT = int(os.environ.get('HARLEY_PORT', 5051))

app = Flask(__name__, static_folder=UI_DIR, static_url_path='')
CORS(app)

# ── Load persona ──────────────────────────────────────────────────────
def load_persona(name='harley'):
    path = os.path.join(PERSONA_DIR, f'{name}.json')
    with open(path) as f:
        return json.load(f)

HARLEY = load_persona()
SYSTEM_PROMPT = HARLEY['system_prompt']

# ── Chat history ──────────────────────────────────────────────────────
chat_histories = {}  # session_id -> messages list
MAX_HISTORY = 30

def get_history(session_id='default'):
    if session_id not in chat_histories:
        chat_histories[session_id] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    return chat_histories[session_id]

def trim_history(session_id='default'):
    h = chat_histories[session_id]
    if len(h) > MAX_HISTORY + 1:
        # Keep system prompt + last N messages
        chat_histories[session_id] = [h[0]] + h[-(MAX_HISTORY):]

# ── API ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    text = data.get('text', '').strip()
    image_b64 = data.get('image')
    session_id = data.get('session', 'default')
    live_mode = data.get('live', False)
    
    if not text and not image_b64:
        return jsonify({'error': 'No input'}), 400
    
    history = get_history(session_id)
    
    # Build user message
    if image_b64:
        msg = {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
            {'type': 'text', 'text': text or 'What do you see?'}
        ]}
    else:
        msg = {'role': 'user', 'content': text}
    
    history.append(msg)
    trim_history(session_id)
    
    # Send to llama.cpp
    send_history = history.copy()
    if live_mode:
        # Inject live context
        send_history[0] = {'role': 'system', 'content': SYSTEM_PROMPT + '\n\nYou are watching through a live camera feed right now. React to what you see naturally.'}
    
    try:
        payload = json.dumps({
            'model': MODEL_PATH,
            'messages': send_history,
            'stream': False,
            'max_tokens': 500 if live_mode else 800
        }).encode()
        
        req = urllib.request.Request(f'{LLAMA_URL}/v1/chat/completions', 
                                     data=payload,
                                     headers={'Content-Type': 'application/json'})
        result = json.loads(urllib.request.urlopen(req, timeout=60).read())
        reply = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        history.append({'role': 'assistant', 'content': reply})
        trim_history(session_id)
        
        return jsonify({'reply': reply, 'model': MODEL_PATH})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vision', methods=['POST'])
def api_vision():
    """Single image analysis"""
    data = request.get_json()
    image_b64 = data.get('image')
    prompt = data.get('prompt', 'Describe what you see in detail.')
    
    if not image_b64:
        return jsonify({'error': 'No image'}), 400
    
    try:
        payload = json.dumps({
            'model': MODEL_PATH,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}},
                {'type': 'text', 'text': prompt}
            ]}],
            'stream': False,
            'max_tokens': 500
        }).encode()
        
        req = urllib.request.Request(f'{LLAMA_URL}/v1/chat/completions',
                                     data=payload,
                                     headers={'Content-Type': 'application/json'})
        result = json.loads(urllib.request.urlopen(req, timeout=30).read())
        reply = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/persona')
def api_persona():
    return jsonify({
        'name': HARLEY['name'],
        'description': HARLEY['description'],
        'version': HARLEY['version']
    })

@app.route('/api/health')
def api_health():
    try:
        req = urllib.request.Request(f'{LLAMA_URL}/health')
        resp = urllib.request.urlopen(req, timeout=5).read()
        return jsonify({'status': 'ok', 'llama': json.loads(resp)})
    except:
        return jsonify({'status': 'degraded', 'llama': 'unreachable'}), 503

@app.route('/api/reset', methods=['POST'])
def api_reset():
    session_id = request.get_json().get('session', 'default')
    chat_histories[session_id] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    return jsonify({'ok': True})

if __name__ == '__main__':
    print(f'\n  Harley Inference Engine v{HARLEY["version"]}')
    print(f'  Persona: {HARLEY["name"]} — {HARLEY["description"]}')
    print(f'  Model: {MODEL_PATH}')
    print(f'  Backend: {LLAMA_URL}')
    print(f'  Server: http://localhost:{PORT}\n')
    app.run(host='0.0.0.0', port=PORT, debug=False)
