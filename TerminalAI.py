import requests
import time
import datetime
import os
import sys
import shutil
import pandas as pd
import select
import json
import re
import threading
import socket
from rain import rain

if os.name == "nt":
    import msvcrt
else:
    import termios
    import tty

# ANSI colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
AI_COLOR = "\033[32m"

SERVER_URL = ""
selected_server = None
selected_api = None
IDLE_TIMEOUT = 30
REQUEST_TIMEOUT = 60
CSV_PATH = "endpoints.csv"
CONV_DIR = "conversations"
LOG_DIR = "logs"

def api_headers():
    return {"Content-Type": "application/json"}


def start_thinking_timer():
    start = time.time()
    stop_event = threading.Event()

    def updater():
        while not stop_event.is_set():
            elapsed = int(time.time() - start)
            print(
                f"\r{AI_COLOR}\U0001f5a5️ : thinking... ({elapsed}s){RESET}",
                end="",
                flush=True,
            )
            time.sleep(1)

    threading.Thread(target=updater, daemon=True).start()
    return start, stop_event


def stop_thinking_timer(start, stop_event, timed_out=False):
    stop_event.set()
    elapsed = int(time.time() - start)
    status = " [Timed out]" if timed_out else ""
    print(
        f"\r{AI_COLOR}\U0001f5a5️ : thinking... ({elapsed}s){status}{RESET}"
    )
    return elapsed


def get_input(prompt):
    if os.name == "nt":
        sys.stdout.write(prompt)
        sys.stdout.flush()
        buf = ""
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print()
                return buf.strip()
            elif ch == "\b":
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch == "\x1b":
                print()
                return "ESC"
            else:
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdout.write(prompt)
            sys.stdout.flush()
            buf = ""
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], None)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in ("\n", "\r"):
                        print()
                        return buf.strip()
                    elif ch == "\x7f":
                        if buf:
                            buf = buf[:-1]
                            sys.stdout.write("\b \b")
                            sys.stdout.flush()
                    elif ch == "\x1b":
                        print()
                        return "ESC"
                    else:
                        buf += ch
                        sys.stdout.write(ch)
                        sys.stdout.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)



def load_servers():
    try:
        df = pd.read_csv(CSV_PATH, keep_default_na=False)
    except Exception as e:
        print(f"{RED}Failed to load CSV: {e}{RESET}")
        return []
    if "id" in df.columns and "nickname" not in df.columns:
        df = df.rename(columns={"id": "nickname"})
    if "nickname" not in df.columns:
        df["nickname"] = df["ip"]
    df["port"] = df["port"].astype(int)
    df["is_active"] = df["is_active"].astype(bool)
    df["api_type"] = df["api_type"].str.lower()
    df["ping"] = pd.to_numeric(df.get("ping"), errors="coerce").fillna(float("inf"))
    df = df[df["is_active"] & (df["api_type"] == "ollama")]
    df = df.sort_values("ping")
    servers = []
    for ip, group in df.groupby("ip", sort=False):
        apis = {row["api_type"]: row["port"] for _, row in group.iterrows()}
        nickname = group.iloc[0]["nickname"]
        country = group.iloc[0].get("country", "")
        ping_val = group.iloc[0]["ping"]
        servers.append({"ip": ip, "nickname": nickname, "apis": apis, "country": country, "ping": ping_val})
    return servers

def persist_nickname(server, new_nick):
    try:
        df = pd.read_csv(CSV_PATH, keep_default_na=False)
        if "nickname" in df.columns:
            df.loc[df["ip"] == server["ip"], "nickname"] = new_nick
        if "id" in df.columns:
            df.loc[df["ip"] == server["ip"], "id"] = new_nick
        df.to_csv(CSV_PATH, index=False)
    except Exception as e:
        print(f"{RED}Failed to persist nickname: {e}{RESET}")

def conv_dir(model):
    safe = re.sub(r'[\\/:*?"<>|]', '_', model)
    path = os.path.join(CONV_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path

def list_conversations(model):
    path = conv_dir(model)
    convs = []
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".json"):
            fp = os.path.join(path, fn)
            title = fn[:-5]
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    msgs = data.get("messages", [])
                    if msgs:
                        first = msgs[0].get("content", "")
                        first = first.strip()
                        if first:
                            title = (first[:30] + ("..." if len(first) > 30 else ""))
            except Exception:
                pass
            convs.append({"file": fn, "title": title})
    return convs

def load_conversation(model, file):
    path = os.path.join(conv_dir(model), file)
    messages = []
    context = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                messages = data.get("messages", [])
                context = data.get("context")
            else:
                messages = data
        except Exception:
            messages = []
            context = None
    history = []
    for i in range(0, len(messages), 2):
        user = messages[i].get("content", "") if i < len(messages) else ""
        ai = messages[i + 1].get("content", "") if i + 1 < len(messages) else ""
        history.append({"user": user, "ai": ai})
    return messages, history, context

def save_conversation(model, file, messages, context=None):
    path = os.path.join(conv_dir(model), file)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"messages": messages, "context": context}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{RED}Failed to save conversation: {e}{RESET}")

def create_conversation_file(model, first_prompt):
    safe_prompt = re.sub(r'[\\/:*?"<>|]', '_', first_prompt.strip())
    safe_prompt = safe_prompt[:30]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_prompt}.json"
    # ensure directory exists
    conv_dir(model)
    return filename

def has_conversations(model):
    path = conv_dir(model)
    return any(fn.endswith(".json") for fn in os.listdir(path))


def ping_time(ip, port):
    """Measure latency to an IP and port using a TCP handshake."""
    try:
        start = time.time()
        with socket.create_connection((ip, int(port)), timeout=1):
            end = time.time()
        return (end - start) * 1000
    except Exception:
        return None


def update_pings():
    try:
        df = pd.read_csv(CSV_PATH, keep_default_na=False)
    except Exception:
        return
    if "ping" not in df.columns:
        df["ping"] = ""
    if "is_active" not in df.columns:
        df["is_active"] = True
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for idx, row in df.iterrows():
        active_val = str(row.get("is_active", "")).strip().lower()
        if active_val not in ("true", "1", "yes", "t"):
            continue
        latency = ping_time(row["ip"], row["port"])
        if latency is None:
            df.at[idx, "ping"] = ""
            df.at[idx, "is_active"] = False
            df.at[idx, "inactive_reason"] = "ping timeout"
        else:
            df.at[idx, "ping"] = latency
            df.at[idx, "is_active"] = True
            df.at[idx, "inactive_reason"] = ""
        df.at[idx, "last_check_date"] = now
    df.to_csv(CSV_PATH, index=False)

def select_server(servers):
    print(f"{CYAN}Available Servers:{RESET}")
    for i, s in enumerate(servers, 1):
        ping = "?" if s.get("ping", float("inf")) == float("inf") else f"{s['ping']:.1f} ms"
        print(f"{GREEN}{i}. {s['nickname']} ({s['ip']}) - {ping}{RESET}")
    while True:
        c = input(f"{CYAN}Select server: {RESET}").strip()
        if c.isdigit() and 1 <= int(c) <= len(servers):
            return servers[int(c) - 1]
        print(f"{RED}Invalid selection{RESET}")

def select_model(models):
    print(f"{CYAN}Available Models:{RESET}")
    for i, model in enumerate(models, 1):
        mark = " *" if has_conversations(model) else ""
        print(f"{GREEN}{i}. {model}{mark}{RESET}")
    print(f"{GREEN}0. [Back]{RESET}")
    while True:
        c = get_input(f"{CYAN}Select model: {RESET}")
        if c in ("ESC", "0") or c.lower() == "back":
            return None
        if c.isdigit() and 1 <= int(c) <= len(models):
            return models[int(c) - 1]
        print(f"{RED}Invalid selection{RESET}")

def fetch_models():
    try:
        r = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []) or data.get("data", []) or []:
            if isinstance(m, dict):
                models.append(m.get("id") or m.get("name") or "")
            elif isinstance(m, str):
                models.append(m)
        models = [m for m in models if m]
        return models
    except Exception as e:
        print(f"{RED}Failed to fetch models: {e}{RESET}")
        return []

def build_url(server, api):
    return f"http://{server['ip']}:{server['apis'][api]}"

def redraw_ui(model):
    os.system("cls" if os.name == "nt" else "clear")
    ip = selected_server['ip']
    port = selected_server['apis'][selected_api]
    print(f"{BOLD}{GREEN}AI Terminal Interface 🖥️ | {ip}:{port}{RESET}")
    print(f"{GREEN}Active Model: {model} | {selected_server['nickname']}{RESET}")
    print(f"{YELLOW}Type prompts below. Commands: /exit, /clear, /paste, /back, /print, /nick (Esc=Back){RESET}")

def display_connecting_box(x, y, w, h):
    for i in range(h):
        print(f"\033[{y+i};{x}H{' '*w}{RESET}")
    sys.stdout.flush()
    msg = "HACK THE PLANET"
    mx = x + (w - len(msg)) // 2
    my = y + h // 2
    print(f"\033[{my};{mx}H\033[1;32m{msg}{RESET}", flush=True)
    time.sleep(1.5)


def render_markdown(text):
    lines = text.splitlines()
    in_think = False
    in_code = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("<think>"):
            print(f"{YELLOW}{BOLD}🧠 Thinking...{RESET}")
            in_think = True
            continue
        if s.endswith("</think>"):
            in_think = False
            continue
        if in_think:
            continue
        if s == "</s>":
            print("-" * 60)
            continue
        if s.startswith("```"):
            in_code = not in_code
            print("=" * 60)
            continue
        print(ln.replace("</s>", ""))
    print()

def reprint_history(history):
    for e in history:
        print(f"🧑 : {e['user']}")
        print(f"{AI_COLOR}🖥️ : ",end='')
        render_markdown(e['ai'])

def select_conversation(model):
    convs = list_conversations(model)
    if not convs:
        print(f"{CYAN}No previous conversations found.{RESET}")
        return None, [], [], None

    print(f"{CYAN}Conversations:{RESET}")
    print(f"{GREEN}1. Start new conversation{RESET}")
    for i, c in enumerate(convs, 2):
        print(f"{GREEN}{i}. {c['title']}{RESET}")
    print(f"{GREEN}0. [Back]{RESET}")

    while True:
        choice = get_input(f"{CYAN}Select conversation: {RESET}")
        if choice in ("ESC", "0") or choice.lower() == "back":
            return 'back', None, None, None
        if choice == '1':
            return None, [], [], None
        if choice.isdigit() and 2 <= int(choice) < len(convs) + 2:
            file = convs[int(choice) - 2]['file']
            messages, history, context = load_conversation(model, file)
            return file, messages, history, context
        print(f"{RED}Invalid selection{RESET}")

def chat_loop(model, conv_file, messages=None, history=None, context=None):
    global SERVER_URL, selected_server, selected_api
    redraw_ui(model)
    if messages is None:
        messages = []
    if history is None:
        history = []
    if history:
        reprint_history(history)
    try:
        while True:
            start = time.time()
            sys.stdout.write(f"{RESET}\U0001f9d1 : ")
            sys.stdout.flush()
            user_input = ""
            if os.name == "nt":
                while True:
                    if time.time() - start > IDLE_TIMEOUT:
                        rain(persistent=True)
                        redraw_ui(model)
                        reprint_history(history)
                        sys.stdout.write(f"{RESET}\U0001f9d1 : {user_input}")
                        sys.stdout.flush()
                        start = time.time()
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('\r', '\n'):
                            print()
                            break
                        elif ch == '\x1b':
                            print()
                            return "back"
                        elif ch == '\b':
                            if user_input:
                                user_input = user_input[:-1]
                                sys.stdout.write('\b \b')
                                sys.stdout.flush()
                        else:
                            user_input += ch
                            sys.stdout.write(ch)
                            sys.stdout.flush()
                        start = time.time()
                    else:
                        time.sleep(0.1)
                user_input = user_input.strip()
                if not user_input:
                    continue
            else:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                try:
                    while True:
                        if time.time() - start > IDLE_TIMEOUT:
                            rain(persistent=True)
                            redraw_ui(model)
                            reprint_history(history)
                            sys.stdout.write(f"{RESET}\U0001f9d1 : {user_input}")
                            sys.stdout.flush()
                            tty.setcbreak(fd)
                            start = time.time()
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if rlist:
                            ch = sys.stdin.read(1)
                            if ch in ('\n', '\r'):
                                print()
                                break
                            elif ch == '\x7f':
                                if user_input:
                                    user_input = user_input[:-1]
                                    sys.stdout.write('\b \b')
                                    sys.stdout.flush()
                            elif ch == '\x1b':
                                print()
                                return "back"
                            else:
                                user_input += ch
                                sys.stdout.write(ch)
                                sys.stdout.flush()
                            start = time.time()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                user_input = user_input.strip()
                if not user_input:
                    continue

            cmd = user_input.lower()
            if cmd == "/exit":
                return "exit"
            elif cmd == "/clear":
                redraw_ui(model)
                continue
            elif cmd == "/print":
                if history:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs(LOG_DIR, exist_ok=True)
                    fn = os.path.join(LOG_DIR, f"chat_{ts}.txt")
                    with open(fn, "w", encoding="utf-8") as f:
                        for h in history:
                            f.write(f"User: {h['user']}\nAI: {h['ai']}\n\n")
                    print(f"{YELLOW}Saved to {fn}{RESET}")
                else:
                    print(f"{RED}No history{RESET}")
                continue
            elif cmd == "/paste":
                print(f"{CYAN}Paste (Ctrl+D/Ctrl+Z to end){RESET}")
                lines = []
                try:
                    while True:
                        lines.append(input())
                except:
                    pass
                user_input = "".join(lines).strip()

            elif cmd == "/back":
                return "back"

            elif cmd == "/nick":
                new = input(f"{CYAN}New nickname: {RESET}").strip()
                if new:
                    selected_server["nickname"] = new
                    persist_nickname(selected_server, new)
                    print(f"{YELLOW}Nickname saved{RESET}")
                redraw_ui(model)
                continue
            if conv_file is None:
                conv_file = create_conversation_file(model, user_input)

            start, stop_event = start_thinking_timer()
            headers = api_headers()
            resp = None
            server_failed = False
            timed_out = False
            timeout_val = REQUEST_TIMEOUT * (len(messages) + 1)

            chat_paths = [
                "/v1/chat/completions",
                "/v1/chat",
                "/api/chat",
                "/chat",
            ]

            for p in chat_paths:
                try:
                    payload = {
                        "model": model,
                        "messages": messages + [{"role": "user", "content": user_input}],
                        "stream": False,
                    }
                    r = requests.post(
                        f"{SERVER_URL}{p}",
                        headers=headers,
                        json=payload,
                        timeout=(timeout_val, timeout_val),
                    )
                    r.raise_for_status()
                    data = r.json()
                    msg = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                        or data.get("choices", [{}])[0].get("text")
                        or data.get("completion")
                    )
                    if msg:
                        resp = msg
                        break
                except requests.exceptions.Timeout:
                    print(f"\n{RED}{p} request timed out after {timeout_val}s{RESET}")
                    server_failed = True
                    timed_out = True
                    break
                except requests.exceptions.HTTPError as e:
                    print(f"\n{RED}Failed {p}: {e}{RESET}")
                    continue
                except requests.exceptions.RequestException as e:
                    print(f"\n{RED}Failed {p}: {e}{RESET}")
                    server_failed = True
                    break

            if not resp:
                gen = [
                    "/v1/completions",
                    "/v1/complete",
                    "/api/generate",
                    "/generate",
                ]
                for p in gen:
                    try:
                        if p == "/api/generate":
                            payload = {"model": model, "prompt": user_input, "stream": False}
                            if context:
                                payload["context"] = context
                        else:
                            prompt_text = "\n".join([
                                f"{m['role']}: {m['content']}" for m in messages + [{"role": "user", "content": user_input}]
                            ])
                            payload = {"model": model, "prompt": prompt_text, "stream": False}
                        r = requests.post(
                            f"{SERVER_URL}{p}",
                            headers=headers,
                            json=payload,
                            timeout=(timeout_val, timeout_val),
                        )
                        r.raise_for_status()
                        data = r.json()
                        msg = (
                            data.get("choices", [{}])[0].get("text")
                            or data.get("completion")
                            or data.get("response")
                        )
                        if msg:
                            resp = msg
                            if "context" in data:
                                context = data.get("context")
                            break
                    except requests.exceptions.Timeout:
                        print(f"\n{RED}{p} request timed out after {timeout_val}s{RESET}")
                        server_failed = True
                        timed_out = True
                        break
                    except requests.exceptions.HTTPError as e:
                        print(f"\n{RED}Failed {p}: {e}{RESET}")
                        continue
                    except requests.exceptions.RequestException as e:
                        print(f"\n{RED}Failed {p}: {e}{RESET}")
                        server_failed = True
                        break

            if not resp:
                print(f"{RED}[Error] No response{RESET}")
                server_failed = True

            elapsed = stop_thinking_timer(start, stop_event, timed_out)
            log_msg = (
                f"Timed out after {elapsed}s" if timed_out else f"Finished thinking in {elapsed}s"
            )
            print(f"{YELLOW}{log_msg}{RESET}")

            if server_failed:
                try:
                    df = pd.read_csv(CSV_PATH, keep_default_na=False)
                    df.loc[(df["ip"] == selected_server["ip"]) & (df["port"] == selected_server["apis"][selected_api]), "is_active"] = False
                    df.to_csv(CSV_PATH, index=False)
                    print(f"{RED}Server marked inactive due to failure{RESET}")
                except Exception as ex:
                    print(f"{RED}Failed to update CSV: {ex}{RESET}")
                return "server_inactive"

            if not timed_out and resp:
                print("\r\033[K", end='')
                print(f"{AI_COLOR}\U0001f5a5️ : ", end='')
                render_markdown(resp)
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": resp})
                history.append({"user": user_input, "ai": resp})
                save_conversation(model, conv_file, messages, context)

    except KeyboardInterrupt:
        print(f"{YELLOW}Session interrupted{RESET}")
        if history:
            save = input(f"{CYAN}Save log? (y/n): {RESET}").lower()
            if save == "y":
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(LOG_DIR, exist_ok=True)
                fn = os.path.join(LOG_DIR, f"chat_{ts}.txt")
                with open(fn, "w", encoding="utf-8") as f:
                    for h in history:
                        f.write(f"User: {h['user']}\nAI: {h['ai']}\n\n")
                print(f"Saved to {fn}")

# Main loop
if __name__ == "__main__":
    selected_api = 'ollama'

    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    box_w, box_h = 30, 5
    box_x = (cols - box_w) // 2
    box_y = (rows - box_h) // 2
    stop_rain = threading.Event()
    rain_thread = threading.Thread(
        target=rain,
        kwargs={
            "persistent": True,
            "stop_event": stop_rain,
            "box_top": box_y,
            "box_bottom": box_y + box_h - 1,
            "box_left": box_x,
            "box_right": box_x + box_w - 1,
        },
    )
    rain_thread.start()
    display_connecting_box(box_x, box_y, box_w, box_h)
    ping_thread = threading.Thread(target=update_pings, daemon=True)
    ping_thread.start()
    ping_thread.join()
    stop_rain.set()
    rain_thread.join()
    display_connecting_box(box_x, box_y, box_w, box_h)

    while True:
        # 1. Load and pick a server
        servers = load_servers()
        srv_list = [s for s in servers if selected_api in s["apis"]]
        if not srv_list:
            print(f"{RED}No active servers available{RESET}")
            sys.exit(1)

        selected_server = select_server(srv_list)
        SERVER_URL = build_url(selected_server, selected_api)

        # 2. Fetch models
        models = fetch_models()
        if not models:
            print(f"{RED}No models found on selected server{RESET}")
            try:
                df = pd.read_csv(CSV_PATH, keep_default_na=False)
                df.loc[(df["ip"] == selected_server["ip"]) & (df["port"] == selected_server["apis"][selected_api]), "is_active"] = False
                df.to_csv(CSV_PATH, index=False)
            except Exception as ex:
                print(f"{RED}Failed to update CSV: {ex}{RESET}")
            continue

        # 3. Pick a model
        while True:
            chosen = select_model(models)
            if chosen is None:
                break  # back to server selection

            # 4. Pick or start conversation
            while True:
                conv_file, messages, history, context = select_conversation(chosen)
                if conv_file == 'back':
                    break  # back to model selection

                result = chat_loop(chosen, conv_file, messages, history, context)
                if result == 'back':
                    continue  # back to conversation selection
                elif result == 'server_inactive':
                    conv_file = 'server_inactive'
                    break
                elif result == 'exit':
                    sys.exit(0)
                else:
                    sys.exit(0)
            if conv_file == 'back':
                continue  # select model again
            elif conv_file == 'server_inactive':
                break  # back to server selection
            else:
                break
