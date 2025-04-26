import requests
import time
import datetime
import os
import random
import sys
import shutil
import pandas as pd
import select

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
AI_COLOR = "\033[32;40m"

SERVER_URL = ""
selected_server = None
selected_api = None
IDLE_TIMEOUT = 30
CSV_PATH = "endpoints.csv"

def api_headers():
    return {"Content-Type": "application/json"}

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
    df = df[df["is_active"] & (df["api_type"] == "ollama")]
    servers = []
    for ip, group in df.groupby("ip"):
        apis = {row["api_type"]: row["port"] for _, row in group.iterrows()}
        nickname = group.iloc[0]["nickname"]
        servers.append({"ip": ip, "nickname": nickname, "apis": apis})
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

def select_server(servers):
    print(f"{CYAN}Available Servers:{RESET}")
    for i, s in enumerate(servers, 1):
        print(f"{GREEN}{i}. {s['nickname']} ({s['ip']}){RESET}")
    while True:
        c = input(f"{CYAN}Select server: {RESET}").strip()
        if c.isdigit() and 1 <= int(c) <= len(servers):
            return servers[int(c) - 1]
        print(f"{RED}Invalid selection{RESET}")

def select_model(models):
    print(f"{CYAN}Available Models:{RESET}")
    for i, model in enumerate(models, 1):
        print(f"{GREEN}{i}. {model}{RESET}")
    while True:
        c = input(f"{CYAN}Select model: {RESET}").strip()
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
    print(f"{YELLOW}Type prompts below. Commands: /exit, /clear, /paste, /back, /print, /nick{RESET}")

def display_connecting_box():
    cols, rows = shutil.get_terminal_size(fallback=(80,24))
    w,h = 30,5
    x = (cols-w)//2; y=(rows-h)//2
    print("\033[?25l", end='')
    os.system("cls" if os.name=="nt" else "clear")
    for i in range(h):
        print(f"\033[{y+i};{x}H\033[40m{' '*w}{RESET}")
    msg="CONNECTING TO SERVER"
    mx = x + (w-len(msg))//2; my = y + h//2
    print(f"\033[{my};{mx}H\033[1;32;40m",end='')
    for char in msg:
        print(char, end='', flush=True)
        time.sleep(0.08)
    time.sleep(1.5)
    print(RESET+"\033[?25h")

def matrix_rain(persistent=False, duration=3):
    charset = "01$#*+=-░▒▓▌▐▄▀▁▂▃▅▆▇█ﾊﾐﾋｰｱｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ""01$#*+=-"
    try:
        print("\033[?25l", end='')
        os.system("cls" if os.name == "nt" else "clear")
        end_time = time.time() + duration if not persistent else None
        columns, rows = shutil.get_terminal_size(fallback=(80, 24))
        trail_length = 6
        drops = [random.randint(0, rows) for _ in range(columns)]
        if os.name != "nt":
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        while persistent or time.time() < end_time:
            for col in range(columns):
                drop_pos = drops[col]
                for t in range(trail_length):
                    y = drop_pos - t
                    if 0 < y < rows:
                        char = random.choice(charset)
                        shade = "\033[92m" if t==0 else "\033[32m" if t==1 else "\033[32;2m" if t==2 else "\033[2;32m" if t==3 else "\033[30m"
                        print(f"\033[{y};{col+1}H{shade}{char}{RESET}", end='')
                drops[col] = (drops[col] + 1) % (rows + trail_length)
            sys.stdout.flush()
            time.sleep(0.08)
            if persistent:
                if os.name == "nt":
                    if msvcrt.kbhit():
                        msvcrt.getch()
                        break
                else:
                    dr, _, _ = select.select([sys.stdin], [], [], 0)
                    if dr:
                        os.read(sys.stdin.fileno(), 1)
                        break
    finally:
        if os.name != "nt":
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        time.sleep(0.5)
        print("\033[0m\033[2J\033[H\033[?25h", end='')

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

def chat_loop(model):
    global SERVER_URL, selected_server, selected_api
    redraw_ui(model)
    history = []
    try:
        while True:
            start = time.time()
            sys.stdout.write(f"{RESET}\U0001f9d1 : ")
            sys.stdout.flush()
            user_input = ""
            while True:
                if time.time() - start > IDLE_TIMEOUT:
                    matrix_rain(persistent=True)
                    redraw_ui(model)
                    reprint_history(history)
                    sys.stdout.write(f"{RESET}\U0001f9d1 : ")
                    sys.stdout.flush()
                    start = time.time()
                if os.name == "nt":
                    if msvcrt.kbhit():
                        user_input = input().strip()
                        if not user_input:
                            continue
                        break
                    time.sleep(0.1)
                    continue
                else:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        user_input = input().strip()
                        if not user_input:
                            continue
                        break

            cmd = user_input.lower()
            if cmd == "/exit":
                return "exit"
            elif cmd == "/clear":
                redraw_ui(model)
                continue
            elif cmd == "/print":
                if history:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = f"chat_{ts}.txt"
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

            print(f"{AI_COLOR}\U0001f5a5️ : thinking...", end='', flush=True)
            headers = api_headers()
            resp = None
            server_failed = False

            chat_paths = ["/v1/chat/completions"]
            if selected_api == "ollama":
                chat_paths.append("/api/chat")

            for p in chat_paths:
                try:
                    r = requests.post(f"{SERVER_URL}{p}", headers=headers,
                                      json={"model": model, "messages": [{"role": "user", "content": user_input}], "stream": False},
                                      timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    msg = data.get("choices", [{}])[0].get("message", {}).get("content") or data.get("choices", [{}])[0].get("text") or data.get("completion")
                    if msg:
                        resp = msg
                        break
                except Exception as e:
                    print(f"\n{RED}Failed {p}: {e}{RESET}")
                    server_failed = True

            if not resp:
                gen = ["/v1/completions"]
                if selected_api == "ollama":
                    gen.append("/api/generate")
                for p in gen:
                    try:
                        r = requests.post(f"{SERVER_URL}{p}", headers=headers,
                                          json={"model": model, "prompt": user_input, "stream": False}, timeout=60)
                        r.raise_for_status()
                        data = r.json()
                        msg = data.get("choices", [{}])[0].get("text") or data.get("completion")
                        if msg:
                            resp = msg
                            break
                    except Exception as e:
                        print(f"\n{RED}Failed {p}: {e}{RESET}")
                        server_failed = True

            if not resp:
                print(f"{RED}[Error] No response{RESET}")
                server_failed = True

            if server_failed:
                try:
                    df = pd.read_csv(CSV_PATH, keep_default_na=False)
                    df.loc[(df["ip"] == selected_server["ip"]) & (df["port"] == selected_server["apis"][selected_api]), "is_active"] = False
                    df.to_csv(CSV_PATH, index=False)
                    print(f"{RED}Server marked inactive due to failure{RESET}")
                except Exception as ex:
                    print(f"{RED}Failed to update CSV: {ex}{RESET}")
                return "back"

            print("\r\033[K", end='')
            print(f"{AI_COLOR}\U0001f5a5️ : ", end='')
            render_markdown(resp)
            history.append({"user": user_input, "ai": resp})

    except KeyboardInterrupt:
        print(f"{YELLOW}Session interrupted{RESET}")
        if history:
            save = input(f"{CYAN}Save log? (y/n): {RESET}").lower()
            if save == "y":
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"chat_{ts}.txt"
                with open(fn, "w", encoding="utf-8") as f:
                    for h in history:
                        f.write(f"User: {h['user']}\nAI: {h['ai']}\n\n")
                print(f"Saved to {fn}")

# Main loop
if __name__ == "__main__":
    selected_api = 'ollama'

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
        chosen = select_model(models)
        display_connecting_box()
        matrix_rain(duration=10)

        # 4. Chat loop
        result = chat_loop(chosen)
        if result == "back":
            continue  # <-- Go back to server selection, not model
        else:
            break
