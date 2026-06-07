# OpenAI to Anthropic reverse proxy
Reverse proxy for interfacing OpenAI-compatible chat UIs with Claude.

## Requirements
- Anthropic API key from Claude Console. https://platform.claude.com/settings/workspaces/default/keys
- [Python](https://www.python.org/downloads/).
    - This guide assumes it's in your path. On Windows, this means you've checked the `Add python.exe to PATH` when installing.
- [cloudflared](https://developers.cloudflare.com/tunnel/downloads/)

## Setup
0. Clone or download this repository. Open the repo folder in the terminal.
    - On Windows, if you're unsure what I mean, install [Windows Terminal](https://apps.microsoft.com/detail/9n0dx20hk701?hl=en-us&gl=EN).

1. Create the python environment file.
    ```bash
    cp env_example.ini .env
    ```

2. In the .env file, edit at least these two lines
    ```ini
    ANTHROPIC_API_KEY=your_anthropic_api_key
    PROXY_KEY=key_you_write_in_janitor
    ```
    `ANTHROPIC_API_KEY` is the key you generated in Claude Console. `PROXY_KEY` is the key you actually put into Janitor as the Proxy key. They can be the same, but why spread your private key around?

3. Setup the environment.
    ```bash
    python -m venv .venv
    ```

4. Load the environment.

    For Windows:
    ```ps
    ./.venv/Scripts/Activate.ps1
    ```

    For Linux
    ```bash
    source .venv/bin/activate
    ```

    If you get an error in Windows run
    ```ps1
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    ```
    And re-run the activate command.

5. Install Python requirements
    ```bash
    pip install -r requirements.txt
    ```
    If you get an pip not recognized error you can try
    ```bash
    py -m pip install -r requirements.txt
    ```

6. Start the server
    ```bash
    python server.py
    ```
    Expected output
    ```log
    Starting Claude reverse proxy
    Local URL: http://127.0.0.1:5001
    Chat completions: http://127.0.0.1:5001/chat/completions
    Cloudflare Tunnel service URL should point to this local address:
      http://127.0.0.1:5001
    ```

7. Get cloudflared. Just download the standalone executable from their [site](https://developers.cloudflare.com/tunnel/downloads/). If you have to ask - `amd64 / x86-64` on Linux, or `64-bit` on Windows. Open a new terminal window, go to where you downloaded cloudflared, and run:
    ```bash
    cloudflared tunnel --url http://127.0.0.1:5001
    ```
    You should see output like:
    ```log
    2026-05-27T08:45:21Z INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, is a quick way to experiment and try it out. However, be aware that these account-less Tunnels have no uptime guarantee, are subject to the Cloudflare Online Services Terms of Use (https://www.cloudflare.com/website-terms/), and Cloudflare reserves the right to investigate your use of Tunnels for violations of such terms. If you intend to use Tunnels in production you should use a pre-created named tunnel by following: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps
    2026-05-27T08:45:21Z INF Requesting new quick Tunnel on trycloudflare.com...
    2026-05-27T08:45:25Z INF +--------------------------------------------------------------------------------------------+
    2026-05-27T08:45:25Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
    2026-05-27T08:45:25Z INF |  https://convenience-forum-transition-shame.trycloudflare.com                              |
    ```
    The link at the very bottom is the link you put into Janitor as the proxy link.

    In this case it was `https://convenience-forum-transition-shame.trycloudflare.com`.

    Then the link you paste in Janitor will be `https://convenience-forum-transition-shame.trycloudflare.com/chat/completions`.

    This link will change every time you re-launch cloudflared.

After the first setup, you usually only need to:
1. Open the repo folder.
2. Activate `.venv` if this is a new terminal (step 4).
3. Start `server.py` (step 6).
4. Start `cloudflared` (step 7).

### Troubleshooting
> Janitor gives me an error - nothing appears in the server console!

This is 99% a Janitor-side issue, or you've typed the link wrong in its UI.

> Janitor gives an error, some text appeared in the server console.

Well, this is a server-side issue then. You can open an issue here on GitHub, but you can also ask in the Janitor Discord Claude thread, which I read. The text in your terminal should provide a hint what's happening, and you can always ask your friendly LLM for help. Give it the error message as-is + the `server.py` file, even the free LLMs are usually good about finding bugs in simple stuff like this.

## Command-Line Interface

The proxy has a small CLI (command-line interface) embedded in it. In the same terminal you launched `server.py`, type `help` to list all the available commands. Type `quit` to quit the server.

## Model selection

At startup the proxy fetches the available model list from Anthropic.

In CLI type `m` to see the available model list (along with the currently selected model).

Type `m <number>` to select a specific model from the list. The model you set in Janitor's UI has no effect.

CLI model selection is runtime-only. To make a model the default after restart, edit `MODEL=` in `.env`.

## Caching

The proxy implements prompt caching. For a detailed guide how caching works, you can read Anthropic's [official docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

The short version is that Anthropic provides you up to 4 markers to place at any user (that's you), assistant (that's the LLM) or system (that's the bot definitions, advanced prompts and summaries) messages in your chat. Messages up to and including these markers will be cached.

What is the benefit of caching a sequence of messages? When caching, you pay 200%/125% once (cache write), then for 1 hour/5 minutes you pay 10% for the messages that were cached (cache read). When you read a cache it refreshes for free, so if you keep the messages coming every 5 minutes, 5 minute writes are better, if not, 1 hour writes are better.

In practice one read is necessary to make a 5 minute cache write profitable, and two for a 1 hour cache write.

**THE BIG GOTCHA:** Messages are cached sequentially. So, if a message in the middle changes, ALL the messages after it will need to be re-cached. Why is this a gotcha? Well, Claude caches messages in the following order:
```md
system message -> user + assistant message pairs
```

Which in Janitor's case becomes:
```md
Core definitions (persona->scenario->voice samples) -> lorebooks/user script additions, summary -> intro message -> chat
```
So if the bot uses lorebooks heavily, and these lorebooks often insert/remove entries, then every time they do that your entire chat will need to be re-cached, breaking the entire point of caching.

In practice, for bots that do not use lorebooks at all, you can drop your average input token cost by a factor of 6 for long chats. For lorebook-heavy bots, the gains will be much more modest, but I generally have never been in the red.

In any case, for every message the proxy will display both the session and the current request cost, including whether caching provided a gain or a loss, so you can keep track of it yourself.

In CLI, use `c 0` to disable caching globally and `c 1` to enable caching globally again. Individual markers still depend on their own settings.

### The four markers

The proxy can place up to four markers in your chat, controlled from CLI and variables in `.env`.

#### 1 auto marker
```ini
CACHE_AUTO_TTL=1h
CACHE_AUTO_MSG=3
```
> c a 1h

> c a 3

Marker placed at `CACHE_AUTO_MSG` messages from end. 1 will place it at the last user message. 0 to disable.

#### 1 manual marker
```ini
CACHE_MANUAL_TTL=1h
CACHE_MANUAL_MSG=1
```
> c m 1h

> c m 1

Marker placed at `CACHE_MANUAL_MSG` from start. 1 will place it at the intro message. 0 to disable.

#### 2 system markers
```ini
CACHE_SYSTEM=true
CACHE_SYSTEM_TTL=1h
```
> c s 1

> c s 1h

For reasons described above, the system message is split into the core definition + lorebook additions, each with individual cache markers. That way the core definition never has to be re-cached. This split only affects caching, it does not affect how Claude parses these sections.

For lorebook-heavy bots, you can consider disabling all except the system cache markers.

## Thinking (reasoning, `<think></think>` blocks in Janitor)

Thinking can be enabled/disabled from the CLI. Some parameters (prefill, temperature) are incompatible with thinking, they are automatically disabled when thinking is enabled.

Disable thinking
> t 0

Enable thinking
> t 1

Older models (<4.6) use budget for thinking, which should not exceed your max tokens
> t budget <budget_tokens>

Newer models use effort. Some of them support a budget too, but this proxy defaults to using the effort parameter for all models with version >=4.6.
> t effort <low|medium|high|xhigh|max>

The proxy will automatically select the appropriate parameter based on the model selected from CLI.

Thinking will contribute to output tokens (increasing cost), but the thoughts are generally not preserved (unless you enable `PRESERVE_THINKING_BLOCKS` in this proxy), so only the final output will be a part of the input tokens for the next message.

### Thinking preservation

Anthropic *does* allow you to preserve thinking blocks and send them back to Claude. Not as the raw `<think></think>` text blocks, but as special signed and encoded blocks from Anthropic. Under the hood they're the same text you see in `<think></think>`, but signed by Anthropic (why, yes, this is indeed what the model was thinking, officer). Janitor does not save these blocks.

This proxy can preserve these blocks if `PRESERVE_THINKING_BLOCKS > 0`. In that case, these special blocks will be appended to the end of the assistant message with the `~~~` prefix (making them invisible unless you're editing the message). When the message is re-sent, the proxy extracts these preserved blocks from the message and sends them to Anthropic in the appropriate fields.

While I have implemented this to test a specific feature for my bots, I have generally not found it to be useful. But it's there if you want to experiment with it.

Note that because your chats will have these invisible thinking blocks embedded in them, this will technically make them incompatible with any other proxy except this one. Unless you manually go over your chat and remove all the preserved blocks yourself.

The number in `PRESERVE_THINKING_BLOCKS` controls how many assistant messages from end will have their thoughts preserved. `inf` is accepted meaning all the assistant messages. Naturally, using this feature makes thinking contribute to input tokens.

## Summaries

The proxy supports summarizing and replacing arbitrary messages!

In any **assistant** message add the following text at the end:
```xml
<summary_block_beg tag="arbitrary_unique_tag">
```

Then in some **assistant** message after it add:
```xml
<summary_block_end tag="arbitrary_unique_tag">
Intense handholding.
</summary_block_end>
```

All the messages between and including the two tags will be replaced by the summary you provided in the end block.

Since you can't place the tag at the intro message, a special reserved `all` tag is used to summarize all messages starting from the intro.
```xml
<summary_block_end tag="all">
I was born at a very young age...
</summary_block_end>
```
In that case a `summary_block_beg` block is not necessary.

The proxy should print a warning if you made mistakes with your tags somewhere (forgot to close one, mistyped the tag, etc...).

While these tags can be placed in user messages too, and it will work, it is generally not recommended to do so unless you know what you're doing. Claude expects alternating user+assistant pairs, and will merge several user/assistant messages into one if they arrive sequentially. This can cause the messages to be parsed not exactly as you expected.

## If you've read this far...

Check out my [bot](bot_link_will_go_here)!
