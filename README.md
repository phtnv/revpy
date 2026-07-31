# Chat completions proxy
Compatibility adapter that takes OpenAI-style `/chat/completions` requests from a chat frontend and forwards them to whichever API your model actually speaks. Typical usage is as an intermediary server between JanitorAI and a provider you hold the key for.

Three upstream protocols are supported, and you say which one a provider speaks by declaring it in the matching list in `.env` (see `env_example.ini`):

| Variable                        | Endpoint               | Providers                         |
| ------------------------------- | ---------------------- | --------------------------------- |
| `V1_MESSAGES_PROVIDERS`         | `/v1/messages`         | Anthropic (Claude)                |
| `V1_CHAT_COMPLETIONS_PROVIDERS` | `/v1/chat/completions` | GLM, Kimi, Aion Labs, most others |
| `V1_RESPONSES_PROVIDERS`        | `/v1/responses`        | OpenAI                            |

Nothing is guessed from the URL, so a provider goes wherever you put it. Every configured provider's models appear in one CLI `model` list; selecting one switches the active backend.

A separately configured image model can also generate images, either from a block written in a chat message or through the proxy's own `/v1/images/generations` endpoint, without disturbing the conversation — see [Image generation](#image-generation).

Model-agnostic features (summary blocks, lorebook handling, cost tracking, instruction prefill, auto-trim, dumps) work for every backend. Anthropic-protocol features (explicit cache markers, signed thinking-block preservation, assistant prefill) apply only to providers on `/v1/messages`. The shared thinking settings are translated into the provider's own dialect for GPT, Aion, GLM and Kimi models (see [Thinking on OpenAI-style backends](#thinking-on-openai-style-backends)). Any other provider-specific request option is passed through verbatim via `<NAME>_EXTRA_BODY`.

## Requirements
- An API key for at least one provider. For Claude, from Claude Console: https://platform.claude.com/settings/workspaces/default/keys
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
    CLAUDE_API_KEY=your_anthropic_api_key
    PROXY_KEY=key_you_write_in_janitor
    ```
    `CLAUDE_API_KEY` is the key you generated in Claude Console. `PROXY_KEY` is the key you actually put into Janitor as the Proxy key. They can be the same, but why spread your private key around?

    The example config starts on Claude. To use a different provider, uncomment its list and its `<NAME>_*` block, and point `MODEL` at it:
    ```ini
    V1_CHAT_COMPLETIONS_PROVIDERS=glm
    MODEL=glm/glm-4.7
    ```

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
    Configured providers:
      claude      messages    https://api.anthropic.com/v1

    Retrieved 12 model(s) from provider 'claude'.
    === Switching to claude/claude-sonnet-4-6 ===
    Using cost family 'claude:sonnet'.
    === Switching to claude/claude-sonnet-4-6 complete ===
    Starting proxy
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

Well, this is a server-side issue then. You can open an issue here on GitHub. The text in your terminal should provide a hint what's happening, and you can always ask your friendly LLM for help. Give it the error message as-is plus `server.py` and the backend module for the model you were using (see [Code layout](#code-layout)) — even the free LLMs are usually good about finding bugs in simple stuff like this.

A red `Proxy error (not an API response)` line means the fault is in the proxy rather than something the provider refused; the full traceback for those goes to `ERROR_LOG_PATH`.

## Command-Line Interface

The proxy has a small CLI (command-line interface) embedded in it. In the same terminal you launched `server.py`, type `help` to list all the available commands. Type `quit` to quit the server.

## Model selection

At startup the proxy fetches the model list of every configured provider, in the order the provider lists are given above.

In CLI type `m` to see the available model list (along with the currently selected model).

Type `m <number>` to select a specific model from the list. The model you set in Janitor's UI has no effect. Selecting a model from another provider switches the active backend with it, including the endpoint the proxy calls and the prices it bills at.

CLI model selection is runtime-only. To make a model the default after restart, edit `MODEL=` in `.env`. Write it as `provider/model-id`: that form names its own provider, so it still resolves when the provider's model list is unavailable.

If a provider's `/models` request fails, the proxy warns and carries on with the others.

## Caching

Everything in this section is an Anthropic-protocol feature and applies to providers declared in `V1_MESSAGES_PROVIDERS`. For everyone else see [Caching on OpenAI-style backends](#caching-on-openai-style-backends). For a detailed guide how caching works, you can read Anthropic's [official docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

The short version is that Anthropic provides you up to 4 markers to place at any user (that's you), assistant (that's the LLM) or system (that's the bot definitions, advanced prompts and summaries) messages in your chat. Messages up to and including these markers will be cached.

What is the benefit of caching a sequence of messages? When caching, you pay 200%/125% once (cache write), then for 1 hour/5 minutes you pay 10% for the messages that were cached (cache read). When you read a cache it refreshes for free, so if you keep the messages coming every 5 minutes, 5 minute writes are better, if not, 1 hour writes are better.

In practice one read is necessary to make a 5 minute cache write profitable, and two for a 1 hour cache write.

**THE BIG GOTCHA:** Messages are cached sequentially. So, if a message in the middle changes, ALL the messages after it will need to be re-cached. Why is this a gotcha? Well, Claude caches messages in the following order:
```md
system message -> user + assistant message pairs
```

Which in Janitor's case becomes:
```md
Advanced prompt -> Core (scenario->persona->example_dialogs->summary) -> lorebooks/user scripts -> intro -> chat
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

### Caching on OpenAI-style backends

None of the above applies to providers on `/chat/completions` or `/responses`: they cache automatically, with no markers to place and no TTL to choose, so the `c` commands do nothing while one of them is selected. OpenAI caches prompts from 1024 tokens up, and bills a cache hit at the `cache_read` rate. From `gpt-5.6` on it also charges for the write, at 1.25x the input rate, which the proxy reports in the usual cache-cost line — configure it per model with `cache_write` in `<NAME>_MODEL_<FAMILY>_COST`. Providers that write for free simply leave `cache_write` at the input price, which nets those tokens to zero.

Since the cache still keys on a stable prefix, the big gotcha above is unchanged: a lorebook that shuffles its entries invalidates everything after it.

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

Whether a model can think at all is read from the provider's model record. When that record says nothing about it — which is what you get for a model the proxy could not fetch, or a provider that does not report capabilities — thinking is reported as unsupported and switched off at model selection.

### Thinking on OpenAI-style backends

The same `t 0` / `t 1` / `t effort <level>` commands drive the OpenAI-style providers, but every provider spells thinking differently, so the proxy translates the shared settings into that provider's dialect. What actually reaches the API depends on the selected model — `t` prints the current mapping, and so does switching models.

| Model                                   | On/off                         | Effort                                                                  |
| --------------------------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| `gpt-5.6-sol\|terra\|luna`              | yes (effort `none`)            | `none\|low\|medium\|high\|xhigh\|max`                                   |
| `gpt-5.2` … `gpt-5.5`                   | yes (effort `none`)            | `none\|low\|medium\|high\|xhigh`. `max` maps to `xhigh`.                |
| `gpt-5.1`                               | yes (effort `none`)            | `none\|low\|medium\|high`. `xhigh`/`max` map to `high`.                 |
| `gpt-5`                                 | no, `minimal` is the floor     | `minimal\|low\|medium\|high`. `xhigh`/`max` map to `high`.              |
| `o1`, `o3`, `o4-mini`                   | no, always thinks              | `low\|medium\|high`. `xhigh`/`max` map to `high`.                       |
| `*-chat-latest`                         | does not think at all          | —                                                                       |
| `gpt-4.1`, `gpt-4o` and older           | does not think at all          | — these reject the parameter outright                                   |
| `aion-2.0`                              | yes (`reasoning_effort: none`) | `none\|low\|medium\|high`, default medium. `xhigh`/`max` map to `high`. |
| `aion-2.5`, `aion-3.0`, `aion-3.0-mini` | no, always thinks              | none — these reject `reasoning_effort` with an HTTP 400                 |
| `aion-rp-*`                             | does not think at all          | —                                                                       |
| `glm-*`                                 | yes                            | `reasoning_effort` from glm-5.2 on                                      |
| `kimi-k3`                               | no, always thinks              | `low\|high\|max`. `medium` maps down to `low`, `xhigh` up to `max`.     |
| `kimi-k2.7-*`                           | no, always thinks              | none                                                                    |
| `kimi-k2.5`, `kimi-k2.6`                | yes                            | none                                                                    |

An effort above what a model offers folds down to its highest, so `max` becomes `xhigh` almost everywhere. When a model cannot be told to stop reasoning, `t 0` sends the weakest level it does offer instead, and the CLI says so.

Models with no dialect (DeepSeek, ...) ignore the CLI thinking settings entirely; use `<NAME>_EXTRA_BODY` for those. `EXTRA_BODY` is merged after the dialect, so it also overrides it if you want to force a specific parameter.

Thinking preservation is an Anthropic-protocol feature. On these backends the thoughts are simply wrapped in a `<think>` block for Janitor, and are not sent back.

On every backend the thinking is billed as output whether or not you can read it, so the usage report splits the output tokens into reasoning and visible text with a cost for each:

```log
    Output tokens      =  reasoning +    visible
                  3019 =       2588 +        431
             $0.018114 =  $0.015528 +  $0.002586
```

OpenAI, GLM and Kimi report that count. Anthropic and Aion do not, and rather than print a zero for thinking that demonstrably happened, those backends keep the plain `Output tokens = N` line.

Because reasoning tokens also count against the output limit, a small `max_tokens` can be consumed entirely by thinking and leave an empty message; the proxy warns when it sees that, but the fix is to raise `max_tokens` in Janitor or lower the effort.

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

By default, the summary is inserted as an `assistant` chat message. You can also set `role="user"` on any end tag, or set `role="system"` on the special `all` tag:
```xml
<summary_block_end tag="all" role="system">
I was born at a very young age...
</summary_block_end>
```
`role="system"` is only valid with `tag="all"`. It inserts the summary into the system fields (after the core definitions + lorebook) instead of the message.

The proxy should print a warning if you made mistakes with your tags somewhere (forgot to close one, mistyped the tag, etc...).

## Image generation

The proxy can generate images through a separately configured image model, while your chat model carries on as normal. Selecting or invoking an image model never changes `MODEL`, the active backend, or the prices your conversation is billed at — it is a secondary service, not a fourth protocol.

Turn it on in `.env`:

```ini
IMAGE_GENERATION_ENABLED=true
IMAGE_PROVIDER=gpt
IMAGE_MODEL=gpt-image-2
IMAGE_OUTPUT_DIR=generated_images
```

`IMAGE_PROVIDER` normally names one of the providers you already declared, whose base URL and key are reused. It can also name one that appears in **no** list — its `<NAME>_BASE_URL` and `<NAME>_API_KEY` are then read directly. That is how you generate images through OpenAI while chatting with Claude, without OpenAI's text models cluttering the `model` list.

### Asking for an image in chat

Write a block in any **user** message. The proxy removes it before the conversation reaches the text model, so the model never sees the control syntax:

```xml
<image_generation>
prompt  : "Cassia seated at a small dinner table in a candlelit stone chamber",
quality : "high"
</image_generation>
```

Only `prompt` is required; everything else falls back to your configured defaults. The syntax is json5 without the surrounding braces, and a block with no `field:` syntax at all is taken as a bare prompt — `<image_generation>a gray tabby cat</image_generation>` works.

The reply comes back with a reference appended:

```text
[Generated image: img_8f281a — generated_images/20260730_105900_8f281a.png]
```

If the message contained *nothing but* the block, no request is made to the text model at all — you get the reference on its own.

Two things worth knowing:

- **Only the latest user message triggers.** Chat clients resend the whole history every turn, so a block written three turns ago arrives again with every request. Old blocks are still stripped (the model never sees them) but never re-run — otherwise every turn would regenerate every image the chat has ever asked for.
- **The assistant cannot request images.** Only user messages are scanned, so the model cannot spend your money by writing about the tag.

A failed generation never costs you the turn. The model's prose still arrives, with the reason inlined:

```text
[Image generation failed: content policy violation]
```

### Overridable fields

`prompt`, `size`, `quality`, `output_format`, `background`, `n`, `batch`, `filename`.

Anything else — model, provider, base URL, API key, headers, output path — is proxy policy and is rejected rather than ignored, so a typo is reported instead of silently taking a default. `filename` is a *name*, never a path: separators, traversal and unusual characters are refused, the extension comes from `output_format`, and an existing file is never overwritten (a linear index is appended instead).

`size` is validated against the model's constraints rather than a list: edges must be multiples of 16 and at most 3840px, the aspect ratio at most 3:1, and the total between 655,360 and 8,294,400 pixels. `auto` is accepted. Note that `background: transparent` is **not** supported by `gpt-image-2` — only `opaque` and `automatic`.

### The direct endpoint

```text
POST /v1/images/generations
{"prompt": "A gray tabby cat hugging an otter"}
```

Every omitted field is filled from configuration. The reply is metadata, not Base64 — the images are already on disk:

```json
{
  "created": 1785400000,
  "model": "gpt-image-2",
  "data": [{"id": "img_8f281a", "path": "generated_images/20260730_105900_8f281a.png"}],
  "usage": {"estimated_cost_usd": 0.005945, "cost_is_estimate": false,
            "text_input_tokens": 13, "image_output_tokens": 196}
}
```

Each image also appends a record to `img_generation.json` in the output directory (`IMAGE_MANIFEST_ENABLED`; `IMAGE_MANIFEST_PROMPTS` decides whether the prompt itself is kept, since the manifest outlives the session).

### Batch API

Setting `batch: true` submits through the provider's Batch API instead of generating immediately. A batch is **not** the same thing as `n: 4` — `n` asks for several images now, a batch trades latency (up to `IMAGE_BATCH_COMPLETION_WINDOW`, default 24h) for a lower rate. It therefore cannot return a file inside a chat turn.

You don't have to chase it. The proxy polls unsettled batches in the background every `IMAGE_BATCH_POLL_SECONDS` (default 300, floored at 30) and saves their images the moment they finish, announcing it in the terminal:

```text
=== image batch #3 completed ===
  saved img_9deea2c9a2304746 -> generated_images/20260731_125500_9deea2c9.png
  cost $0.002968
=== image batch #3 end ===
```

Set `IMAGE_BATCH_AUTO_POLL=false` to turn that off and retrieve them by hand instead.

**Batches are referred to by a small number, not the provider's hash.** The number is assigned at submission, persisted, and never reused, so the one printed when you submitted still names the same batch after a restart:

```text
image batch list       →   1  completed   gpt-image-2  submitted 2026-07-31T12:52:32+0300  1 image(s)
                           2  pending     gpt-image-2  submitted 2026-07-31T13:15:15+0300
image batch get 2          Retrieve #2 now.
image batch poll           Check everything unsettled now, instead of waiting out the interval.
```

`GET /v1/images/batches/2` works the same way, and both forms still accept a raw provider id so a batch this proxy never submitted can be fetched.

The parameters each batch was submitted with are kept in `img_batches.json` in the output directory, because the provider returns only the images and the id it was given. Retrieving a completed batch saves its images once and records that it did, so neither a second retrieval nor the poller racing you can write a duplicate or bill you twice. A batch that failed, expired or was cancelled is settled too, so the poller stops asking about it.

A batch reports the same token counts as an immediate request, so nothing in the response reveals the discount — set `IMAGE_BATCH_COST_MULTIPLIER` to state it. It defaults to `1.0` (no discount assumed, so a budget is never quietly understated); OpenAI bills the Batch API at 50%, so `0.5` is right for that provider.

### CLI

```text
image                  Show image status.
image gen <prompt>     Generate now, using the configured defaults.
image model            List the image provider's image models.
image model <number>   Select one. Your chat model is untouched.
image batch <prompt>   Submit a one-request batch.
image batch get <n>    Retrieve batch number <n> and save its images.
image batch list       List batches submitted from this output directory.
image batch poll       Check every unsettled batch now.
image cost             Show image spending.
```

### Cost

Image spending is tracked separately from text spending and reported apart, because an image is not an output token and folding it into the text totals would make the per-token averages meaningless:

```text
Text model cost:      $0.412300
Image immediate cost: $0.005945
Image batch cost:     $0.000000
Combined session:     $0.418245
```

Prices are configuration-driven, in USD per million tokens, under their own `<PREFIX>_IMAGE_MODEL_*` namespace so they never collide with the text families:

```ini
GPT_IMAGE_MODEL_GPTIMAGE2_REGEX=^gpt-image-2
GPT_IMAGE_MODEL_GPTIMAGE2_COST={text_input: 5.00, image_input: 8.00, image_output: 30.00}
```

The gpt-image family returns exact token counts, so this is real accounting rather than a guess — a low-quality 1024×1024 image reports 196 output tokens and bills at $0.0059, a high-quality one lands around $0.16–$0.21. `IMAGE_PRICE_TABLE` supplies fallback per-image estimates and is consulted only when a provider returns no usage object at all; anything costed that way is flagged as an estimate.

## Code layout

The backend boundary is the upstream wire protocol — one module per protocol, each exposing the same five entry points:

| Module                   | Speaks                              | Declared in                     |
| ------------------------ | ----------------------------------- | ------------------------------- |
| `v1_messages.py`         | Anthropic `/v1/messages`            | `V1_MESSAGES_PROVIDERS`         |
| `v1_chat_completions.py` | OpenAI-style `/v1/chat/completions` | `V1_CHAT_COMPLETIONS_PROVIDERS` |
| `v1_responses.py`        | OpenAI `/v1/responses`              | `V1_RESPONSES_PROVIDERS`        |

Which module serves a provider is the list you declared it in, and nothing else. Selecting one of its models is what binds the backend.

Around them:

- **`server.py`** — the HTTP endpoint Janitor talks to, the runtime CLI, and every model-agnostic transform (summary blocks, lorebook handling, chat snapshots). It picks the backend module per request; the Anthropic-protocol features are the only place it asks which one it got.
- **`providers.py`** — the registry every backend shares: provider config, the aggregated model list, model selection and pricing, HTTP transport, and the request message list the two OpenAI-style modules build from. It imports none of the backends, so a provider can never depend on the endpoint that happens to be selected.
- **`common.py`** — configuration from `.env`, cost accounting, text helpers, error rendering.

Image generation sits beside them rather than among them, since it is a secondary service the chat backends never call:

- **`v1_images_generation.py`** — the `/images/generations` adapter: provider and price resolution, validation, transport, Base64 decoding, atomic file saving, the manifest, and the Batch API. It never writes to `cfg.backend`, `cfg.model` or the text price fields.
- **`image_orchestrator.py`** — finds, parses and strips `<image_generation>` blocks in user messages, and decides whether the turn still needs the text model. It decides *whether*; `v1_images_generation.py` decides *how*.

Both the direct endpoint and the chat trigger build the same `ImageRequest` and hand it to the same module, so validation, storage and accounting exist exactly once.

Tests live in `tests/` and run with `python tests/test_driver.py`. They need no network and no API key: the Anthropic client is faked, the provider tests assert the exact request body each backend builds, and the image tests cover extraction, validation, filename confinement and cost without contacting anything.

## If you've read this far...

Check out my [bot](https://janitorai.com/profiles/c71e4e8f-c4bc-478a-9cb7-6f90cdb5cb16_profile-of-narrava)!
