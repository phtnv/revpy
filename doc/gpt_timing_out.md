# OpenAI `/responses` streams cut off during long reasoning

Investigation notes from 2026-07-28. Written so it can be picked up cold, either to
implement background mode or to re-check whether OpenAI has fixed anything.

## Symptom

Streaming a large prompt to `gpt-5.6-sol` at high thinking effort: the client receives
the start of the reasoning stream and then nothing — no further messages, no error. On a
fresh chat the same request appears to hang and return nothing at all. The assistant
message ends up empty (see the last message in `bugged_chat.json`, kept as the sample).

## What is actually happening

OpenAI accepts the request (HTTP 200), sends the lifecycle events, then **closes the
connection mid-response** with no terminal event and no error. Raw SSE capture, unparsed:

```
HTTP 200
[  1.0s] response.created      status='in_progress'
[  1.0s] response.in_progress  status='in_progress'
[  1.7s] response.output_item.added
[ 27.5s] response.output_item.done
--- body ends at 56s, status still 'in_progress' ---
```

No `response.completed`, no `response.incomplete`, no `error` event, and zero
`response.output_text.delta` / `response.reasoning_summary_text.delta` events.

The proxy was not losing events — the provider stopped sending them.

## Measurements

Request replayed from `bugged_chat.json`: 4 messages, 81049 chars (~20k tokens), model
`gpt-5.6-sol`. One variable changed at a time.

| Variable                        | Result                                             |
| ------------------------------- | -------------------------------------------------- |
| `max_output_tokens` 128000      | fails at 56s                                        |
| `max_output_tokens` 16000       | fails at 25s                                        |
| `store: true`                   | fails at 12.2s                                      |
| `summary` removed               | fails at 12.6s                                      |
| `effort: high`                  | fails — 6.4s / 12.2s / 12.6s / 25s / 38s / 56s      |
| `effort: medium`                | fails at 44.4s                                      |
| **`effort: low`**               | **succeeds** — 11161 output tokens over 244s        |
| short prompt, `effort: high`    | succeeds in 2.7s                                    |

Two conclusions:

- **It is not a time limit.** The successful low-effort run lasted 244s, far longer than
  any failure. Failures happen *while the model is reasoning silently*, before output.
- **It is not `max_output_tokens`, `store`, or `summary`.** Only effort and prompt size
  matter. The successful low-effort run used just 129 reasoning tokens — barely any
  silent phase — then streamed 11k tokens, which keeps the connection busy.

Mechanism: `/responses` sends **nothing at all** while reasoning. Summary text only
arrives once the reasoning item closes. So a long reasoning phase is a long silent
connection, and something closes it.

## Already fixed in this repo (do not redo)

The proxy used to relay a cut stream as a *successful empty reply*:

```
HTTP 200  finish_reason='stop'  text=0 chars  usage: all zeros
```

That is why the failure was silent — a truncated stream was indistinguishable from the
model choosing to say nothing. Fixed in `v1_responses.py`:

- `TERMINAL_STATUSES` + a `stream_completed` flag in `generate_stream()`. Reaching the
  end of the body without a terminal status now raises `ProviderError`, so the client
  gets an error chunk and the console gets a red line.
- `truncated_stream_message()` explains the failure and suggests `t effort low`.
- `generate_non_stream()` raises on `status: "failed"`, which also arrives as HTTP 200.

Regression tests: `STREAM_CASES` in `tests/test_driver.py`, using `FakeStreamClient` to
replay canned SSE bodies. No network needed. Mutation-tested: disabling the guard makes
three of them fail with exactly the original symptom.

**This does not make the request work.** It only stops the proxy from lying about it.

## This is a known upstream problem

- [Gpt-5 with reasoning set to high is timing out](https://community.openai.com/t/gpt-5-with-reasoning-set-to-high-is-timing-out/1362385)
  — ~95% failure at high effort, medium and below fine, open ~a month, no OpenAI staff
  reply. Notes the model "can run for a long time without a response or any network
  activity". Also names large context inputs as an aggravating factor.
- [background-agents #681](https://github.com/ColeMurray/background-agents/issues/681) —
  `gpt-5.5` + `xhigh` "thinks" ~10 min then fails; notes the Responses API "does not
  always send a TCP FIN after terminal stream events".
- [Peer connection closed](https://community.openai.com/t/getting-peer-connection-closed-for-requests-exceeding-1-hour/1370670)
  — `gpt-5.2` xhigh, "peer closed connection without sending complete message body".
- [Azure gpt-5.2-codex](https://learn.microsoft.com/en-us/answers/questions/5739406/azure-openai-gpt-5-2-codex-intermittent-internal-e)
  — intermittent failures 30–50% with Responses + reasoning.
- [Codex CLI](https://community.openai.com/t/stream-disconnected-before-completion-stream-closed-before-response-completed/1342139)
  — identical error string, but the causes there were middleboxes (Zscaler, VPNs). Not
  our case: the repro went straight from httpx to `api.openai.com`.
- [OmniRoute #7285](https://github.com/diegosouzapw/OmniRoute/issues/7285) — the same
  proxy-side defect, independently: "the truncated stream is forwarded to the client as
  a success". Same fix shape as ours.

Difference from those reports: they say medium is safe; here medium failed too. The ~20k
prompt is the likely reason.

[OpenAI's GPT-5 troubleshooting cookbook](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_troubleshooting_guide)
does not mention connection drops, stream resilience, or missing `response.completed`.

## Option A — background mode (the documented fix)

[Background mode](https://developers.openai.com/api/docs/guides/background) exists for
exactly this: "long-running tasks... without having to worry about timeouts or other
connectivity issues." With `background: true` + `stream: true`, every event carries a
`sequence_number`, and a dropped connection is resumed with `starting_after`.

### Tradeoffs to decide first

- **Retention.** Background responses are written to disk for ~10 minutes to support
  polling — *even for Zero Data Retention projects with `store=false`*. This proxy sets
  `store=false` deliberately for privacy (`<NAME>_STORE`), so background mode walks that
  back. This is the main reason it was not just switched on.
- Unlike a naive retry, resuming does **not** re-bill the reasoning: it is the same
  response object.
- OpenAI's docs said "SDK support for resuming the stream is coming soon", so the resume
  loop is manual. The proxy uses raw httpx anyway, so this costs nothing extra.

### Implementation sketch, in this codebase

1. **Config** (`common.py`): add `<NAME>_BACKGROUND` (default false). This *is* a real
   choice — unlike the `<NAME>_API` knob that was removed, both answers are defensible
   because of the retention tradeoff. Document it next to `<NAME>_STORE`.
2. **Request** (`v1_responses.build_body`): set `"background": True` when enabled.
3. **Streaming** (`v1_responses.generate_stream`):
   - Capture the response id from `response.created` (already partly done — `message_id`).
   - Track the highest `sequence_number` seen.
   - The existing `stream_completed` guard is the natural hook: instead of raising
     immediately, reconnect with `GET {base_url}/responses/{id}?stream=true&starting_after=N`
     and keep consuming. Cap the attempts (3?) and raise the existing truncated-stream
     error when they run out, so the failure path stays what it is today.
   - Do not re-yield events at or below `starting_after`, or the client sees duplicates.
4. **Non-streaming** (`v1_responses.generate_non_stream`): a background response returns
   `status: "queued"` immediately, so it needs a poll loop on `GET /responses/{id}` until
   a terminal status, bounded by `OPENAI_REQUEST_TIMEOUT_SECONDS`.
5. **Docs**: README "Seeing GPT's reasoning" section and `env_example.ini`, including the
   ~10 minute retention caveat — it contradicts the privacy stance stated there now.
6. **Tests**: extend `STREAM_CASES`. `FakeStreamClient` already replays canned bodies;
   add a case where the first body truncates and the resume body completes, asserting no
   duplicate deltas.

Known unknown: whether a response cut *this* way is actually resumable, or whether the
connection drop also kills the response server-side. Worth a single manual probe before
building anything — reproduce a cut, grab the id, and try
`GET /responses/{id}?stream=true&starting_after=0`.

## Option B — bounded retry on truncation

Retry the whole request when the truncation guard fires. Simple, but re-bills the
reasoning (potentially expensive at high effort) and may well hit the same wall, since
the failures cluster in the silent reasoning phase. Not recommended over Option A.

## Option C — leave as is

Current state. The failure is clearly reported with an actionable message, and
`t effort low` gets through. Costs nothing, fixes nothing.

## Reproducing

`bugged_chat.json` (repo root) is the failing conversation. To rebuild the request:
system prompt, then the `files[]` entries folded into the first user message
(`filePlacement: firstUser`), then the chat messages; `max_tokens` from
`settings.maxTokens`. Select `gpt/gpt-5.6-sol`, thinking on, effort high, and stream.
Capture raw SSE rather than parsed output — the distinction between "provider stopped"
and "parser dropped events" is the whole diagnosis.
