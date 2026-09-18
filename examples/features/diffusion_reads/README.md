# Structured reads on DiffusionGemma

A discrete diffusion model denoises a whole canvas per forward pass. If the
canvas is seeded with the answer's fixed text and only the answer slots are
left as noise, one denoise step yields a distribution over each slot. A few `extra_args` fields (`vllm_xargs` on the OpenAI server) expose
that:

| field | type | meaning |
|---|---|---|
| `diffusion_seed_canvas` | `list[int]`, exactly `canvas_length` ids | replaces the random initial canvas after prefill |
| `diffusion_slot_positions` | `list[int]`, positions into the canvas | the positions the seed leaves free; every other one is written back to its seed token on every step |
| `diffusion_max_steps` | `int` | denoise steps before the canvas is emitted |
| `diffusion_read_only` | `bool` | emit the argmax canvas as soon as the cap is reached, end the request there, and return temperature-1 logprobs at every position |

The sampler re-noises every position it does not accept, and it noises with
ordinary vocabulary tokens rather than a mask token, so a seeded template
decays from the second step on. `diffusion_slot_positions` is what lets a
read run more than one step over the template it was given: the held
positions keep their seed token, count as accepted, and self-condition on
that token instead of on the model's own guess.

`structured_server.py` is the layer that turns a question schema into those
fields. It speaks `/v1/chat/completions`: the system message is the schema,
the user message is the state JSON, and the reply content is one
distribution per question with a standard error over a few noise draws.

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
    --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 --enable-prefix-caching
python examples/features/diffusion_reads/structured_server.py \
    --upstream http://127.0.0.1:8000 --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64
curl -s localhost:8011/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "{\"questions\": [{\"id\": \"urgent\", \"type\": \"noul\", \"instructions\": \"Does the customer need a reply within the hour?\"}]}"},
    {"role": "user", "content": "{\"ticket\": \"Everything is down and we have a demo at noon.\"}"}
  ]}'
```

Question types: `noul` (yes/no), `choice` with `options`, `score` with
ordered `levels`. Each label must be a single token in the answer template,
which the server checks with the tokenizer before the first request.

`"think": N` in the schema lets the model write up to N tokens in its
thought channel before the read. The thought is an ordinary generation with
the chat template's thinking marker on, and the read then runs with the
thought in its prompt, so the answer slots condition on it. One thought
serves every noise draw of a decision. `diagnostics.thought` returns the
text, its length in tokens, whether the model closed the channel itself and
the generation time.
