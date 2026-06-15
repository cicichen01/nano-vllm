# Why all these versions have to match (explained like you're 5)

Imagine you're building a **LEGO castle** with friends. For the castle to stand,
everybody's bricks have to *snap together*. In an LLM environment, the "bricks"
are torch, CUDA, nvcc, cuDNN, and flash-attn. Here's who each one is.

---

## The characters

🧱 **The GPU (H100)** — the giant LEGO table.
It's the actual machine that does the fast math. Everything else exists just to
talk to it. It speaks one language: **CUDA**.

🗣️ **CUDA** — the *language* the table understands.
CUDA has versions like **12.8** or **13.0**, the way a language has dialects.
If your bricks speak dialect 12.8 but you try them on a table expecting 13.0,
they mostly still work (close dialects), but it's safest if everyone speaks the
**same dialect**. On this machine we picked **CUDA 12.8** for everyone.

🔨 **nvcc** — the *brick-maker* (the CUDA compiler).
When some bricks don't come pre-made, you have to *make* them yourself. `nvcc`
is the tool that turns "instructions" (source code) into a real CUDA brick.
**`nvcc` must speak the same CUDA dialect as everything else** — our `nvcc` is
**12.8**, matching CUDA 12.8. ✅

🤖 **torch (PyTorch)** — the instruction booklet + most of the bricks.
This is the big box that already contains thousands of ready-made bricks for
doing AI math. But each torch box is **built for one CUDA dialect**. Ours is
`torch 2.7.0+cu128` — the `+cu128` part literally means *"these bricks speak
CUDA 12.8."* That's why it has to match nvcc/CUDA.

📚 **cuDNN** — a special bag of *extra-fancy bricks* for neural nets.
torch *expects* this bag to be in the room, even if our castle doesn't use those
fancy bricks. If the bag is missing, torch won't even open
(`libcudnn.so.9 not found`). So we have to put a **cuDNN 9.x** bag in the room.

⚡ **flash-attn** — a *super-fast special brick* someone invented for attention
(the heart of an LLM). It makes the model run much faster. But here's the catch:

> flash-attn does **not** come pre-made for our exact box.
> We have to **make it ourselves with nvcc**, and it snaps onto **torch**.

That's why flash-attn is the fussiest brick of all.

---

## The golden rule

> 🟰 **torch's CUDA, nvcc's CUDA, and cuDNN must all be the same CUDA family,
> and flash-attn must be built against that exact torch.**

On this machine that chain is:

```
CUDA 12.8  ──►  nvcc 12.8  ──►  builds flash-attn  ──►  snaps onto  torch 2.7.0+cu128
                                                                          ▲
                                                          cuDNN 9.x sits next to torch
```

Everybody speaks **12.8**. Everybody is happy. 🎉

---

## What goes wrong if they DON'T match

| Mismatch | What the LEGO castle does |
|----------|---------------------------|
| torch is `cu128` but nvcc is `13.0` | The brick-maker makes bricks in the wrong dialect → flash-attn may build but crash or refuse to load. |
| flash-attn built against torch 2.7, but you swap in torch 2.11 | The special brick no longer fits the booklet → `undefined symbol` / import errors. |
| cuDNN bag missing or wrong major version | torch won't even open its box → `libcudnn.so.9: cannot open shared object file`. |
| Using a *prebuilt* flash-attn brick made for a different torch/CUDA/python | Looks right, snaps on, then shatters at runtime (ABI mismatch). |

**The safest move** is to copy a combination that already worked once
(here: the `mj` environment), instead of grabbing "the newest" of each piece and
hoping they snap together.

---

## The one extra wrinkle on *this* machine

Normally you'd just download the pre-made flash-attn brick (a "wheel") and skip
the brick-making. But this devserver's internet is guarded: it only lets us reach
a few brick stores (PyPI, the Anaconda store, and PyTorch's store). The stores
that sell **pre-made flash-attn bricks** (GitHub, NVIDIA's store) are **locked**.

So we had to:
1. Get the brick-maker (`nvcc`) from the **Anaconda store** (the NVIDIA store was locked).
2. Get **torch** from the **PyTorch store** (it speaks CUDA 12.8).
3. **Make the flash-attn brick ourselves** with nvcc — a ~20–40 minute job — because
   we couldn't buy it pre-made.

That's the whole reason `setup_h100.sh` looks more complicated than the README's
one-line `pip install`.

---

## Cheat sheet (the versions that snap together here)

| Piece | Version | Must agree with |
|-------|---------|-----------------|
| CUDA family | **12.8** | everyone |
| `nvcc` (conda cuda-toolkit) | **12.8.93** | CUDA family |
| `torch` | **2.7.0+cu128** | CUDA family |
| `cuDNN` | **9.x** (e.g. 9.5.1.17) | torch's major (9) |
| `flash-attn` | **2.7.4.post1** | built against this exact torch |
| Python | **3.12** | all wheels are `cp312` |

If you ever change ONE of these, you usually have to re-check ALL of them.
