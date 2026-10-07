# Ousterhout's design philosophy: research

Researched 2026-09-28, following up `ousterhout_talk.md`. The talk covers 4 ideas. The book behind it, *A Philosophy of Software Design* (APoSD), has about 15 principles and a checklist of "red flags".

**Where this comes from:** the author's book page, his published debate with Robert Martin, his course pages, published chapter summaries and reviews (links at the bottom), and what Claude already knew of the book. The book itself wasn't open while writing, so check page-level details against a copy before quoting them. Chapter numbers are from the 2nd edition (2021).

## The core idea: complexity

APoSD defines complexity as *anything about the structure of a system that makes it hard to understand and modify*. The whole book is about reducing it.

**Three symptoms:**
1. **Change amplification:** a simple change needs edits in many places.
2. **Cognitive load:** how much you must know to make a change.
3. **Unknown unknowns:** you can't tell which code you need to change, or what you need to know. This is the worst of the three.

**Two causes:**
1. **Dependencies:** code that can't be understood or changed on its own.
2. **Obscurity:** important information that isn't obvious.

**How it builds up:** incrementally. See the talk [35:05]: many small shortcuts, never one big mistake.

## The design principles

APoSD collects these in a list at the end of the book. The last one was added in the 2nd edition.

| # | Principle | In one line |
|---|---|---|
| 1 | Complexity is incremental: sweat the small stuff | No single shortcut matters; all of them together do |
| 2 | Working code isn't enough | Strategic, not tactical (ch. 3) |
| 3 | Make continual small investments in the design | About 10–20% extra time (talk [45:30]) |
| 4 | Modules should be deep | Small interface, lots of functionality (ch. 4) |
| 5 | Make the most common usage as simple as possible | Java buffering should be the default (talk [21:05]) |
| 6 | A simple interface matters more than a simple implementation | More people use the interface than read the implementation |
| 7 | General-purpose modules are deeper | Interface general, functionality for today (ch. 6, expanded in 2nd ed.) |
| 8 | Separate general-purpose and special-purpose code | Keep special cases out of general code (chs. 6, 9) |
| 9 | Different layers should have different abstractions | If two adjacent layers look the same, one is probably useless (ch. 7) |
| 10 | Pull complexity downward | The module suffers so its users don't (ch. 8) |
| 11 | Define errors and special cases out of existence | Change the meaning of an operation so the error can't happen (ch. 10) |
| 12 | Design it twice | Sketch at least two very different designs before picking one (ch. 11) |
| 13 | Comments should describe things not obvious from the code | Intent, units, invariants, why; not a restatement of the code (ch. 13) |
| 14 | Design for ease of reading, not ease of writing | Code is read far more often than it is written |
| 15 | Development increments should be abstractions, not features | Each step adds a complete abstraction, not a feature bolted on |
| 16 | Separate what matters from what doesn't, and emphasize what matters | 2nd edition, ch. 21 "Decide What Matters" |

## The techniques, with the book's examples

### Information hiding and leakage (ch. 5)
- Each module should hide *design decisions*: data formats, algorithms, representations. The idea comes from David Parnas, "On the Criteria To Be Used in Decomposing Systems into Modules" (1972).
- **Information leakage:** the same design decision shows up in several modules, so changing it means changing all of them.
- **Temporal decomposition** is the most common cause. You split code by the *order things happen* instead of by *what knowledge each piece hides*. Book example: a program reads a file, modifies it, and writes it back. If "reader" and "writer" are separate classes, *both* must know the file format. Better: one class owns the format.

### Different layer, different abstraction (ch. 7)
- **Pass-through method:** does almost nothing except call another method with a similar signature. Red flag: the layers aren't dividing responsibility.
- **Pass-through variable:** threaded through a long chain of functions that don't use it, just to reach one that does. Book fix: a **context object** that holds such shared state, so it isn't threaded through by hand.

### Pull complexity downward (ch. 8)
- If a module can handle some complexity itself, it should, rather than pushing it onto every caller.
- **Configuration parameters** often push complexity *up*: "I don't know the right retry interval, so the user can set it." It's usually better for the module to compute a good value itself.

### Better together or better apart? (ch. 9)
- Combine pieces when they share information, are always used together, overlap in concept, or can't be understood one without the other.
- Split special-purpose from general-purpose code.
- Split a method only if the result is two pieces you can understand *independently*. If you must keep flipping between parent and child to understand either one, they are **conjoined** (a red flag).

### Define errors out of existence (ch. 10)
Exceptions are a major source of complexity, because every exception a module throws becomes part of its interface. Four techniques, best first:
1. **Define the error away** by changing the operation's meaning:
   - Tcl's `unset` raised an error when the variable didn't exist. It should have meant "make sure this variable doesn't exist".
   - Windows refuses to delete a file that's open. Unix removes the name right away and frees the file when the last user closes it.
   - Java's `substring` throws on out-of-range indexes. Clamping would be simpler, as Python slicing does: `"abc"[1:100] == "bc"`.
2. **Mask the exception:** handle it low down so higher layers never see it. TCP resends lost packets, so applications never see packet loss.
3. **Aggregate exceptions:** one handler for many errors. For example, one catch at the top of a server's request loop instead of one per operation.
4. **Just crash** when recovery isn't worth it, e.g. out of memory.

### Design it twice (ch. 11)
Your first idea is rarely the best. Sketch two or more *radically different* designs, compare them, then pick one (or a mix). It costs a little time on each important decision, and you learn what separates a good design from a bad one.

### Comments (chs. 12–15)
- **Comments are part of the abstraction.** An interface comment says what a user needs to know; an implementation comment says what the code does and why.
- **Write the comments first,** before the body. If a method's comment is hard to write or has to be long, the abstraction is weak. It's a design tool, not an afterthought.
- **Names** should be precise. If a good name is hard to find, the thing may not have a clean purpose.
- **Put subtle knowledge in code comments, not commit messages.** Nobody reading the code will find it in the git log.

### Modifying code, consistency, obviousness (chs. 16–18)
- **After a change,** the code should look as if it had been designed that way from the start (talk [47:21]).
- **Consistency gives leverage:** learn a convention once, and it applies everywhere.
- **"Obvious" is decided by the reader.** If a reviewer finds your code unclear, it isn't obvious, whatever you think.

### Software trends (ch. 19)
His views on common practices:

| Practice | His view |
|---|---|
| Implementation inheritance | Adds dependencies; prefer composition |
| Agile | Fine if the increments are abstractions, not features |
| Unit tests | Strongly in favor |
| Test-driven development (TDD) | Too focused on features, not design |
| Design patterns | Fine when they fit; harmful when forced |
| Getters/setters | Mostly shallow; they expose representation |

### Designing for performance (ch. 20)
The chapter most relevant to this project.
- **Know which operations are fundamentally expensive:** network round trips, disk I/O, dynamic memory allocation, cache misses. Keep them in mind while designing, without optimizing blindly.
- **Measure before modifying.** Your intuition about where time goes is often wrong. Measure again after the change, and back it out if it didn't help.
- **Design around the critical path:**
  1. Ask what the *minimum* code is that the common case must execute.
  2. Restructure so the real path is close to that minimum.
  3. Move special cases off the path, ideally behind a single check.
- **Clean design and speed usually agree:** simple code with few special cases tends to be fast. The book's example is a buffer class from RAMCloud, redesigned around its critical path to be both simpler and faster.

### Decide what matters (ch. 21, 2nd edition)
- Good design separates the few things that matter from the many that don't. Structure the system around the important ones.
- Make the important things visible: names, documentation, prominent parameters.
- **Two failure modes:** treating too many things as important (clutter, too many options), and missing what is important.

## The red-flag checklist

The part meant to be used during review. If you see one, reconsider the design.

| Red flag | Symptom |
|---|---|
| Shallow module | The interface isn't much simpler than the implementation |
| Information leakage | One design decision is reflected in several modules |
| Temporal decomposition | Code structure follows execution order, not information hiding |
| Overexposure | To use a common feature you must learn about rare ones |
| Pass-through method | A method only forwards its arguments to a similar method |
| Repetition | The same nontrivial code appears over and over |
| Special-general mixture | Special-purpose code isn't cleanly separated from general-purpose code |
| Conjoined methods | You can't understand one method without reading the other |
| Comment repeats code | Everything in the comment is already obvious from the code |
| Implementation contaminates interface | An interface comment describes implementation details users don't need |
| Vague name | The name is too imprecise to convey much |
| Hard to pick name | You can't find a precise, intuitive name |
| Hard to describe | Complete documentation for it would have to be long |
| Nonobvious code | You can't easily understand what the code does or means |

## Where people disagree

### Ousterhout vs Robert Martin ("Uncle Bob", *Clean Code*)

A written debate between them, September 2024 to February 2025, published on GitHub (link below). Both want less cognitive load for the reader; they disagree on how to get there.

| Topic | Ousterhout | Martin |
|---|---|---|
| **Method length** | Split only when it gives deep pieces. "If two pieces of code are tightly related, the solution is to bring them together." Tiny methods cause *entanglement*: you must read several at once. | "One Thing Rule": extract a method whenever it can be meaningfully named. "I'd rather err on the side of decomposition." Accepts some entanglement. |
| **Comments** | Irreplaceable for what code can't say (intent, tradeoffs, subtle algorithms). A missing comment costs 10–100× more than a wrong one. Writes 5–10× more comment lines than Martin. | "Comments are always failures", a necessary evil. Prefers long descriptive names. "It's harder to ignore a name than a comment." |
| **Tests** | "Bundling": write a chunk of code (a few methods, up to a few hundred lines), then thorough unit tests. Code isn't done until it's tested. | TDD's three laws: seconds-long test/code cycles inside red-green-refactor loops. |

**The case study** was Martin's `PrimeGenerator` from *Clean Code* (8 tiny methods).
- **Ousterhout's rewrite:** one 65-line method with heavy comments.
- **Martin's revision:** split into 4 methods. Splitting a loop made it **3–4× slower**, so he merged the loops back.
- **The final result:** Martin's version ran about 21% faster than Ousterhout's.
- **What both concluded:** the original had tangled internal decomposition, and some algorithms are hard to explain no matter how you structure them.
- **Ousterhout's criticism:** Martin had "dropped the ball" on performance by putting decomposition first.

### Other reviewers
- **Gergely Orosz** (The Pragmatic Engineer):
  - Agrees with deep modules, information hiding, design it twice, and strategic vs tactical.
  - Disagrees with the anti-exception stance (fine in backends with monitoring), the criticism of event-driven code, and the weight on comments ("inline comments are an invitation for refactoring").
  - Says the book barely covers testing, technical debt, and team design reviews (RFCs).
  - Recommends chapters 1–9 and 14 most.
- **YAGNI objection** ("you aren't gonna need it"): general-purpose design invites speculative features. Ousterhout's answer in the talk [30:39]: generality belongs in the *interface*, not the *functionality*. Build only what you need today.
- **Style objection:** deep modules mean longer functions and more comments, the opposite of *Clean Code*. That's the same debate as above.

### Claude's read (a synthesis, not from the sources)
- **They mostly agree:** tangled code is bad, and interfaces should be simple. The real disagreement is *where to draw boundaries* and *how much to write in English*.
- **In performance code, "better together" wins more often.** The `PrimeGenerator` loop split that cost 3–4× is a small case of a big pattern: every separate pass over data re-reads memory. GPU inference relies on **kernel fusion** for this reason. FlashAttention fuses attention's matmul → softmax → matmul so the large score matrix never goes out to GPU main memory. This connects Ousterhout's "better together" with M6–M7 of this project.
- **This project's per-function order** (signature → test → body, from CLAUDE.md) sits between the two. It's test-first, like TDD, but in function-sized bundles, like Ousterhout, not seconds-long cycles.

## What this means for this engine

One line per idea, tagged with the milestone where it shows up. The details belong to each milestone (just-in-time).

| Idea | Where | Concrete example |
|---|---|---|
| Deep module | M1, M3 | llama2.c's `forward(transformer, token, pos)` returns the logits (`run.c:231`): three inputs, the whole transformer behind them. |
| General-purpose interface | M3, M7 | llama2.c has one `matmul(xout, x, w, n, d)` (`run.c:217`) for all 8 weight multiplies (`run.c:260–360`), not `compute_query`, `compute_ffn_up`, ... For M7: build CPU f32 today, but don't bake "CPU" or "f32" into the interface. CLAUDE.md already asks for that in M1 (keep a later GPU device in mind). |
| Information leakage / temporal decomposition | M2 | Only one place should know the checkpoint's weight order. In llama2.c that's `memory_map_weights` (`run.c:111`). The trap: a "read header" step and a "read weights" step that each need to know the format. |
| Define errors out of existence | M4 | llama2.c's `encode` has no "unknown character" error: anything not in the vocabulary becomes raw byte tokens, id = byte + 3 (`run.c:524–531`). This is also the "handle unknown tokens" interview question (see the M4 tokenizer side project). |
| Pull complexity downward | M5, M6 | The forward pass shouldn't know whether weights are f32 or Q8_0; the matmul should hide it. The thread count should have a good default, not be a required flag. |
| Pass-through variables / context object | M3 | llama2.c's `Transformer` struct (`run.c:67–75`) holds config, weights and run state in one context object. In Rust, one big `&mut` context can fight the borrow checker; decide that in M3. |
| Design for performance | M6 | "Measure before modifying" is M6's "profile first". The critical path is the per-token decode loop. llama2.c's `forward` never allocates (0 `alloc` calls in `run.c:231–365`); all buffers are allocated once in `malloc_run_state` (`run.c:77`), keeping allocation, one of the book's expensive operations, off the hot path. |
| Design it twice | every decision | Already in the 5-step loop: step 3's pros/cons and the *alternatives* field in DECISIONS.md. |
| Write the comments first | every function | Option: add a `///` doc comment to the "signature first" step, before the test and body. Luigi's call. |
| Strategic, 10–20% | whole project | CLAUDE.md's rules (a test per kernel, clippy/fmt clean, commit per function, DECISIONS.md) are exactly these small, steady investments. |
| Change looks designed-in | M6, M7 | Batched prefill (M6) and the device abstraction (M7) will reshape `forward`. Aim for "designed this way from the start", not a bolt-on. |
| Red flags | every review | Claude can use the red-flag table as the checklist when reviewing Luigi's code. |

## His method, if you want to practice it

- **The CS 190 projects are public** (Winter 2023 page below):
  1. Raft leader election plus a remote shell.
  2. Raft log replication, producing a replicated state machine.
  3. "Tiny Make", a small version of `make`.

  Raft is a fault-tolerant distributed system, which is Performance Engineer role project 5. It could be an alternative to the weight-distribution side project after v1.
- **The loop he says actually teaches design:** write → review → **rewrite**. This project's 5-step loop has the review (step 4), but not an explicit rewrite pass after it. One could be added.

## Sources

- Talk: <https://youtu.be/LtRWu9DErgU> (notes in `ousterhout_talk.md`)
- Author's book page (2nd edition changes, extract, Clean Code debate link): <https://web.stanford.edu/~ouster/cgi-bin/aposd.php>
- Ousterhout vs Martin debate: <https://github.com/johnousterhout/aposd-vs-clean-code>
- CS 190 Winter 2023: <https://web.stanford.edu/~ouster/cs190-winter23/>
- Chapter summary, 2nd ed.: <https://www.sglavoie.com/posts/2025/03/30/book-summary-philosophy-software-design-2nd-edition/>
- Notes (performance, errors, temporal decomposition): <https://linghao.io/notes/a-philosophy-of-software-design>
- Red flags list: <https://sportebois.medium.com/software-design-red-flags-wisdom-nuggets-from-john-ousterhout-8a9b0045e2bb>
- Gergely Orosz's review: <https://blog.pragmaticengineer.com/a-philosophy-of-software-design-review/>
- A critical review (not read in full; the page blocked automated access): <https://andremoniy.medium.com/not-my-philosophy-of-software-design-13d9f1e09451>
- D. L. Parnas, "On the Criteria To Be Used in Decomposing Systems into Modules", *Communications of the ACM*, 1972.
- Geoff Colvin, *Talent Is Overrated*, 2008.
