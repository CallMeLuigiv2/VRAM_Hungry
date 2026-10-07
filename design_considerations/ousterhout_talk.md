# Ousterhout: "Can Great Programmers Be Taught?"

Notes taken 2026-09-28. Research on the wider philosophy behind the talk: `ousterhout_research.md`.

- **Talk:** John Ousterhout, "Can Great Programmers Be Taught?", Agile Lunch & Learn, 62 min. <https://youtu.be/LtRWu9DErgU>
- **Speaker:** Stanford CS professor. Built Tcl/Tk, the Sprite OS and its log-structured file system, RAMCloud, and co-designed the Raft consensus algorithm. Author of *A Philosophy of Software Design* (APoSD, 2018; 2nd edition 2021).
- **How these notes were made:** from the YouTube auto-generated transcript. Timestamps are `[mm:ss]` into the video. Quotes are cleaned up from auto-captions, so the wording may differ slightly from what he said.

## The argument in one paragraph

The most important idea in computer science is **problem decomposition**: how you cut a big problem into pieces you can solve mostly independently. Nobody teaches it. The best programmers are worth hundreds of average ones, yet nobody teaches what makes them good either. Ousterhout thinks it *can* be taught, the same way writing is taught: write, get heavy feedback, rewrite, repeat. He does this in a Stanford course (CS 190). His principles are philosophy, not a recipe. They only click once you see them applied to your own code in a review.

## Why he thinks it can be taught

- **[02:04] Problem decomposition** is his answer to "the most important concept in CS". Knuth's answer was "layers of abstraction", which is one kind of decomposition.
- **[03:09] 10x programmers.** Google VPs were asked how many of their worst programmers they'd give up to keep their best one. Typical answers were 300 to 500.
- **[04:28] *Talent Is Overrated*** (Geoff Colvin): across many fields, the best people are separated from average ones by lots of careful, deliberate practice, not an innate gift. So programming skill should be learnable too.
- **[05:00] Most faculty can't teach it** because they stopped programming after grad school. He writes 5,000 to 10,000 lines of code a year.
- **[07:18] Great programmers can't always explain it.** At lunch with Jeff Dean, Ken Thompson and Rob Pike, all three struggled to say *how* they write great code. They had never thought about it consciously.

## How he teaches it: CS 190, Software Design Studio

- **[08:34] Modeled on high-school writing class:** write, get it back covered in red ink, revise, resubmit. Most CS classes only grade "does it work?" and never let you revise.
- **Three phases per quarter:**
  1. Teams of two build a 2,000 to 3,000-line system from scratch in 2 to 3 weeks, with no hints on structure.
  2. Code reviews: students read each other's code and present parts of it in class; he reads every line and meets each team.
  3. Teams revise the code (and add features), then get reviewed again. Finally, a brand-new system from scratch and a final review.
- **[14:04] Teams of two, not three or four,** so nobody can coast. Both people must be fully engaged.
- **[13:04] Red flags.** The principles are vague on their own, so he also gives concrete, objective signs that a design has a problem. A beginner who can't yet design well can still spot red flags, fix them, and repeat until none remain. That usually gives decent code.
- **[14:29] Static analyzers don't help** with design. They find memory leaks and bugs, not design problems.
- **[49:42] It takes 5 to 10 years** of conscious work and good feedback to become a great developer. A 10-week class is only the start.

## Idea 1: classes should be deep [14:59]

"Maybe the single most important concept of them all."

- **The picture:** a class is a rectangle. Its area is the functionality it provides (the benefit). Its top edge is the interface (the cost). The interface is *everything a user must know to use it*, not just the function signatures.
- **Deep** = small interface, lots of functionality. **Shallow** = big interface, little functionality. Deep classes give you leverage against complexity: learn almost nothing, get a lot done.
- **Methods, modules and subsystems** can be deep or shallow too, not just classes.
- **[17:18] Worst-case shallow method:** calling it takes more keystrokes than writing its body inline. You must know everything it does to use it, so it only adds interface.
- **[19:43] He strongly disagrees with *Clean Code*** (Robert Martin) on "methods should be small, then smaller still" (at most one `if`/`while`, whose body is one statement). That advice leads to **classitis** [20:26]: every feature becomes a new tiny class, so the system is all interface and hides almost nothing.
- **[21:05] Java example:** reading serialized objects from a file used to take three stacked objects (`FileInputStream`, `BufferedInputStream`, `ObjectInputStream`). Nearly everybody wants buffering, so it should be the default. If you forget it, everything still works, just very slowly (one kernel call per character). Lesson: **make the common case easy.**
- **[23:36] Key quote:** "It's more important to have a simple interface than it is to have a simple implementation, because many more people are affected by the interface." Get the interface deep first. If the implementation grows too complicated, split the *implementation*.
- **[24:01] His favorite deep interface: Unix file I/O.** Five calls (open, read, write, lseek, close) with simple arguments. Behind them sit hundreds of thousands of lines of kernel code: disk layout, block allocation, directories, path lookup, permissions, disk scheduling, the block cache, device independence.
- **[25:50] Can a class be too deep?** He's never seen one. The implementation might get too complicated, but that's a separate problem.

## Idea 2: general-purpose classes are deeper [26:00]

- **He changed his mind.** He used to teach "build what you need now, specialized; generalize later if it gets reused." After years of reading student projects, he decided that was "really wrong". Specialized designs end up with complicated APIs and **information leakage**: several classes sharing knowledge about the same thing.
- **[28:10] Text editor example:** students had to write the class that stores the editor's text.
  - **Specialized version:** one method per UI action: insert character, backspace, delete key, delete selection. UI ideas like "cursor" and "selection" leak into the storage class. Every new UI action needs a new storage method.
  - **General-purpose version:** two methods, insert a string at a position and delete a range. The UI represents the cursor as one position and the selection as two. The split between text storage and UI is clean.
- **[30:39] His rule, "somewhat general-purpose":** build the *functionality* you need today, but design the *interface* so it isn't tied to how today's caller uses it. Don't go overboard.
- Even if the class is only ever used once, the general version still comes out simpler and easier to maintain. He only figured this out after 30+ years of programming.

## Idea 3: avoid specialization [31:57]

- Deep classes *hide* complexity. Avoiding specialization *removes* it, which is the best outcome.
- The same idea applies at every level: general-purpose classes, code without special case after special case, fewer places that have to handle exceptions. Special cases and exceptions *are* specialization.
- He names "define errors out of existence" at [12:14] as one of his principles but doesn't cover it in this talk (see the research file).

## Idea 4: strategic vs tactical programming [34:06]

- **Tactical:** the main goal is getting the next feature or bug fix working. "I mostly care about design, but this one small shortcut is fine."
- **[35:05] Complexity is incremental.** Systems don't get complicated from one big mistake. They get complicated from hundreds or thousands of small shortcuts, none of which seemed like a big deal. That's why it's so hard to fix afterwards: there's no single thing to fix, so people give up and add the next shortcut.
- **[35:53] Tactical tornadoes:** very fast engineers whom managers treat as heroes. They get things 80–90% done and leave a trail of mess for others. Every organization has at least one.
- **Strategic:** working code is required but not enough. The main goal is a great design, because most of a project's development is still in the future. Sweat the small stuff.
- **[38:01] The (admittedly made-up) graph:** tactical starts faster and keeps slowing down. Strategic starts a bit slower, then overtakes. Even strategic slows over time: "You can't win the battle against complexity, you can only slow it down."
- **[39:07] Crossover point:** his guess is 6 months to 2 years, because by 6 months you've forgotten how your own code works. He has no data and doesn't know how to measure it.
- **[40:01] Technical debt** is the tactical curve: borrowing time from the future, paid back with interest.
- **[40:24] Agile** is neither tactical nor strategic. The failure mode is "debugging the system into existence" with no design at all.
- **[41:05] Startups** are usually 100% tactical, planning to clean up after the next funding round. He doubts that cleanup ever really happens. Facebook's "move fast and break things" produced a notoriously unstable codebase: two of his grad students who interned there said every Monday's infrastructure push broke everything until Thursday. Facebook later changed the motto to "move fast with stable infra". Google and VMware had strong design cultures and succeeded anyway.
- **[44:32] The fastest way to build good software is to hire good engineers,** and good engineers don't want to work in spaghetti code. So a design culture is in a company's own self-interest.

## How much to invest [45:00]

- **Many small investments, not heroic ones.** You can't design everything up front, because you can't foresee the impact of every decision.
- **[45:30] About 10–20% extra time** over a purely tactical approach. [61:08] He clarified in Q&A that this is extra over tactical, not "3× slower".
- Leave time for iteration and refactoring. Document the code as you write it.
- **[46:53] When changing existing code,** don't aim for "fewest lines changed". That instinct comes from fear of breaking things, and it adds mess.
- **[47:21] The goal after a change:** the system should look as if it had been designed that way from the start, knowing everything you know now. Not always affordable, but it's the target.
- **[48:09] Always improve something** when you touch code. You're probably dropping a bit of litter yourself, so pick up someone else's.
- Ask yourself, then your boss, then their VP: "Is this really the most we can afford to invest right now?"

## Q&A

- **[57:14] Changing requirements mid-project** would reward good design. He once had teams take over another team's project for the third phase. It was realistic, but students disliked spending their time understanding someone else's code, so he switched back to a fresh design.
- **[59:17] Flowcharts and UML:** skeptical. Flowcharts focus on low-level details, not abstractions. He admits he has little experience with UML.
- **Language:** everyone in the class uses the same one (Java early on, C++ now) so students can review each other's code.
- **[62:18]** "Compliments make me happy, but criticism makes me better." He wants to hear disagreement.
