# Multi-agent orchestration has a routing problem, not a modeling problem

I spent a week running a numerical analysis campaign with a multi-agent harness, and I
measured the leak. In a single day, six agent workflows consumed roughly 5.2 million
subagent tokens. At least 1.77 million of those, a third of the total, were spent
deriving results downstream of a premise that had already been refuted, in the same
session, before the work finished.

To be clear about who did what, because it changes how the rest of this should be
read: the derivations, the solver code, and the analysis were produced by AI agents
running under a harness I built and directed. I set the gates, chose what to probe,
and adjudicated the results. I did not derive the corner algebra by hand. The failure
modes below are therefore failures of a human-directed AI system, which is the system
most people reading this are also running, and the routing fix applies to it rather
than to a model in isolation.

The output was not wrong. It was excellent. Four independent derivations agreeing to
1.96e-16 on the live grid, exact symbolic closure checks, a twenty-variant mutation
battery, a sign convention pinned by discrimination rather than assumption. It
certified, beautifully, the algebra of a construction whose endpoint I had already
proven meaningless.

That is not a reasoning failure. Every agent in that workflow did its job correctly.
It is a routing failure, and it has a fix that is older than transformers.

## What actually happened

The campaign was computing a self-similar blowup profile. Partway through I launched a
workflow to certify some correction terms, which required three independent
derivations reconciled against each other. While it was in flight, working from data
already on disk, I established that those corrections vanish identically in the limit
the construction was aimed at. The endpoint died. The workflow did not notice, because
nothing connected the two.

The refusal was recorded. It was timestamped, typed, and sitting in the same ledger
the workflow could have read. Nobody walked the edge from the dead claim to the
in-flight job, because the edge did not exist.

## The missing structure

Agent frameworks build dependency graphs over tasks. Task B waits for task A. That is
scheduling, and it is solved.

What is missing is a dependency graph over *claims*, where nodes carry truth values
that change and the changes propagate. The distinction matters because tasks complete
and stay completed, while claims get refuted, superseded, and consumed, and everything
derived from them has to move when they do.

The graph is small:

    claim nodes:  {id, statement, status, evidence, minted_by, gates}
    edge types:   REFUTES, SUPERSEDES, CONSUMES, NULLIFIES, GATES, DERIVES-FROM
    agent jobs:   declare premise_ids BEFORE spawning
    router:       on status change, walk the DERIVES-FROM closure, mark suspect;
                  any in-flight job whose premise is suspect gets HALT or REDIRECT

The load-bearing part is the third line. Agents must declare what they are standing on
at spawn time, not describe it afterward. Once a job carries its premises as data, the
router is a graph walk.

## This is a truth maintenance system

The propagation mechanism is assumption-based truth maintenance, worked out in the
1980s by de Kleer and others: when an assumption dies, invalidate its dependents and
recompute. Well understood, thoroughly published, and to my knowledge never pointed at
agent orchestration, because until agents could burn millions of tokens on a dead
branch there was no bottleneck worth pointing it at.

Calling this new would be wrong. The contribution is the target, not the theory.

## Declaring premises works when done by hand

Before writing any of the above I had adopted a discipline of enumerating, at planning
time, the axes a plan's checks do not span, and recording each as an open tension
before starting work. Two of those entries were minted before the corresponding
campaigns ran.

One of them predicted the exact failure that then occurred three separate times: a
fit window that drifts with the quantity being measured, producing an apparent trend
that is an artifact of the window rather than a property of the flow. I hit that
failure three times in one afternoon, and each time the prediction was already
written down.

That is premise declaration performed manually. It is tedious, it is easy to skip
under momentum, and it works. Automating the declaration is what makes the router
possible, and the manual version is the evidence that the declaration is worth having.

## The byproduct is a dataset nobody has

Over the campaign the ledger accumulated 59 refusals against 37 promotions. The engine
refused more than it blessed, which is the ratio you want, and every refusal is
something more useful than a log entry.

Each one is a labeled example of reasoning that looked correct and was not, with the
diagnosis attached. From one day:

- A Richardson extrapolation that returned the published constant exactly. It was
  algebraically circular: the formula reconstructs whatever value you feed it, for any
  input. It would have been the most persuasive line in the writeup.
- A fitted exponent landing 0.20 percent from a known published value. The fitted
  quantity swept monotonically across a range that contains the target, so it crosses
  the right answer exactly once. Reading the crossing as a measurement means choosing
  the moment that returns the answer you already knew.
- A length scale frozen at 3.168 across every snapshot, reported as a physical result.
  It was pi plus an offset, the signature of a centering bug in my own analysis code.
- A degree-counting screen that passed on generic fields and failed on the solution
  manifold, where a term it certified as subleading is comparable instead.
- A global maximum used as the observable for a problem whose structure is local to a
  corner. The maximum wanders between distinct structures, so its growth rate
  oscillates between strongly positive and negative and no growth law fits.
- A correction pattern that survived a naive shuffle at p ~ 1e-4 and failed a
  phase-matched permutation control at p = 0.10, meaning it was a property of the
  current activity rather than a trend in time.

Six failure shapes, each with a mechanical verdict explaining why. This category of
data barely exists in public, because the incentive is to publish what worked and
discard the rest. A model trained on it would not be better at mathematics. It would
be better at recognizing, before spending, that a claim is about to be circular,
window-dependent, or validated on the wrong space.

## What the human was for

Since the agents did the deriving, the fair question is what the direction consisted
of. In this campaign it was two things, and neither was mathematics.

The first was anomaly detection. Not knowing an answer is wrong, but noticing that a
result carries the wrong texture before there is any argument against it. Twice this
week that took the form of an offhand objection that turned out to be structural. Once
it was noticing that a set of validation runs I was leaning on had been produced under
an older configuration and could not be compared to current output. The runs in
question had clean convergence numbers and looked authoritative. Their initial
condition was unrecorded, which made every comparison against them meaningless, and
nothing in the numbers said so.

The second was insisting that plans be written out before they were funded. Before
committing paid compute to a large run, I asked for the plan on paper and read back,
on the theory that staging a thing in prose exposes what staging it in code does not.
That pass found that every time value in the campaign was normalized against a
reference constant taken from a published paper whose configuration differed from
ours, and that our runs had been stopping at roughly half the distance to the event
they were meant to characterize. Five separate analyses had been quietly measuring
the wrong regime. The check cost nothing and arrived before the spend.

Both functions are pattern matching over texture rather than content, performed by
someone with sparse domain knowledge and a low tolerance for results that feel
convenient. I state it plainly because it is the part that does not automate, and
because the router described above is built precisely to carry everything that does.

## What this does not solve

A trained model is not required to run any of this. A frontier model runs the harness
today without modification. Training buys reflex and cost: catching the shape before
the spend rather than after.

The ceiling sits somewhere else. A router can kill dead branches, halt doomed jobs,
and propagate a refutation through everything that rests on it. It cannot decide which
axis to enumerate in the first place. Every unspanned axis in this campaign was named
by a person, and when I tested whether the ordering of past corrections could predict
the next one, a permutation control killed the claim. The record predicts. The
ordering does not.

Choosing what to check stays human. Everything downstream of that choice is a graph
walk, and the graph walk is worth 1.77 million tokens a day.
