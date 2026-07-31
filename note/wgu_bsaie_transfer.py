#!/usr/bin/env python3
"""WGU BSAIE transfer + pacing engine.  CU-accurate.

AUTHORITATIVE SOURCE: WGU Program Guidebook, Bachelor of Science AI Engineering,
Program Code BSAIE, Catalog Version 202607, published 2026-05-07 (~/Downloads/BSAIE.pdf).
Course list, CU values and standard-path term assignments below are transcribed from
its Standard Path table.  121 total CUs across 40 courses, standard path = 9 TERMS.

Transfer articulations: wgucollegeofinformationtechnology.sophia.org (fetched 2026-07-29).
Study.com policy page (fetched 2026-07-29) publishes no BSAIE mapping; the program
postdates their table.

VERBATIM POLICY, from the guidebook (these settle several questions):
  * "WGU does not waive any requirements based on a student's professional experience
    and does not perform a 'resume review' or 'portfolio review' that will
    automatically waive any degree requirements."
        -> Research output, public repos, and shipped work earn ZERO credit. Closed.
  * "Even when you do not directly receive credit, the knowledge you possess may help
    you accelerate the time it takes to complete your degree program."
        -> The benefit is speed through assessments, not waivers.
  * "Certifications verified through third parties may also be included in your
    program as a way to demonstrate competency."
        -> This is the hook for asking about CompTIA Security+ and Azure AI-900.
  * "undergraduate students must enroll in at least 12 competency units each term"
  * "you must complete at least 66.67% of the units you attempt over the length of
    your program"  (SAP; matters if using financial aid)
  * Terms are 6 months. Tuition $4,200/term + $200 resource fee = $4,400.

From the Sophia page, the post-March-2026 restriction:
  "After March 2026, WGU can accept general education and lower-division courses for
   transfer, but upper-division courses must be completed at WGU."
And from WGU's Student Policy Handbook: transfer coursework "shall not be used to
fulfill upper division requirements."  This applies to ALL transfer sources, not just
ACE partners, so routing credit through Charter Oak / TESU / Excelsior does not help.

Status codes:
  SOPHIA    published Sophia articulation, name-level match
  CC        lower-division at a regionally accredited community college; permitted by
            policy, and the strongest lever because it covers the math
  UNCERTAIN plausible but the WGU target differs in rigor or has no published mapping
  WGU_ONLY  no route; complete at WGU
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict

TUITION_PER_TERM = 4400
MIN_CU_PER_TERM = 12


@dataclass
class C:
    n: int
    name: str
    cu: int
    term: int          # standard-path term from the guidebook
    status: str
    via: str = ""
    note: str = ""


# Transcribed from the guidebook Standard Path table. 40 courses, 121 CUs, 9 terms.
P = [
    # term 1 (14 CU)
    C(1, "Introduction to AI Engineering", 3, 1, "WGU_ONLY"),
    C(2, "Object Oriented Programming in Python", 3, 1, "UNCERTAIN",
      "Sophia: Introduction to Python Programming",
      "Sophia's Python maps to WGU 'Introduction to Programming in Python', a "
      "DIFFERENT course. A community-college OOP/CS2 course is the better bet."),
    C(3, "Calculus I", 4, 1, "CLEP",
      "CLEP Calculus exam (~$95, single sitting)",
      "TRAP: Sophia's Calculus I articulates to WGU APPLIED ALGEBRA, which BSAIE does "
      "not require. Do not buy Sophia for this. Take it at a community college."),
    C(4, "Health, Fitness, and Wellness", 4, 1, "SOPHIA",
      "Sophia: Introduction to Nutrition [SOPH-0063]"),
    # term 2 (15 CU)
    C(5, "Web Development Foundations", 3, 2, "SOPHIA",
      "Sophia: Introduction to Web Development [SOPH-0043]"),
    C(6, "Applied Probability and Statistics", 3, 2, "SOPHIA",
      "Sophia: Introduction to Statistics [SOPH-0005]"),
    C(7, "Calculus II for Engineers", 3, 2, "WGU_ONLY", "",
      "Verify the CC course maps to the 'for Engineers' variant."),
    C(8, "Data Management - Foundations", 3, 2, "SOPHIA",
      "Sophia: Introduction to Relational Databases [SOPH-0047]"),
    C(9, "Network and Security - Foundations", 3, 2, "UNCERTAIN",
      "possibly CompTIA Security+",
      "ASK: guidebook permits third-party certifications to demonstrate competency."),
    # term 3 (13 CU)
    C(10, "Version Control", 1, 3, "WGU_ONLY", note="1 CU. Trivial for him."),
    C(11, "Calculus III for Engineers", 3, 3, "WGU_ONLY", "",
      "Multivariable. Heaviest single course in the program; also the most useful "
      "for interpretability work. Worth taking with a live instructor."),
    C(12, "AI Engineering with C#", 3, 3, "WGU_ONLY"),
    C(13, "Composition: Successful Self-Expression", 3, 3, "SOPHIA",
      "Sophia: English Composition I [SOPH-0015]"),
    C(14, "Applied Discrete Mathematics", 3, 3, "WGU_ONLY",
      "",
      "Typically 200-level, so lower division and transferable. Verify."),
    # term 4 (14 CU)
    C(15, "Azure AI Fundamentals", 3, 4, "UNCERTAIN",
      "possibly Microsoft AI-900 held beforehand",
      "Program embeds AI-900. ASK whether holding it already satisfies the course."),
    C(16, "Linear Algebra for Engineers", 3, 4, "WGU_ONLY",
      "",
      "Directly load-bearing for interpretability."),
    C(17, "Data Structures and Algorithms I", 4, 4, "WGU_ONLY",
      "",
      "Common CC offering (CS2). Verify mapping."),
    C(18, "Discrete Mathematics II", 4, 4, "UNCERTAIN",
      note="Sequel course; may be treated as upper division. Verify."),
    # term 5 (14 CU)
    C(19, "Data Structures and Algorithms II", 4, 5, "WGU_ONLY",
      note="Likely upper division."),
    C(20, "Big Data Foundations", 4, 5, "WGU_ONLY"),
    C(21, "Introduction to Systems Thinking and Applications", 3, 5, "WGU_ONLY"),
    C(22, "Mathematics of AI", 3, 5, "WGU_ONLY", note="Program-specific."),
    # term 6 (15 CU)
    C(23, "Advanced C#", 3, 6, "WGU_ONLY"),
    C(24, "Introduction to Communication: Connecting with Others", 3, 6, "SOPHIA",
      "Sophia: Business Communication [SOPH-0059]"),
    C(25, "Ethical Engineering", 3, 6, "UNCERTAIN",
      "Sophia: Introduction to Ethics [SOPH-0020]",
      "Sophia's Ethics maps to Introduction to Humanities, not Ethical Engineering."),
    C(26, "Computer Architecture", 3, 6, "WGU_ONLY", "",
      "Commonly offered at CC as Computer Organization/Architecture. Verify."),
    C(27, "Fundamentals of Information Security", 3, 6, "UNCERTAIN",
      "possibly CompTIA Security+", "Same certification ask as course 9."),
    # term 7 (14 CU)
    C(28, "C# .NET Back End Development", 3, 7, "WGU_ONLY"),
    C(29, "Machine Learning", 3, 7, "WGU_ONLY"),
    C(30, "Computer Systems for AI", 3, 7, "WGU_ONLY"),
    C(31, "Deep Learning for AI Engineers", 3, 7, "WGU_ONLY"),
    C(32, "Data and Information Governance", 2, 7, "WGU_ONLY"),
    # term 8 (12 CU)
    C(33, "American Politics and the US Constitution", 3, 8, "UNCERTAIN",
      "Sophia / Study.com American Government",
      "Not in the published Sophia IT table; check the Partners Portal."),
    C(34, "Machine Learning DevOps", 2, 8, "WGU_ONLY"),
    C(35, "Natural Language Processing for AI Engineers", 3, 8, "WGU_ONLY"),
    C(36, "General Chemistry I", 3, 8, "CC", "community college Gen Chem I (single course, no prereq chain)",
      "TRAP: Sophia's Introduction to Chemistry maps to Integrated Physical Sciences, "
      "a different and lighter course. Take real Gen Chem at a CC."),
    C(37, "General Chemistry I Lab", 1, 8, "CC", "community college Gen Chem I Lab"),
    # term 9 (10 CU)
    C(38, "Software Engineering", 4, 9, "WGU_ONLY"),
    C(39, "Computer Vision for AI Engineers", 3, 9, "WGU_ONLY"),
    C(40, "Applied AI Engineering (Capstone)", 3, 9, "WGU_ONLY",
      note="Capstone. Cannot be transferred under any circumstance."),
]

TOTAL_CU = sum(c.cu for c in P)


def bucket(st):
    return [c for c in P if c.status == st]


def cu(rows):
    return sum(c.cu for c in rows)


def terms_for(remaining_cu, pace):
    return -(-remaining_cu // pace)


def report():
    print("=" * 79)
    print("WGU BS AI ENGINEERING (BSAIE) -- CU-ACCURATE TRANSFER + PACING")
    print(f"Guidebook 202607, pub 2026-05-07.  {len(P)} courses, {TOTAL_CU} CUs, "
          "standard path 9 TERMS.")
    print("=" * 79)

    print("\n!! WGU's marketing page says 'approximately 24 months'. The guidebook's")
    print("   standard path is 9 terms = 54 months. 24 months requires ~30 CU/term,")
    print("   which is 2.5x the 12 CU/term full-time minimum.")

    for st, label in (("SOPHIA", "SOPHIA ($99/mo, non-proctored)"),
                      ("CC", "COMMUNITY COLLEGE (only where no prereq chain)"),
                      ("UNCERTAIN", "UNCERTAIN (verify in Partners Portal)"),
                      ("WGU_ONLY", "MUST COMPLETE AT WGU")):
        rows = bucket(st)
        print(f"\n--- {label}: {len(rows)} courses, {cu(rows)} CU ---")
        for c in rows:
            tag = f"  {c.n:2d}. [{c.cu} CU] {c.name}"
            print(tag)
            if c.via:
                print(f"        via {c.via}")
            if c.note:
                print(f"        ! {c.note}")

    s, c_, u, w = (cu(bucket(x)) for x in ("SOPHIA", "CC", "UNCERTAIN", "WGU_ONLY"))
    cl = cu(bucket("CLEP"))
    c_ = c_ + cl
    print("\n" + "=" * 79)
    print(f"  Sophia route ................ {s:3d} CU  ({s/TOTAL_CU:5.1%})")
    print(f"  Community college route ..... {c_:3d} CU  ({c_/TOTAL_CU:5.1%})")
    print(f"  ---- prep floor ............. {s+c_:3d} CU  ({(s+c_)/TOTAL_CU:5.1%})")
    print(f"  Uncertain (upside) .......... {u:3d} CU")
    print(f"  ---- prep ceiling ........... {s+c_+u:3d} CU  "
          f"({(s+c_+u)/TOTAL_CU:5.1%})")
    print(f"  Irreducibly at WGU .......... {w:3d} CU  ({w/TOTAL_CU:5.1%})")

    print("\n  WGU TIME AND COST AFTER PREP  (term = 6 months, $4,400)")
    print(f"  {'scenario':<22}{'CU left':>8}{'@12':>8}{'@18':>8}{'@25':>8}{'@30':>8}")
    for label, cleared in (("no prep", 0), ("Sophia only", s),
                           ("prep floor", s + c_), ("prep ceiling", s + c_ + u)):
        rem = TOTAL_CU - cleared
        row = f"  {label:<22}{rem:>8}"
        for pace in (12, 18, 25, 30):
            t = terms_for(rem, pace)
            row += f"{t:>5}t/${t*TUITION_PER_TERM//1000}k"
        print(row)
    print("\n  @12 = full-time minimum.  @30 = the pace WGU's '24 months' assumes.")
    print("  Realistic for him: fast on AI/ML/programming, slow on Calc III and")
    print("  Discrete II, so ~18-25 CU/term is the honest planning band.")
    print("=" * 79)


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps([asdict(c) for c in P], indent=1))
    else:
        report()
