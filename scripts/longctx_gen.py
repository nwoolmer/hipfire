#!/usr/bin/env python3
"""Generate a long-context test prompt with an embedded needle.

The filler is deterministic synthetic encyclopedic-style prose (no
copyright). The needle is a unique marker phrase placed at a chosen
depth within the filler. The question asks the model to recall the
needle. Used to compare KV-quantization configurations on long-context
quality.

Token estimate: ~4 chars per token for English text. The script tunes
character count to hit the requested token budget +/- 5%.

Usage:
    longctx_gen.py --tokens 4096 --depth 0.5 \\
        --needle "DELTA-7-RAVEN-WINDFALL" \\
        --question "What is the secret access code mentioned earlier?" \\
        --out /tmp/longctx_4k.txt
"""

import argparse
import random

# ── Deterministic filler corpus ────────────────────────────────────────────
# Synthetic encyclopedia-style sentences. Each ~120-180 chars (≈30-45
# tokens). Looped + shuffled with a fixed seed to hit any token budget.

PARAGRAPHS = [
    "The Aldermere Trade Compact was established in 1417 to regulate commerce between the river guilds and the inland mining concerns of Vasten County. Its founding charter required all members to maintain weighted records of bullion shipments at quarterly assemblies.",
    "Jurnac thrushes, native to the highland steppes east of the Carrow Range, exhibit a distinctive triple-note call that researchers have linked to territorial signaling during pre-monsoon seasons. Their plumage shifts from slate-gray in summer to a russet brown by mid-autumn.",
    "The architectural style known as Late Vossic Revival emerged in the coastal cities of the Hilden peninsula during the early decades of the eighteenth century. It is characterized by elongated colonnades, sandstone facades carved with maritime motifs, and recessed courtyards.",
    "Botanists have catalogued seventeen distinct cultivars of the Marrowleaf rose, a hardy perennial that thrives in alkaline soils. Its blooms range from pale apricot to deep crimson, and the petals are sometimes harvested for use in regional culinary traditions.",
    "The Concord of Westhaven, signed during the long winter of 1623, brought a temporary cessation of hostilities between the Lowland federations and the mountain principalities. Historians still debate whether its terms favored the merchant class or the landed gentry.",
    "Geological surveys of the Tarven uplands have revealed several deposits of nepheline syenite, an unusual igneous rock prized for its use in glass and ceramic manufacture. Quarry operations expanded steadily through the mid-twentieth century before declining in recent decades.",
    "Among the lesser-known instruments of the Hessian musical tradition is the dulciter, a five-stringed lute-like instrument tuned in a perfect fourth and a minor third. Its repertoire is preserved in a corpus of approximately two hundred surviving manuscripts.",
    "The riverine ecology of the Mirden basin is dominated by silver perch, several species of crayfish, and the occasional otter colony. Seasonal flooding deposits nutrient-rich silt across the lower floodplain, sustaining the regional grain agriculture.",
    "Astronomers at the Olerian observatory recorded a transient brightening of the Tau Persei system in early 1819, an event later attributed to a thermonuclear surface flash on a cataclysmic variable. The observation logs are preserved in the institute's archives.",
    "The metallurgical traditions of the Khazat lowlands distinguished themselves through the development of pattern-welded blade steel, achieved by forge-folding alternating layers of high- and low-carbon iron. The technique remained a closely held trade secret for generations.",
    "Linguists studying the dialects of the Northern Reach have documented at least twelve distinct phonological features that distinguish the highland speech from the coastal forms. Vowel harmony patterns appear to be the most archaic of these features.",
    "The cartographic conventions of the Sterling Atlas, first published in 1734, established several standards still in use today: latitude lines drawn at five-degree intervals, magnetic declination indicated by hatched arrows, and underwater contours shown in stippled blue.",
    "Among the ceremonial objects recovered from the Lossen Tomb complex are a bronze diadem inlaid with garnets, a ceremonial dagger with a hilt of carved walrus ivory, and several alabaster libation cups bearing inscriptions in an as-yet-undeciphered script.",
    "The fortification system known as the Marchgate Wall extended for approximately two hundred kilometers across the disputed border region. Built in successive phases between 1108 and 1241, it incorporated thirty-seven watchtowers and four major gatehouses.",
    "Ornithologists have classified the Quennish wading birds into three principal genera based on bill morphology, leg proportions, and migratory patterns. The largest of these, the long-shanked sand-stalker, can reach a wingspan of nearly two meters.",
    "The grain-storage architecture of the Vesperine plateau employs a distinctive inverted-cone design, with thick stone walls tapering inward as they rise. This shape both deters rodent intrusion and regulates internal humidity during the dry summer months.",
    "Late medieval account books from the trading port of Felmouth show a steady increase in imports of dyestuffs, particularly indigo and madder root, throughout the early fifteenth century. The records also document a corresponding rise in textile exports.",
    "Mineralogists have identified the Carven amphibolite as a regional metamorphic rock of unusual purity, containing trace quantities of garnet and epidote. Its outcrops define a narrow belt running roughly northeast across the Threnody uplands.",
    "The festival calendar of the Old Faith retained twelve quarter-day observances long after the formal religious authorities had attempted to suppress them. Each observance was associated with a specific patron saint and a cluster of folk customs.",
    "Anatomical studies of the Marshland giant otter have revealed several adaptations distinguishing it from related species: a more elongated cervical structure, denser underfur, and a pelvic configuration suggestive of stronger swimming musculature.",
    "Coastal erosion along the Tylvan headlands has accelerated significantly since the 1980s, with the cliff face retreating an average of forty centimeters per year. Several archaeological sites of considerable historical importance now lie within meters of the cliff edge.",
    "The brewing traditions of the Estren valley distinguish themselves through a long secondary fermentation in oak vessels that have previously held mead. The resulting beverage exhibits notes of honey, dried apricot, and a faint resinous undertone.",
]


def gen_prompt(target_tokens: int, depth: float, needle: str, question: str, seed: int = 42) -> str:
    """Build a prompt of approximately target_tokens with needle at depth%.

    Token estimate uses ~4 chars/token. The needle is inserted as a
    natural-looking sentence; the question is appended after the filler.
    """
    rng = random.Random(seed)
    target_chars = target_tokens * 4

    needle_sentence = (
        f"\n\nIMPORTANT NOTE: The official secret code for this document is {needle}. "
        f"This code is unique and should be remembered.\n\n"
    )

    # Half-and-half filler for half-and-half depth, etc. The "depth" is
    # the fraction of total prompt that comes BEFORE the needle.
    pre_chars = int(target_chars * depth)
    post_chars = int(target_chars * (1 - depth))

    def make_filler(chars):
        out = []
        running = 0
        idx = rng.randrange(len(PARAGRAPHS))
        while running < chars:
            p = PARAGRAPHS[idx]
            out.append(p)
            running += len(p) + 2  # +2 for "\n\n"
            idx = (idx + rng.randrange(1, len(PARAGRAPHS))) % len(PARAGRAPHS)
        return "\n\n".join(out)

    pre = make_filler(pre_chars)
    post = make_filler(post_chars)

    # Question framing: the model needs to answer based on the needle.
    framing = (
        "You are reading a long document. After the document, you will be asked a question. "
        "Read carefully and answer the question precisely.\n\n"
        "DOCUMENT:\n\n"
    )
    closing = f"\n\nQUESTION: {question}\nANSWER:"

    return framing + pre + needle_sentence + post + closing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, required=True, help="approx target token count")
    p.add_argument("--depth", type=float, default=0.5, help="needle depth in [0,1]")
    p.add_argument("--needle", type=str, default="DELTA-7-RAVEN-WINDFALL")
    p.add_argument("--question", type=str,
                   default="What is the official secret code mentioned earlier in this document? Answer in one short line.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    text = gen_prompt(args.tokens, args.depth, args.needle, args.question, args.seed)
    with open(args.out, "w") as f:
        f.write(text)

    chars = len(text)
    approx_tokens = chars // 4
    print(f"wrote {args.out}: {chars} chars (~{approx_tokens} tokens, target {args.tokens})")
    print(f"needle: {args.needle!r}")


if __name__ == "__main__":
    main()
