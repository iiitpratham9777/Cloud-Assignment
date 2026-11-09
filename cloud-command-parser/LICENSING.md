# Licensing status - please read before you publish

## Current situation

The archive this work started from contains **no `LICENSE` file** and no licence statement. Under copyright law, code with no
licence is "all rights reserved" by default: you may read it, but you have **no automatic right** to copy, modify or
redistribute it. The README only asks that attribution to the original author (Vivek Sagarwal) be retained.

Nobody who worked on the portability changes is the rights holder of the original code, so **no licence has been added**.
Adding a standard open-source licence in someone else's name would claim a permission that has not been given.

## What you need to do to publish legally

1. **Ask the original author, Vivek Sagarwal**, in writing (an email or message is fine) whether you may copy, modify and publish
   the project, and under which licence. If they agree, ask them to add a `LICENSE` file or to state the licence in writing.
2. **Check the "provided kit" files** listed in `ATTRIBUTION.md` (`data/validation.jsonl`, `data/schema.json`,
   `data/rate_card.json`, `tools/*`, `network/packets_sample.jsonl`). Their author is not recorded. If they came from a
   course/instructor, that party must permit redistribution; otherwise remove them from a public copy (the tests and CI
   reference `data/validation.jsonl` and `data/schema.json`, so you would need to supply your own).
3. **Keep the dataset licence** in mind: SLURP text is CC BY 4.0 (attribution required).
4. Until you have permission, keep your repository **private** (GitHub: *Settings -> General -> Danger zone -> Change visibility*,
   or create it with `--private`), or share only with people the author approved.

## Once the rights holder has agreed

`docs/LICENSE.mit.template` is a ready-to-use MIT text with a clearly marked placeholder for the copyright line. It is only a
template: **it is not in force** until the rights holder confirms the licence and the placeholder is filled in correctly.

```bash
cp docs/LICENSE.mit.template LICENSE      # Windows: copy docs\LICENSE.mit.template LICENSE
# edit LICENSE: replace the bracketed copyright line, e.g.
#   Copyright (c) <year> Vivek Sagarwal and contributors
git add LICENSE && git commit -m "Add licence agreed with the original author"
```

MIT is only a suggestion; the rights holder may prefer another licence (Apache-2.0, GPL, ...). Whatever is chosen:

* keep `ATTRIBUTION.md` and the credit lines;
* your own additions can carry your name in the copyright line, but only the original author can license their original parts;
* third-party components keep their own licences (`ATTRIBUTION.md`).

This file is documentation, not legal advice.
