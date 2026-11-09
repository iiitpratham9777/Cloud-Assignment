# Publish your own copy to GitHub

Goal: your own independent repository, with your changes, the original credits intact and no secrets or weights inside.

> **Before you make it public:** read [`../LICENSING.md`](../LICENSING.md). The original has no licence, so get the author's
> permission first, and create the repository as **private** until you have it.

The downloaded zip has no `.git` folder, so you start a new Git history. (If the original author gives you a repository URL you
may instead *fork* it on GitHub - that preserves the original history. The steps from "Add your changes" onward are the same.)

## 1. One-time preparation

* Install [Git](https://git-scm.com/downloads) and create a free GitHub account.
* Tell Git who you are (this becomes the author of your commits):
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "you@example.com"   # or your GitHub no-reply address
  ```
* Authenticate **without writing secrets into files**: either install the [GitHub CLI](https://cli.github.com) and run
  `gh auth login`, or use HTTPS and let Git Credential Manager (included with Git for Windows/macOS) ask you to sign in, or set
  up an SSH key. Never paste a personal access token into a file, a URL (`https://TOKEN@github.com/...`) or a commit.

## 2. Check the project is clean

```bash
python scripts/dev.py setup          # if you have not already
python scripts/dev.py test
python scripts/dev.py audit          # must say "nothing suspicious found"
```

Look through `ATTRIBUTION.md` and add your name next to the existing credits (do not delete them). Update `STATUS.md` with only
things you actually ran.

## 3. Create the local repository

```bash
git init -b main
git status                 # .env, .venv/, model_out/ and weights must NOT be listed
git add .
git status                 # read the list once more
python scripts/dev.py audit
git commit -m "Initial commit: customised cloud-command-parser (based on work by Vivek Sagarwal)"
```

If `git status` shows a file that should be private, stop, add it to `.gitignore` (or delete it) and `git rm --cached <file>`
if it was already added. A secret that was committed is considered leaked even if you delete it later - rotate it.

## 4. Create the empty repository on GitHub and push

**With the GitHub CLI** (easiest):
```bash
gh repo create cloud-command-parser --private --source . --remote origin --push
```

**With the website:**
1. github.com -> **New repository** -> name it, choose **Private**, and do **not** tick "Add a README", ".gitignore" or "licence"
   (the repository must start empty).
2. Copy its URL (HTTPS or SSH), then:
   ```bash
   git remote add origin https://github.com/<your-user>/<your-repo>.git
   git push -u origin main
   ```

Open the repository page and check that `.env` and any weights are not there. The **Actions** tab runs the CI; on a new repository the
model-evaluation jobs are skipped until you publish a model release (next step).

## 5. Optional: publish your trained weights as a Release

Weights are too large for Git and are ignored. After `python scripts/dev.py train --version v1`:

```bash
git add models/v1/model.sha256 && git commit -m "Add checksum of model v1" && git push
gh release create model-release model_out/model-v1.tar.gz --title "Model v1" --notes "Fine-tuned model; see README"
```

Others can then run `python scripts/fetch_model.py --repo <your-user>/<your-repo> --tag model-release --version v1`.
Check the licences of the base model and data (`ATTRIBUTION.md`) before publishing weights. For the Kubernetes workflow, run
`REPO_SLUG=<your-user>/<your-repo> make up`.

## 6. Day-to-day

```bash
git switch -c my-feature     # work on a branch
# ... edit, then:
python scripts/dev.py lint && python scripts/dev.py test && python scripts/dev.py audit
git add -A && git commit -m "Describe the change" && git push -u origin my-feature
```

Then open a pull request on GitHub. Make the repository public only after the licence question is settled.
