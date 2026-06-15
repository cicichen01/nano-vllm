# Pushing nano-vllm changes to your own fork

This repo was cloned from the upstream main (`GeeeekExplorer/nano-vllm`). These
steps create your **own fork** and push to it, **without ever touching upstream**.

## ⚠️ Run these in your OWN terminal, not through Claude Code
The Claude Code agent's network identity is firewalled from `github.com`
(`agent_id=agent:claude_code` is not allowlisted), so the agent cannot push.
Your normal interactive SSH session to the devserver has regular GitHub access
(that's how this repo was cloned). Run every `git push` / `gh` / fetch command
from a plain terminal, not via the `!` prefix inside Claude Code.

Replace `cicichen01` below if your GitHub username differs.

---

## 1. Create the fork on GitHub
Web UI: open `https://github.com/GeeeekExplorer/nano-vllm` → click **Fork**.

Or with the CLI (re-auth first if needed):
```bash
gh auth login                 # GitHub.com → SSH
cd /home/cicichen/nano-vllm
gh repo fork --remote=false   # creates github.com/cicichen01/nano-vllm, leaves remotes alone
```

## 2. Point remotes so push goes ONLY to your fork
```bash
cd /home/cicichen/nano-vllm

# drop the stray remote whose name is literally the URL (created by accident)
git remote remove "https://github.com/GeeeekExplorer/nano-vllm" 2>/dev/null || true

git remote rename origin upstream                      # upstream = fetch-only main
git remote add origin git@github.com:cicichen01/nano-vllm.git   # origin = YOUR fork (SSH)
git remote set-url --push upstream DISABLE             # belt-and-suspenders: block pushing to main

git remote -v   # verify: origin=your fork; upstream=GeeeekExplorer with push=DISABLE
```
If `git fetch upstream` ever fails over HTTPS, switch it to SSH too:
```bash
git remote set-url upstream git@github.com:GeeeekExplorer/nano-vllm.git
```

---

## 3. Get your changes onto your fork's `main`

Recommended layout: keep helper files (setup_h100.sh, test_h100.py, *.md) on
`main` so they're always present, and do each code change on its own topic branch.

### Case A — files are still uncommitted
```bash
git checkout main
git add h100_setup/           # all 5 helper files now live under h100_setup/
git commit -m "Add H100 devserver setup, test driver, and repo notes"
git push origin main          # -> your fork's main; upstream untouched
```

### Case B — you already committed them on a branch (e.g. h100-setup) and want them on main
Works as a clean fast-forward when the branch is just `main + commits`:
```bash
git checkout main
git merge h100-setup          # fast-forward: main now contains the branch's commits
git push origin main          # -> your fork's main

# optional: remove the now-redundant branch (local + on your fork)
git branch -d h100-setup
git push origin --delete h100-setup
```

Verify:
```bash
git log --oneline -1 main     # shows your "Add H100 devserver..." commit
git status                    # the files are tracked on main
```

---

## 4. Do future code changes on topic branches
```bash
git checkout main
git checkout -b fp8-kv-cache  # one branch per feature/optimization
# ...edit code...
git commit -am "Implement FP8 KV cache"
git push -u origin fp8-kv-cache
```

## 5. Pull upstream updates later (without affecting upstream)
```bash
git fetch upstream
git checkout main
git merge upstream/main        # or: git rebase upstream/main
git push origin main           # mirror upstream progress into your fork
```

---

## Mental model / gotchas
- `git remote remove/rename/add` only edits remote bookmarks — it **never** touches files.
- An **untracked** file shows up on every branch (git ignores it on checkout). Once you
  **commit** it to a branch, switching away removes it from the working dir until you
  switch back (or merge that branch). That's why helper files belong on `main`.
- Keeping your fork's `main` equal to `upstream/main` (plus standalone helper files) makes
  step 5 a clean fast-forward; diverging `main` with lots of edits makes syncing conflict-prone.
- `upstream` push is disabled, and `origin` is your fork, so a stray `git push` can never
  reach `GeeeekExplorer/nano-vllm`.
