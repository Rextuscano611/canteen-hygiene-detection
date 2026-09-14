# 🚀 GitHub Upload Commands — Canteen Hygiene Detection

Follow these steps in ORDER inside **Command Prompt or PowerShell**.

---

## STEP 0 — One-time Git setup (skip if already done)

```bash
git --version
# If not installed → https://git-scm.com/download/win

git config --global user.name "Rex"
git config --global user.email "your@email.com"
```

---

## STEP 1 — Navigate to your project folder

```powershell
cd "C:\Users\Rex\OneDrive\Desktop\social_pill\school"
```

---

## STEP 2 — Create assets folder & rename demo images

```powershell
mkdir assets
copy "WhatsApp Image 2026-08-21 at 3.34.21 PM.jpeg" "assets\demo_haircap.jpeg"
copy "WhatsApp Image 2026-08-21 at 3.34.22 PM.jpeg" "assets\demo_no_haircap.jpeg"
```

> Swap the names if needed based on which image is which.

---

## STEP 3 — Initialize Git

```bash
git init
git branch -M main
```

---

## STEP 4 — Stage files

```bash
git add .
git status
```

### ✅ Should be staged:
```
README.md
.gitignore
GITHUB_UPLOAD_COMMANDS.md
best.pt
build_final_dataset.py
extract_frames.py
assets/demo_haircap.jpeg
assets/demo_no_haircap.jpeg
canteen-hygiene-cctv.v1.../data.yaml
final_dem_2.v1.../data.yaml
final_merged_dataset/data.yaml
```

### ❌ Should NOT appear (excluded by .gitignore):
```
final_merged_dataset.zip   (391MB)
Screen Recording *.mp4     (151MB)
cctv_frames/
*/train/   */valid/   */test/
```

> If zip or mp4 show up as staged — STOP and check your .gitignore is in the right folder.

---

## STEP 5 — Commit

```bash
git commit -m "Initial commit: YOLOv8 canteen hygiene hair cap detection model"
```

---

## STEP 6 — Create GitHub repo

1. Go to → https://github.com/new
2. Name: `canteen-hygiene-detection`
3. Description: `YOLOv8 model to detect hair cap compliance in canteen CCTV footage`
4. Public or Private — your choice
5. ❌ Do NOT tick "Add a README" (you already have one)
6. Click **Create repository**

---

## STEP 7 — Connect and push

```bash
# Replace <your-username> with your GitHub username
git remote add origin https://github.com/<your-username>/canteen-hygiene-detection.git
git push -u origin main
```

> If it asks for a password → GitHub no longer accepts passwords.
> Create a Personal Access Token:
> GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
> → Generate → tick ✅ repo → copy the token → paste it as the password

---

## STEP 8 — Verify on GitHub

1. Open `https://github.com/<your-username>/canteen-hygiene-detection`
2. ✅ README renders with demo images
3. ✅ `best.pt` is listed (19.5MB, uploads fine without LFS)
4. ✅ No large files, no image folders, no mp4/zip

---

## 🔄 Push future changes

```bash
cd "C:\Users\Rex\OneDrive\Desktop\social_pill\school"
git add .
git commit -m "describe what changed"
git push
```

---

## ❓ Common Issues

| Problem | Fix |
|---|---|
| `git: command not found` | Install from git-scm.com |
| `error: src refspec main does not match` | Run `git branch -M main` |
| `remote: File xxx exceeds 100MB` | Not in .gitignore — run `git rm --cached <file>`, add to .gitignore, then push |
| `Authentication failed` | Use Personal Access Token not your password |
