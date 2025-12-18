# Git Setup Guide for Options Backtester

## STEP 1: Create GitHub Repository (Do This First on GitHub.com)

1. Go to https://github.com/new
2. Repository name: `options-backtester` (or whatever you want)
3. Description: "Options strategy backtester with pattern detection"
4. Keep it Public or Private (your choice)
5. **DO NOT** check "Add a README" (we already have one)
6. **DO NOT** add .gitignore (we already have one)
7. Click "Create repository"
8. Copy the repository URL (looks like: https://github.com/YOUR_USERNAME/options-backtester.git)


## STEP 2: Run These Commands in VS Code Terminal

Open VS Code terminal (Ctrl+` or View > Terminal) and run these commands ONE BY ONE:

```bash
# Navigate to your project folder (change this path to where you put the files)
cd /path/to/options_backtester

# Initialize git repository
git init

# Configure your identity (use your GitHub email)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files to staging
git add .

# Create first commit
git commit -m "Initial commit: Options backtester with pattern detection"

# Rename branch to main (GitHub's default)
git branch -M main

# Connect to your GitHub repository (REPLACE with your actual URL)
git remote add origin https://github.com/YOUR_USERNAME/options-backtester.git

# Push to GitHub
git push -u origin main
```

## STEP 3: Verify It Worked

After pushing, refresh your GitHub repository page. You should see:
- README.md (displays automatically)
- main.py
- requirements.txt
- src/ folder
- tests/ folder


## For Future Changes

After you make edits, push updates with:

```bash
git add .
git commit -m "Description of what you changed"
git push
```


## Troubleshooting

**"remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/options-backtester.git
```

**Authentication failed**
You may need to use a Personal Access Token instead of password:
1. Go to GitHub > Settings > Developer settings > Personal access tokens
2. Generate new token with "repo" permissions
3. Use the token as your password when prompted

**"failed to push some refs"**
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```
