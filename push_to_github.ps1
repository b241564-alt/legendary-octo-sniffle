# Push local repository to your GitHub account
param(
    [Parameter(Mandatory=$true)]
    [string]$RepoUrl
)

Write-Host "Configuring remote origin to: $RepoUrl" -ForegroundColor Cyan
git remote remove origin 2>$null
git remote add origin $RepoUrl
git branch -M main
Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
git push -u origin main
Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
