param(
    [Parameter(Mandatory=$true)]
    [string]$RepositoryUrl
)

$ErrorActionPreference = "Stop"

git init
git add .
git commit -m "Initial public STRATUS artifact"
git branch -M main
git remote add origin $RepositoryUrl
git push -u origin main
