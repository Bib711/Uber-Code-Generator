# Restore Original Theme Script
# Run this script to revert back to the original UI theme

Write-Host "Restoring original theme..." -ForegroundColor Yellow

# Restore index.css
Copy-Item ".\css-backup\index.css.bak" ".\index.css" -Force
Write-Host "✓ Restored index.css" -ForegroundColor Green

# Restore page CSS files
Copy-Item ".\css-backup\ChatPage.css" ".\pages\ChatPage.css" -Force
Write-Host "✓ Restored ChatPage.css" -ForegroundColor Green

Copy-Item ".\css-backup\DashboardPage.css" ".\pages\DashboardPage.css" -Force
Write-Host "✓ Restored DashboardPage.css" -ForegroundColor Green

Copy-Item ".\css-backup\LandingPage.css" ".\pages\LandingPage.css" -Force
Write-Host "✓ Restored LandingPage.css" -ForegroundColor Green

Copy-Item ".\css-backup\LoginPage.css" ".\pages\LoginPage.css" -Force
Write-Host "✓ Restored LoginPage.css" -ForegroundColor Green

# Restore component CSS files
Copy-Item ".\css-backup\Toast.css" ".\components\Toast.css" -Force
Write-Host "✓ Restored Toast.css" -ForegroundColor Green

Copy-Item ".\css-backup\UserMenu.css" ".\components\UserMenu.css" -Force
Write-Host "✓ Restored UserMenu.css" -ForegroundColor Green

Write-Host ""
Write-Host "Original theme restored successfully!" -ForegroundColor Cyan
Write-Host "Restart your development server to see the changes." -ForegroundColor White
