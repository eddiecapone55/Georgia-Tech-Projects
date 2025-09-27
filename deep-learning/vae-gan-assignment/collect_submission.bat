@echo off
if exist assignment_4_submission.zip del /F /Q assignment_4_submission.zip
tar -a -c -f assignment_4_submission.zip configs models/*.py losses/*.py utils/*.py outputs/vae/*.pth outputs/gan/*.pth outputs/diffusion/*.pth *.py