```bat
@echo off
title Edge AI Project Remote Control
ssh -t iot@hitesh-pi.local "env -u DISPLAY /home/iot/edge-ai-project/edgeai-control.sh"
pause
```