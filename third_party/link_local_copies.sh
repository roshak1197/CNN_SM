#!/bin/bash
# Only for the original server (labhep): symbolic links to the copies already on disk instead of downloading.
set -e
cd "$(dirname "$0")"
ln -sfn ~/ESPINNet ESPINNet
ln -sfn ~/StrainNet StrainNet
ln -sfn ~/CNN_Sphere_Sample/Convergencia/MIST_Alloo23 MIST_Alloo23
ln -sfn ~/CNN_Sphere_Sample/Convergencia/4ruq2-osfstorage-archive fly_4ruq2
ls -l
