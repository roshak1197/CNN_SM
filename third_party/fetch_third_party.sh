#!/bin/bash
# Downloads the third-party code, weights and datasets used for the comparison of section S8 of the supplementary
# material. Nothing of this is redistributed with the repository: everything lands in third_party/ (git-ignored).
#   ESPINNet   Qiao et al., J. Synchrotron Rad. 33 (2026)      https://github.com/QDLDQQiao/ESPINNet       (MIT)
#              test data: OneDrive link in the ESPINNet README (manual download, see below)
#   StrainNet  Boukhtache et al., Opt. Lasers Eng. (2021)       https://github.com/DreamIP/StrainNet        (CC BY-NC-SA 4.0)
#              weights: Google Drive folder in the StrainNet README (manual download, see below); Star frames in the repo
#   MIST       Alloo et al., Sci. Rep. (2023), wattle flower    https://github.com/samanthaalloo/MIST_Alloo23 (MIT; data in repo)
#   Fly        OSF project 4ruq2                                https://osf.io/4ruq2/
# Usage: bash third_party/fetch_third_party.sh
set -e
cd "$(dirname "$0")"
[ -d ESPINNet ]      || git clone https://github.com/QDLDQQiao/ESPINNet.git
[ -d StrainNet ]     || git clone https://github.com/DreamIP/StrainNet.git
[ -d MIST_Alloo23 ]  || git clone https://github.com/samanthaalloo/MIST_Alloo23.git
if [ ! -d fly_4ruq2 ]; then
  curl -L -o fly_4ruq2.zip "https://files.osf.io/v1/resources/4ruq2/providers/osfstorage/?zip="
  mkdir -p fly_4ruq2 && unzip -q fly_4ruq2.zip -d fly_4ruq2 && rm fly_4ruq2.zip
fi
cat <<'MSG'
Manual steps (links require a browser):
 1. ESPINNet test data: download the folders 'refs' and 'sample_in' from the OneDrive link given in
    third_party/ESPINNet/readme.md ("Test data") into third_party/ESPINNet/refs and third_party/ESPINNet/sample_in
    (RandomPlate_001.tif ... RandomPlate_020.tif in each).
 2. StrainNet weights: download StrainNet-h from the Google Drive folder given in third_party/StrainNet/README.md to
    third_party/StrainNet/pretrained_download/StrainNet/StrainNet-h/StrainNet-h.pth.tar
Then: python scripts/prepare_third_party.py
MSG
