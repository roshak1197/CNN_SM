# third_party/

Third-party code, weights and data used only for the comparison of section S8. Nothing here is committed except
this file and the two scripts:

* `fetch_third_party.sh` — clones ESPINNet, StrainNet and MIST_Alloo23 from GitHub, downloads the fly data from
  OSF (project 4ruq2), and lists the two manual downloads (ESPINNet test data, StrainNet-h weights).
* `link_local_copies.sh` — on the original server only: links the copies already on disk.

Expected layout after fetching:
```
third_party/ESPINNet/{DS_network.py, estimate_single.py, trained_model/, refs/, sample_in/}
third_party/StrainNet/{StrainNet/, Star_frames/, pretrained_download/StrainNet/StrainNet-h/StrainNet-h.pth.tar}
third_party/MIST_Alloo23/{Allooetal23_SciRep_MIST.py, DATA_WattleFlower_25keV_2m_SBPCXI/}
third_party/fly_4ruq2/{ref/, sample/}
```
Please cite the original works when using these data (see the main README).
