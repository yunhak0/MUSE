# Preprocessing Process

1. Please download the raw data (Training_Set.tar.gz) from the [original challenge](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge).

2. Unzip the 'Training_Set.tar.gz' file.
    * The file must be in the './data/mssd-org/raw/' folder.

<p align="center">
  <img src="https://github.com/yunhak0/MUSE/assets/40286691/934965ce-712b-4a5f-ba64-4bd557e33bed" alt="drawing" width="400" />
</p>

3. Run '1_prep_data_step0.py' file. (File Integration)

>>> ```python prep_data_step0.py```

<p align="center">
  <img src="https://github.com/yunhak0/MUSE/assets/40286691/4fd1493b-313b-4b13-b30c-f744a5e68693" alt="drawing" width="400" />
</p>


4. Run '2_prep_data_step1.sh' file.

>>> ```sh prep_data_step1.sh```

<p align="center">
  <img src="https://github.com/yunhak0/MUSE/assets/40286691/a38516b8-6f7f-4b71-b0b8-66dbbe9fedc7" alt="drawing" width="400" />
</p>

5. Run '3_prep_data_step2.sh' file.
    * The results file must be in the './data/mssd-{days}-all-{chunk}/all' folder.

>>> ```sh prep_data_step2.sh```

<p align="center">
  <img src="https://github.com/yunhak0/MUSE/assets/40286691/54ccf4a3-d645-4a91-9972-a9ce097c183e" alt="drawing" width="400" />
</p>
