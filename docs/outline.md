# Project outline (Week 1)

## Proposal (1 paragraph) -- TODO: write in our own words

Problem, dataset, planned approach.

## Where will we get data?

- Public: ASCEND (Hugging Face `CAiRE/ASCEND`, CC BY-SA 4.0) -- 10.62 h of spontaneous
  Mandarin-English code-switched conversation, 23 speakers, pre-split train/val/test.
- Our own: 3 conversations x 5-10 min among international students, recorded at the same
  time by three mic types: DJI Mic 2 lavs (close-talk), Tascam DR-70D (far-field),
  two ESP32 boards (low-cost). Devices borrowed from the MAC Equipment Center.

## Do we need to crawl the internet or annotate data ourselves?

- No crawling.
- Yes, annotation for our own recordings: an off-the-shelf model drafts transcripts,
  then we correct them by hand (lav tracks are easiest to hear). Budget several hours.
- Consent forms for every participant; check IRB with Prof. Walsh.

## What deep learning approaches have already been tried? (citations TODO: verify)

- ASCEND baselines: fine-tuned wav2vec 2.0 (Lovenia et al., 2022) -- best MER 27.05%
- Whisper and other multilingual pretrained ASR models
- Other Mandarin-English code-switching corpora and models (e.g. SEAME) -- TODO

## What new approaches can we try?

- Same conversations recorded by three mic types -> measure the effect of the microphone
  on code-switched ASR, before and after fine-tuning.
- Public-only vs. public + in-domain fine-tuning (Hong Kong speakers in ASCEND vs. our
  speakers).
- TODO: "language token prediction" -- clarify with Prof. Walsh.

## Roles -- TODO

- Mia:
- Charlie:
