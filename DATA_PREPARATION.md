# Data Preparation

This document explains where to obtain the original videos for the four datasets used in this project, and where to place them in the repository.

## Summary

The four datasets involved here are:

- `PreVAD-Instruct27k`
- `UCF-Crime`
- `XD-Violence`
- `MSAD`

Among them:

- `UCF-Crime` is publicly downloadable from the UCF CRCV project page.
- `XD-Violence` is publicly downloadable from the official project page.
- `MSAD` requires a request/application for access to the original videos.
- `PreVAD-Instruct27k` is this project's instruction-tuning dataset. Its raw video pool is not represented as a single upstream public dataset package in the current repo layout.

## 1. UCF-Crime

Official project page:

- https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/

Official dataset download entry listed on that page:

- `https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip`
- Dropbox mirror is also listed on the same page.

Recommended local layout:

```text
datasets/
└── UCF-Crime/
    └── videos/
        └── test/
            ├── Arrest039_x264.mp4
            ├── Burglary037_x264.mp4
            └── ...
```

The current annotation file has been normalized to repository-relative paths:

- [datasets/UCF-Crime/ucf-crime_test_anno.json](/data/wxx/毕设/PreVAD-Instruct27k/datasets/UCF-Crime/ucf-crime_test_anno.json:1)

## 2. XD-Violence

Official project page:

- https://roc-ng.github.io/XD-Violence/

The official page lists video downloads and annotations under its `Download` section, including Baidu Netdisk and OneDrive entries.

Recommended local layout matching your current JSON:

```text
datasets/
└── xd-violence/
    └── other_datasets/
        └── xd_videos/
            ├── Brick.Mansions.2014__#00-41-25_00-42-36_label_B1-0-0.mp4
            ├── Taken.Extended.Cut.2008__#00-50-45_00-51-04_label_B2-B6-0.mp4
            └── ...
```

Your current annotation file already uses relative paths:

- [datasets/xd-violence/xd_test_anno.json](/data/wxx/毕设/PreVAD-Instruct27k/datasets/xd-violence/xd_test_anno.json:1)

## 3. MSAD

Reference repository:

- https://github.com/Tom-roujiang/MSAD

According to the MSAD repository README, the original video dataset is distributed via a request form, and extracted features are provided separately.

Recommended local layout matching your current JSON:

```text
datasets/
└── MSAD/
    └── other_datasets/
        └── msad_videos/
            ├── Assault_2.mp4
            ├── Assault_4.mp4
            └── ...
```

Your current annotation file already uses relative paths:

- [datasets/MSAD/msad_test_anno.json](/data/wxx/毕设/PreVAD-Instruct27k/datasets/MSAD/msad_test_anno.json:1)

Important:

- MSAD is not a direct anonymous public zip in the same sense as UCF-Crime
- the official release text states that access to original videos should be requested through their application form

## 4. PreVAD-Instruct27k

This dataset is different from the three external benchmarks.

From the current annotation structure:

- [datasets/PreVAD-Instruct27k/filter_train.json](/data/wxx/毕设/PreVAD-Instruct27k/datasets/PreVAD-Instruct27k/filter_train.json:1)
- [datasets/PreVAD-Instruct27k/filter_test.json](/data/wxx/毕设/PreVAD-Instruct27k/datasets/PreVAD-Instruct27k/filter_test.json:1)

the expected raw video layout is:

```text
datasets/
└── PreVAD-Instruct27k/
    ├── AbnormalVideos/
    └── NormalVideos/
```

However, the filenames indicate that this benchmark is built from multiple upstream sources rather than one standalone original dataset package. The annotations reference videos such as:

- custom abnormal video clips
- normal clips with names resembling `MSR-VTT`, `VATEX`, and `VALOR32K` source formats

So for `PreVAD-Instruct27k`, you have two realistic options:

1. Release only the annotation files and clearly state that the raw video collection is not redistributed in this repository.
2. Add a separate reconstruction guide documenting how each source subset was collected and mapped into `AbnormalVideos/` and `NormalVideos/`.

If you want a clean public release, option 1 is usually safer unless you have redistribution rights for all constituent videos.

## Recommended normalization before GitHub release

Before pushing the final public version, keep dataset paths in repository-relative form:

- `datasets/PreVAD-Instruct27k/AbnormalVideos/...`
- `datasets/PreVAD-Instruct27k/NormalVideos/...`
- `datasets/UCF-Crime/videos/test/...`
- `datasets/xd-violence/other_datasets/xd_videos/...`
- `datasets/MSAD/other_datasets/msad_videos/...`

## Notes on licensing and redistribution

For a public GitHub release, do not assume all videos can be mirrored inside the repository.

Safer practice:

- keep annotation JSONs in Git
- keep raw videos outside Git
- provide official dataset links or access instructions
- explain any licensing or application requirements in `README.md`

## Sources

- UCF-Crime official project page: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
- XD-Violence official page: https://roc-ng.github.io/XD-Violence/
- MSAD repository README: https://github.com/Tom-roujiang/MSAD/blob/master/README.md
