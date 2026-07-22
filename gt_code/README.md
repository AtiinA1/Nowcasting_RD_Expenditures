# Google Trends data preparation

These scripts document the Google Trends topic-discovery, repeated-sampling,
and filtering workflow used to construct the cached topic panels under
[`../data/GT/`](../data/GT/).

| Script | Role |
| --- | --- |
| `1_related_topic_kw_data.py` | Collect related topic and keyword candidates |
| `2_topic_gt_data_multi_sampling.py` | Download repeated Google Trends samples by topic and country |
| `3_filtered_topics_df.py` | Filter and assemble the cross-country topic panel |

Google Trends values can vary across downloads because of sampling and
normalization. The checked-in files in `data/GT/` are therefore the research
vintage used by the saved analysis panel. Re-downloading the data is not
expected to reproduce every value exactly.

