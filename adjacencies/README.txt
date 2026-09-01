Filename,                          Entries,       Directed,   Description
Slashdot0811.pkl,                  77316,         True,       Slashdot
adjacency_list_uni.pkl,            6095713,       True,       gpt 4 all directed
gpt4_io.pkl,                       6050977,       True,       gptkb 4 instances only
gpt4o_adj_from_dataset.pkl,        2920221,       True,       gpt4o all
gpt4o_io.pkl,                      2657109,       True,       gptkb 4o instances only
wiki-topcats.pkl,                  1791489,       True,       wiki-topcats

gpt4_io.pkl entsteht aus adjacency_list_uni.pkl, gefiltert auf type == instance
nach nodes/gpt4_nodes.pkl:
    python build_instances_only.py --adjacency adjacency_list_uni \
        --nodes gpt4_nodes --out gpt4_io
Zuordnung der nodes-Dateien ueber die Knotenmengen (siehe README,
"Entwurfsentscheidungen"): gpt4_nodes deckt 100.00 % von adjacency_list_uni ab,
gpt1_nodes 95.26 % von gpt4o_adj_from_dataset.
