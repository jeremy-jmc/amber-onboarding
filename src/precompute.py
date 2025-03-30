from vectorization import *
from db import *
from utilities import *
from llm import *
from difflib import SequenceMatcher, HtmlDiff, ndiff
from IPython.display import display
import unicodedata
import pandas as pd

shutil.rmtree('../data/diff', ignore_errors=True)
os.makedirs('../data/diff', exist_ok=True)

TDR_V4 = '../data/tdr_v4.pdf'
TDR_V6 = '../data/tdr_v6.pdf'

CHARACTER_SHIFT = 1000

tv4_dict = get_docs(TDR_V4)
tv6_dict = get_docs(TDR_V6)


print(tv4_dict.keys())
original_to_clean = tv4_dict['section_mapping_original_to_clean']


docs_tdr4: list[Document] = tv4_dict['docs']
docs_tdr4_map = {
    doc.metadata['section'][-1]: doc
    for doc in docs_tdr4
}
docs_tdr6: list[Document] = tv6_dict['docs']
docs_tdr6_map = {
    doc.metadata['section'][-1]: doc
    for doc in docs_tdr6
}


# get_all_keys(tv6_dict['tree']).symmetric_difference(get_all_keys(tv4_dict['tree']))

sections_v4 = sorted(list(get_all_keys(tv4_dict['tree'])))
sections_v6 = sorted(list(get_all_keys(tv6_dict['tree'])))
assert len(docs_tdr4) == len(docs_tdr6)
assert sections_v4 == sections_v6

# print(json.dumps(tv4_dict['tree'], indent=2, ensure_ascii=False))
# print(json.dumps(tv6_dict['tree'], indent=2, ensure_ascii=False))


def is_trivial_change(frag1: str, frag2: str) -> bool:
    """ Devuelve True si la diferencia es solo por espacios o saltos de línea """
    return frag1.strip() == frag2.strip()


def groupping_ranges(df, threshold = 100):

    merged_ranges = []
    current_range = df.iloc[0].copy()

    for i in range(1, len(df)):
        row = df.iloc[i]
        
        if current_range['i2'] + threshold >= row['i1']:
            current_range['i2'] = max(current_range['i2'], row['i2'])
            current_range['j1'] = min(current_range['j1'], row['j1'])
            current_range['j2'] = max(current_range['j2'], row['j2'])
        else:
            merged_ranges.append(current_range)
            current_range = row.copy()

    merged_ranges.append(current_range)
    df_merged = pd.DataFrame(merged_ranges)

    return df_merged


def get_diffs(doc1: Document, doc2: Document, custom_name: str = ""):
    text1 = clean_text(doc1.page_content)
    text2 = clean_text(doc2.page_content)

    differ = ndiff(text1.splitlines(), text2.splitlines())
    # print(f"{differ=}")
    diff = '\n'.join(differ)
    # print(diff)

    html_diff = HtmlDiff().make_file(text1.splitlines(), text2.splitlines())
    with open(f"../data/diff/differences{custom_name}.html", "w", encoding="utf-8") as f:
        f.write(html_diff)

    matcher = SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()
    print(f"Similarity: {similarity:.2%}")

    op_codes = matcher.get_opcodes()
    if not len(op_codes):
        return []
    
    diff_list = []
    for tag, i1, i2, j1, j2 in op_codes:
        # print(f"{tag}: text1[{i1}:{i2}] -> text2[{j1}:{j2}]")

        frag1 = matcher.a[i1:i2]
        frag2 = matcher.b[j1:j2]

        if is_trivial_change(frag1, frag2):
            continue  # Omitir cambios triviales

        diff_list.append({
            # "tag": tag,
            "i1": i1,
            "i2": i2,
            "j1": j1,
            "j2": j2,
            # "chunk_size1": i2 - i1,
            # "chunk_size2": j2 - j1,
        })

    df_diff = pd.DataFrame(diff_list)
    
    if len(df_diff):
        df_diff = groupping_ranges(df_diff)

        df_diff["chunk_v4"] = df_diff.apply(
            lambda row: text1[max(row["i1"]-CHARACTER_SHIFT, 0):min(row["i2"]+CHARACTER_SHIFT, len(text1))],
            axis=1
        )
        df_diff["chunk_v6"] = df_diff.apply(
            lambda row: text2[max(row["j1"]-CHARACTER_SHIFT, 0):min(row["j2"]+CHARACTER_SHIFT, len(text2))],
            axis=1
        )
        df_diff = df_diff.drop_duplicates(subset=["chunk_v4", "chunk_v6"])
        # df_diff["text1"] = df_diff.apply(
        #     lambda row: text1[row["i1"]:row["i2"]],
        #     axis=1
        # )
        # df_diff["text2"] = df_diff.apply(
        #     lambda row: text2[row["j1"]:row["j2"]],
        #     axis=1
        # )

    return df_diff


automatic_seq_idxs = []
total_diffs = {}

for idx, (s4, s6) in enumerate(zip(sections_v4, sections_v6)):
    # print(f"{idx=}")
    # if idx != 5:
    #     continue
    
    try:
        diff_list = get_diffs(docs_tdr4_map[s4], docs_tdr6_map[s6], f"{idx}")
        if len(diff_list):
            automatic_seq_idxs.append(idx)
        total_diffs[idx] = (idx, s4, s6, diff_list.assign(
            idx=idx,
            section_v4=s4,
            section_v6=s6
        ))
    except KeyError:
        print("KeyError")
    except Exception as e:
        print("Fallo")
        raise e
    # break

print(f"{automatic_seq_idxs=}")

df_total_diffs = pd.DataFrame()
for tup in total_diffs.values():
    df_total_diffs = pd.concat([df_total_diffs, tup[3].assign(idx=tup[0])], ignore_index=True)
df_total_diffs['chunk_length_v4'] = df_total_diffs['chunk_v4'].apply(len)
df_total_diffs['chunk_length_v4'] = df_total_diffs['chunk_v4'].apply(len)

# df_total_diffs.groupby('idx').size()
# total_diffs[6][2]


def add_records_from_dataframe(df: pd.DataFrame):
    records = [SectionDiff(
        section_v4=row['section_v4'],
        section_v6=row['section_v6'],
        chunk_v4=row['chunk_v4'],
        chunk_v6=row['chunk_v6']
    ) for _, row in df.iterrows()]
    Session.bulk_save_objects(records)
    Session.commit()

add_records_from_dataframe(df_total_diffs[['section_v4', 'section_v6', 'chunk_v4', 'chunk_v6']])

