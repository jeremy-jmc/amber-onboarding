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

def remove_accents(text: str) -> str:
    """Elimina las tildes de un texto"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def clean_text(t: str) -> str:
    t = remove_accents(t)
    # t = re.sub(r'\s+', ' ', t)
    return re.sub(r'(?<=[a-z\(\[]) *\n *(?=[a-z\)\]])', ' ', t)


def is_trivial_change(frag1: str, frag2: str) -> bool:
    """ Devuelve True si la diferencia es solo por espacios o saltos de línea """
    return frag1.strip() == frag2.strip()


def get_diffs(doc1: Document, doc2: Document, custom_name: str = ""):
    display(doc1)
    display(doc2)

    text1 = clean_text(doc1.page_content)   # .replace("\n", " ")
    text2 = clean_text(doc2.page_content)   # .replace("\n", " ")

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
        print(f"{tag}: text1[{i1}:{i2}] -> text2[{j1}:{j2}]")

        frag1 = matcher.a[i1:i2]
        frag2 = matcher.b[j1:j2]

        if is_trivial_change(frag1, frag2):
            continue  # Omitir cambios triviales

        if tag == 'equal':
            # print(matcher.a[i1:i2])
            continue
        elif tag == 'replace':
            print(f"\t{matcher.a[i1:i2]=}")
            print(f"\t{matcher.b[j1:j2]=}")
            
            change_score = SequenceMatcher(None, frag1, frag2).ratio()
            print(f"\t(Score: {change_score:.2f}) | Replace: {frag1} -> {frag2}")
        elif tag == 'delete':
            print(f"\t{matcher.a[i1:i2]=}")

            change_score = SequenceMatcher(None, frag1, "").ratio()
            print(f"\t(Score: {change_score:.2f}) | Delete: {frag1}")
            
        elif tag == 'insert':
            print(f"\t{matcher.b[j1:j2]=}")

            frag2 = text2[j1:j2]
            change_score = SequenceMatcher(None, "", frag2).ratio()
            print(f"\t(Score: {change_score:.2f}) | Insert: {frag2}")

        print(f"\t{frag1=}")
        print(f"\t{frag2=}")

        diff_list.append({
            "tag": tag,
            "label": f"{tag}: text1[{i1}:{i2}] ({frag1}) -> text2[{j1}:{j2}] ({frag2})",
            "text1": frag1,
            "text2": frag2,
            "i1": i1,
            "i2": i2,
            "j1": j1,
            "j2": j2,
            "chunk1": text1[max(i1-CHARACTER_SHIFT, 0):min(i2+CHARACTER_SHIFT, len(text1))],
            "chunk2": text2[max(j1-CHARACTER_SHIFT, 0):min(j2+CHARACTER_SHIFT, len(text2))],
            "diff": diff,
        })
    
    df_diff = pd.DataFrame(diff_list).groupby(
        ["chunk1", "chunk2"]
    ).agg({
        "label": list
    }).reset_index(drop=False)
    return df_diff


automatic_seq_idxs = []
total_diffs = {}
for idx, (s4, s6) in enumerate(zip(sections_v4, sections_v6)):
    # print(f"{idx=}")
    if idx != 6:
        continue
    
    try:
        diff_list = get_diffs(docs_tdr4_map[s4], docs_tdr6_map[s6], f"{idx}")
        if len(diff_list):
            automatic_seq_idxs.append(idx)
        total_diffs[idx] = (s4, s6, diff_list)
    except KeyError:
        print("KeyError")
    except Exception as e:
        print("Fallo")
        raise e
    # break

# for chunk in diff_list:
#     print(json.dumps(chunk, indent=2, ensure_ascii=False))
print(f"{automatic_seq_idxs=}")


total_diffs[6][2]



sequence_matcher_idxs = \
    [5, 6, 7, 8, 11, 15, 17, 19, 20, 23, 24, 26, 27, 28, 30, 34, 35, 36, 37, 38]

attnt_idxs = [
    6, 7, 8, 15, 17, 23, 26, 34, 38
]

for idx, (s4, s6) in enumerate(zip(sections_v4, sections_v6)):
    # print(f"{idx=}, {s4=}, {s6=}")
    if idx not in sequence_matcher_idxs + automatic_seq_idxs:
        continue
    
    try:
        get_diffs(docs_tdr4_map[s4], docs_tdr6_map[s6], f"{idx}")
    except KeyError:
        pass
    except Exception as e:
        print("Fallo")
        raise e
