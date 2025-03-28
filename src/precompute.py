from vectorization import *
from db import *
from utilities import *
from llm import *
from difflib import SequenceMatcher, HtmlDiff, ndiff
from IPython.display import display

TDR_V4 = '../data/tdr_v4.pdf'
TDR_V6 = '../data/tdr_v6.pdf'

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


def get_diffs(doc1: Document, doc2: Document, custom_name: str = ""):
    display(doc1)
    display(doc2)

    text1 = doc1.page_content# .replace("\n", "")
    text2 = doc2.page_content# .replace("\n", "")

    differ = ndiff(text1.split(), text2.split())
    diff = '\n'.join(differ)
    print(diff)

    html_diff = HtmlDiff().make_file(text1.splitlines(), text2.splitlines())
    with open(f"../data/differences{custom_name}.html", "w", encoding="utf-8") as f:
        f.write(html_diff)

    matcher = SequenceMatcher(None, text1, text2)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        print(f"{tag}: text1[{i1}:{i2}] -> text2[{j1}:{j2}]")

        if tag == 'equal':
            # print(matcher.a[i1:i2])
            pass
        elif tag == 'replace':
            print(f"\t{matcher.a[i1:i2]=}")
            print(f"\t{matcher.b[j1:j2]=}")
        elif tag == 'delete':
            print(f"\t{matcher.a[i1:i2]=}")
        elif tag == 'insert':
            print(f"\t{matcher.b[j1:j2]=}")


for idx, (s4, s6) in enumerate(zip(sections_v4, sections_v6)):
    # print(f"{idx=}")
    if idx != 17:
        continue
    
    try:
        get_diffs(docs_tdr4_map[s4], docs_tdr6_map[s6])
    except KeyError:
        print("KeyError")
    except Exception as e:
        print("Fallo")
        raise e
    break



sequence_matcher_idxs = \
    [5, 6, 7, 8, 11, 15, 17, 19, 20, 23, 24, 26, 27, 28, 30, 34, 35, 36, 37, 38]

attnt_idxs = [
    6, 7, 8, 15, 17, 23, 26, 34, 38
]

for idx, (s4, s6) in enumerate(zip(sections_v4, sections_v6)):
    # print(f"{idx=}, {s4=}, {s6=}")
    if idx not in sequence_matcher_idxs:
        continue
    
    try:
        get_diffs(docs_tdr4_map[s4], docs_tdr6_map[s6], f"{idx}")
    except KeyError:
        pass
    except Exception as e:
        print("Fallo")
        raise e
