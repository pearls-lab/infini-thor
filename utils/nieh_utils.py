import os
import json
import re
import random
from PIL import Image
from fuzzywuzzy import fuzz
from word2number import w2n


# Optional synonym map for basic matching
SYNONYM_MAP = {
    "table": ["side table", "sidetable", "coffee table", "desk"],
    "dresser": ["wooden dresser", "drawer", "chest", "on the dresser", "on the drawer"],
    "sofa": ["couch", "loveseat", "on the couch", "on the sofa", "on couch", "to the couch"],
    "desk": ["table", "side table", "on the table", "the table", "to the table", "on the desk"],
    "creditcard": ["card", "credit card"],
    "coffeetable": ["coffee table", "table", "on the table", "on table", "the table", "to the table", "to the coffeetable"],
    "sidetable": ["side table", "table", "on the table", "on table", "the table", "to the table"],
    "armchair": ["chair", "the chair", "on the chair", "to the chair", "on chair"],
    "bed": ["on the bed", "on bed", "to the bed"],
    "pen": ["pencil", "marker", "ink pen", "inkpen"],
    "pencil": ["pen", "marker", "ink pen", "inkpen"],
    "phone": ["cellphone", "cell phone", "mobile phone", "smartphone", "on the phone"],
    "cellphone": ["phone", "cell phone", "mobile phone", "smartphone", "on the cellphone"],
    "keychain": ["keys", "key"],
    "baseballbat": ["stick", "baseball bat", "bat"],
    "remotecontrol": ["remote"],
    "countertop": ["on the table", "to the countertop", "on the countertop", "to the counter", "on the counter"],
    "fridge": ["refrigerator", "fridge", "to the fridge"],
    "sink": ["on the sink", "to the sink", "in the sink"],
    "diningtable": ["table", "on the table", "to the table"],
    "shelf": ["on the shelf", "to the shelf", "on the top shelf", "top shelf"],
    'garbagecan': ['trash can', 'on trash can']
}


def normalize_answer(ans: str) -> str:
    if isinstance(ans, (int, float)):
        return str(ans)

    try:
        ans = str(ans).lower().strip()
    except Exception:
        return str(ans)

    # Attempt to convert word numbers like "one", "zero" to digits
    try:
        ans_num = w2n.word_to_num(ans)
        return str(ans_num)
    except:
        pass
    
    # Remove punctuation
    ans = re.sub(r"[^\w\s]", "", ans)

    # Remove leading phrases like "CD is", "The object is", etc.
    ans = re.sub(r"^\s*\w+\s+is\s+", "", ans)

    return ans.strip()


def are_synonyms(a, b) -> bool:
    a = a.lower()
    b = b.lower()
    for word, syns in SYNONYM_MAP.items():
        if (a == word and b in syns) or (b == word and a in syns):
            return True
        if a in syns and b in syns:
            return True
    return False


def get_gt_imgs(gt_img_idx, img_list, metadata, n_img=5) -> list:
    gt_imgs = [img_list[x] for x in sorted(gt_img_idx)]
    return gt_imgs


def get_single_img_idx(depth, gt_idx, ctx_size, n_img_token, img_list):
    num_ctx_img = int(ctx_size * 1000) // n_img_token

    stride = num_ctx_img // 5

    start_offset = [gt_idx-(i*stride) for i in range(5)]
    end_offset = [x+num_ctx_img for x in start_offset]

    return start_offset[depth], end_offset[depth]


def get_multi_img_idx(depth, gt_idx_list, ctx_size, n_img_token, img_list):
    num_ctx_img = int(ctx_size * 1000) // n_img_token

    if len(gt_idx_list) > num_ctx_img:
        return []

    candi_index = [x for x in range(min(gt_idx_list)+1, max(gt_idx_list))]
    candi_index = list(set(candi_index) - set(gt_idx_list))

    if len(candi_index) == 0 and gt_idx_list[0]+1 == gt_idx_list[1]:
        start, end = get_single_img_idx(depth, gt_idx_list[0], ctx_size, n_img_token, img_list)
        return [x for x in range(start, end)]

    if len(candi_index) < num_ctx_img - len(gt_idx_list):
        need_more = num_ctx_img - len(gt_idx_list)
        candi_index.extend([max(candi_index) + i for i in range(1, need_more+1) if (max(candi_index) + i) < len(img_list)])
        candi_index.extend([min(candi_index) - i for i in range(1, need_more+1) if (min(candi_index) - i) >= 0])

    if num_ctx_img - len(gt_idx_list) < 0:
        return []
    n_sample = min(len(set(candi_index)), num_ctx_img - len(gt_idx_list))
    
    if len(set(candi_index)) < num_ctx_img - len(gt_idx_list):
        return []
    candi_img_idx = random.sample(list(set(candi_index)), num_ctx_img - len(gt_idx_list))
    candi_img_idx.extend(gt_idx_list)
    candi_img_idx = list(set(candi_img_idx))

    return sorted(candi_img_idx)


def get_score(lm_response, gt_ans):
    if len(gt_ans) > 1: # multiple answer case; give partial credit for each answer.
        llm_ans = [x.replace("and", "").strip() for x in lm_response.split(",")]
        n_true = len(gt_ans)
        tp = 0
        for ai in gt_ans:
            for lai in llm_ans:
                match = is_match(lai, ai)
                if match:
                    tp += 1
        match = tp/n_true
    else: # single answer case
        match = is_match(lm_response, gt_ans[0])
        match = 1.0 if match else 0.0

    return match


def load_qa_data(traj_id, metadata_dir):
    img_dir = os.path.join(metadata_dir, traj_id, 'img')

    if not os.path.exists(img_dir):
        img_dir = os.path.join(metadata_dir, traj_id, 'img_768')

    metadata_path = os.path.join(metadata_dir, traj_id, 'metadata.json')
    traj_text_path = os.path.join(metadata_dir, traj_id, 'traj.txt')

    img_list = []
    img_path_list = []
    # iterate over images in img_dir and load PIL and append to img_list
    for img_file in sorted(os.listdir(img_dir)):
        if img_file.endswith(".png"):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            img_list.append(img)
            img_path_list.append(img_path)

    metadata = json.loads(open(metadata_path, 'r').read())
    traj_text = open(traj_text_path, 'r').read()

    return img_list, metadata, traj_text, img_path_list


def is_match(llm_ans, gt_ans, threshold=80) -> bool:
    '''
    Answer Matching

    We use the following flexible answer matching methods:
    - Exact matches
    - Fuzzy string matching (80% similarity threshold)
    - Synonym matching for common objects (see `SYNONYM_MAP` in code)
    - Special handling for yes/no questions
    - Support for multiple correct answers with partial credit

    Args:
        llm_ans (str): The answer generated by the LLM.
        gt_ans (str): The ground truth answer.
        threshold (int): The similarity threshold for fuzzy string matching.

    Returns:
        bool: True if the answer matches the ground truth, False otherwise.
    '''
    llm_norm = normalize_answer(llm_ans)
    gt_norm = normalize_answer(gt_ans)

    # 2. Exact or fuzzy match
    if llm_norm == gt_norm:
        return True
    if fuzz.ratio(llm_norm, gt_norm) >= threshold:
        return True

    # 3. Synonym fallback
    if are_synonyms(llm_norm, gt_norm):
        return True

    # Case 3: For yes/no questions
    if gt_norm in ["yes", "no"]:
        if 'yes' in llm_norm[:4]:
            llm_norm = 'yes'
        if 'no' in llm_norm[:3]:
            llm_norm = 'no'
        return llm_norm == gt_norm

    return False
    

def build_haystack(ctx_size, depth, gt_img_idx, n_img_token, img_list):
    if len(gt_img_idx) == 1:
        # single case
        start_idx, end_idx = get_single_img_idx(depth, gt_img_idx[0], ctx_size, n_img_token, img_list)

        if start_idx < 0 or end_idx > len(img_list): # out of range
            return [], []

        ctx_img_list = img_list[start_idx:end_idx]
        return ctx_img_list, range(start_idx, end_idx)
    else:
        if len(gt_img_idx) == 2 and gt_img_idx[0]+1 == gt_img_idx[1]: 
            # put object case.
            img_idx_list = get_multi_img_idx(depth, gt_img_idx, ctx_size, n_img_token, img_list)
        else:
            # multi-needle in the haystack: multi-depth case.
            img_idx_list = get_multi_img_idx(-1, gt_img_idx, ctx_size, n_img_token, img_list)

        if len(img_idx_list) == 0:
            return [], []

        if img_idx_list[0] < 0 or img_idx_list[-1] > len(img_list): # out of range
            return [], []

        ctx_img_list = [img_list[x] for x in img_idx_list if x < len(img_list)]
        
        return ctx_img_list, [x for x in img_idx_list if x < len(img_list)]
