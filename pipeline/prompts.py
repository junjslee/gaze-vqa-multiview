# prompts.py
"""Prompt templates for all tasks.

Sections (in order):
1) Helpers (string normalization / person reference)
2) Task 1 prompts (gaze target labeling)
3) Shared object-visibility prompts (used in Task 3/4)
4) Task 2 prompts (relative camera rotation)
5) Task 3 prompts (per-view visibility)
6) Task 4 prompts (line-of-sight / verification)
7) Shared utilities (multi-choice helper)
"""


# =============================================================================
# Helpers
# =============================================================================

def lc(s):
    """Lowercase helper used to normalize prompt text (None -> empty)."""
    if s is None:
        return ""
    return str(s).strip().lower()


_CLOTHING_NOUNS = {
    "shirt", "tshirt", "t-shirt", "sweater", "hoodie", "jacket", "coat", "vest",
    "pants", "jeans", "shorts", "skirt", "dress", "leggings",
    "shoes", "sneakers", "sandals", "slippers", "boots", "flip-flops",
    "hat", "cap", "beanie", "glasses", "sunglasses", "scarf", "tie",
    "backpack", "bag", "purse", "apron",
}


def _format_person_desc(person_desc):
    """Normalize a clothing-only description into a short readable phrase."""
    s = lc(person_desc)
    if not s:
        return "person"
    for prefix in ("a ", "the "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    for prefix in ("person ", "man ", "woman ", "boy ", "girl "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    for prefix in ("wearing ", "with "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    tokens = s.split()
    last_idx = -1
    for i, t in enumerate(tokens):
        if t in _CLOTHING_NOUNS:
            last_idx = i
    if last_idx >= 0:
        tokens = tokens[:last_idx + 1]
    groups = []
    cur = []
    for t in tokens:
        cur.append(t)
        if t in _CLOTHING_NOUNS:
            groups.append(" ".join(cur))
            cur = []
    if cur:
        if groups:
            groups[-1] = groups[-1] + " " + " ".join(cur)
        else:
            groups.append(" ".join(cur))
    if len(groups) == 1:
        return groups[0]
    if len(groups) == 2:
        return f"{groups[0]} and {groups[1]}"
    return ", ".join(groups[:-1]) + f", and {groups[-1]}"


def person_ref(person_desc):
    """Return a natural-language person reference used in prompts."""
    pretty = _format_person_desc(person_desc)
    if not pretty or pretty == "person":
        return "the person"
    return f"the person wearing {pretty}"


# =============================================================================
# Task 1 prompts (gaze target labeling)
# =============================================================================

def prompt_person_descriptor(scene_type=None):
    """Task1: describe the person's clothing for later use in prompts."""
    return (
        "Describe this person briefly using apperance (clothing) only (e.g., blue shirt green pants).\n"
        "Output 4-10 words. No punctuation."
    )


def prompt_target_description_ray(person_desc, anchor_cam, scene_type=None):
    """Task1: describe gaze target using a ray+arrow overlay image."""
    who = person_ref(person_desc)
    return (
        f"In {lc(anchor_cam)}, a red line indicates the gaze direction of {lc(who)} and points to the target with a small red arrowhead near the target location.\n"
        "The arrow marker is intentionally offset a little from the exact target point to avoid covering the object.\n"
        "Describe the specific thing they are looking at in 1-2 sentences.\n"
        "Use 'on' ONLY if the object is clearly resting on a surface (visible contact) and realistic. "
        "If it only appears aligned in 2D but is farther away (e.g., a wall-mounted air conditioner above a foosball table), do NOT say 'on'.\n"
        "If the marker points to a small item resting on a larger item as a surface, describe it as '<small> on <large>' (e.g., 'book on table').\n"
        "Be concrete (type + color/material/edge shape/size if visible).\n"
        "If it is furniture, just name the type (table/chair/stool/armchair/sofa/desk/bookshelf/cabinet/etc).\n"
        "Do NOT mention the red arrow marker or red line."
    )


def prompt_distill_object_phrase(target_description, scene_type=None):
    """Task1: distill a short noun phrase from a longer description."""
    return (
        "Extract the object name as a SHORT noun phrase from the description below.\n"
        "Rules:\n"
        "- Output ONLY the noun phrase\n"
        "- 1 to 4 words\n"
        "- No verbs, no 'the'\n"
        "- Avoid generic terms like 'furniture' or 'object', 'thing', 'square/box'\n"
        "- Do not mention dots, arrows, lines, rays, or overlays\n"
        "- If the description is ambiguous, output your best concrete guess\n"
        "- Use 'on' ONLY if contact is visually obvious; avoid 'on' for 2D alignment (e.g., wall AC above a table), no vague objects should be mentioned even if it's on it\n"
        "- If a small item (clearly distinguishable) sits on a larger surface, you may return '<small> on <large>'\n"
        "- Examples: 'piano', 'kitchen sink', 'red mug', 'book on table', 'plastic plate', 'glass plate', 'blue cup','white bottle', 'metal faucet'\n\n"
        f"Description: {lc(target_description)}"
    )


def prompt_masked_object(scene_type=None):
    """Task1: identify object from a cropped mask (optionally overlayed)."""
    return (
        "You see a target object crop.\n"
        "If two images are provided: image A is the raw crop; image B is the same crop with a semi-transparent mask overlay.\n"
        "If multiple objects appear, name the one highlighted by the mask.\n"
        "Return ONLY a short noun phrase naming the object.\n"
        "Rules: 1-4 words. No verbs. No punctuation.\n"
        "Use 'on' ONLY if the object is clearly resting on a surface (visible contact). "
        "If it only appears aligned in 2D but is farther away (e.g., a wall-mounted air conditioner above a foosball table), do NOT say 'on'.\n"
        "If a small item (clearly distinguishable) sits on a larger surface, you may answer '<small> on <large>'.\n"
        "Be specific (avoid generic terms like 'furniture' or 'object', 'thing', 'square/box').\n"
        "Use relative background and setting information (if possible, then run chain thoughts) to help identify the object).\n"
        "Examples: 'piano', 'kitchen sink', 'red mug', 'book on table', 'plastic plate', 'glass plate', 'blue cup', 'white bottle', 'metal faucet'."
    )


def prompt_masked_object_detailed(scene_type=None):
    """Task1/Task4: a slightly richer masked-object label prompt."""
    return (
        "You see a target object crop.\n"
        "Return ONLY a short noun phrase naming the object.\n"
        "Rules: 2-5 words. No verbs. No punctuation.\n"
        "Include color/material if clearly visible (e.g., 'red mug', 'wooden chair', 'plastic plate', 'glass plate', 'blue cup', 'white bottle', 'metal faucet').\n"
        "Avoid broad surfaces like floor/ceiling/wall and generic patterns/textures.\n"
        "Be specific (avoid generic terms like 'object', 'thing', 'pattern', 'fabric')."
    )


def prompt_judge_same_object_phrase(label_a, label_b, scene_type=None):
    """Task1: judge if two labels refer to the same physical object."""
    return (
        "You are validating dataset labels (not judging the exact match).\n"
        f"Phrase A: \"{lc(label_a)}\"\n"
        f"Phrase B: \"{lc(label_b)}\"\n"
        "In terms of semantics and relative applicability, do these refer to the SAME physical object?\n"
        "Answer only:\n"
        "YES; CANONICAL: <1-4 words>\n"
        "or\n"
        "NO"
    )


def prompt_reconcile_triple(ray_label, mask_label, dot_label, ray_desc=None, dot_desc=None, scene_type=None):
    """Task1: reconcile ray/mask/dot labels into one canonical phrase."""
    prompt = (
        "You are consolidating three independently generated object labels into ONE canonical noun phrase.\n"
        "Return ONLY a 1-4 word noun phrase. No verbs. No punctuation.\n"
        "Use 'on' ONLY if contact is visually obvious; avoid 'on' for 2D alignment.\n"
        "If one label is a small item on a larger surface, you may return '<small> on <large>'.\n"
        "Do NOT mention dots, lines, rays, or overlays.\n"
        f"Label from ray cue: {lc(ray_label)}\n"
        f"Label from mask cue: {lc(mask_label)}\n"
        f"Label from dot cue: {lc(dot_label)}\n"
    )
    if ray_desc:
        prompt += f"\nRay description: {lc(ray_desc)}\n"
    if dot_desc:
        prompt += f"Dot description: {lc(dot_desc)}\n"
    return prompt


def prompt_on_relation_plausibility(label, scene_type=None):
    """Task1: decide if 'A on B' is physically plausible (not just 2D alignment)."""
    return (
        "Decide if this relation is physically plausible (object A is truly resting on object B), "
        "not just aligned in 2D due to depth.\n"
        f"Phrase: '{lc(label)}'\n"
        "Example of NOT plausible: 'air conditioner on foosball table' if the AC is wall-mounted above it.\n"
        "Answer YES if plausible contact, otherwise NO. If unsure, answer NO."
    )


def prompt_task1_reasoning_rich(person_desc, canonical_object, scene_type=None):
    """Task1: VLM reasoning prompt (spatial explanation)."""
    return (
        f"Write 1-2 sentences explaining why the {lc(person_desc)} is looking at the '{lc(canonical_object)}'. "
        "Ground the explanation in visible spatial cues: head direction, body orientation, left/right/behind, "
        "and relative placement of the object.\n"
        "Do not mention overlays, gaze rays, masks, annotations, or ground-truth sources."
    )


def prompt_task1_pose_check(person_desc, obj_label, scene_type=None):
    """Task1: yes/no/unclear pose check against the target object."""
    who = person_ref(person_desc)
    return (
        f"In this image, is {lc(who)} actually looking at the '{lc(obj_label)}'?\n"
        "Use head/torso orientation and spatial position. If unclear, answer UNCLEAR.\n"
        "Answer ONLY with YES, NO, or UNCLEAR."
    )


def prompt_task1_semantic_arbiter(
    person_desc,
    anchor_cam,
    candidate_labels,
    scene_type=None,
    mask_area_ratio=None,
    ray_available=True,
):
    """
    Task1: disagreement-focused semantic arbiter.
    Uses multi-cue labels + scene context to pick a final object phrase.
    """
    who = person_ref(person_desc)
    lines = []
    for key, val in (candidate_labels or {}).items():
        if val:
            lines.append(f"- {key}: {lc(val)}")
    cand_block = "\n".join(lines) if lines else "- none"
    mar_txt = "N/A" if mask_area_ratio is None else f"{float(mask_area_ratio):.4f}"
    ray_txt = "available" if ray_available else "not available"
    return (
        "You are resolving conflicting gaze-target labels for dataset quality.\n"
        "Interpret the images as follows:\n"
        "1) segmented mask crop at gaze target\n"
        "2) same crop with target dot\n"
        "3) cue-rich full image (person bbox + gaze ray + mask)\n"
        "4) full raw scene context\n"
        "Goal: return the most accurate concrete object identity that "
        f"{lc(who)} is looking at in {lc(anchor_cam)}.\n"
        "Prefer full object identity over partial fragments. "
        "Use scene context to disambiguate tiny patches.\n"
        "Avoid generic words ('object', 'thing', 'furniture') and avoid mentions of overlays/rays/dots.\n"
        "Use 'on' only when true physical support is visually clear.\n"
        f"Mask area ratio: {mar_txt}. Ray cue: {ray_txt}.\n"
        "Candidate labels:\n"
        f"{cand_block}\n\n"
        "Think step-by-step internally, but output ONLY these lines:\n"
        "FINAL_LABEL: <1-4 words>\n"
        "DECISION: <KEEP|SWITCH_MASK|SWITCH_DOT|SWITCH_RAY|SWITCH_MV|REFINE|UNSURE>\n"
        "CONFIDENCE: <HIGH|MEDIUM|LOW>\n"
        "RATIONALE: <max 14 words>"
    )


def prompt_task1_question(person_desc, scene_type=None):
    """Task1: final QA question for the benchmark."""
    who = person_ref(person_desc)
    return f"Using all camera views, what is {who} looking at?"


# =============================================================================
# Shared object-visibility prompts (Task 3 / Task 4)
# =============================================================================

def prompt_object_visible_yesno(obj_label, scene_type=None):
    """Shared: ask if an object is visible in a single image."""
    return f"Target object: {lc(obj_label)}. Is it clearly visible in this image?"


def prompt_object_tangible_yesno(obj_label, scene_type=None):
    """Shared: check if label is a tangible object (not a surface/texture)."""
    return (
        f"Target object: {lc(obj_label)}. Is this a discrete, tangible object "
        "rather than a broad surface/pattern/texture (e.g., floor, wall, ceiling, grid, pattern)?"
    )


def prompt_object_in_scene_desc(obj_label, scene_type=None):
    """Shared: describe object if visible; otherwise return NONE."""
    return (
        f"In this image, look for the object described as '{lc(obj_label)}'. "
        "If it is visible, describe it in 2-5 words (include color/material if visible). "
        "If it is not visible, output ONLY 'NONE'."
    )


# =============================================================================
# Task 2 prompts (relative camera rotation)
# =============================================================================

def prompt_task2_body_orientation(cam_name, scene_type=None):
    """Task2: describe body-facing direction in a single view."""
    return (
        f"This is {lc(cam_name)} view. Describe the person's body-facing direction qualitatively.\n"
        "Include front/back and left/right cues when visible (e.g., face visible, back visible, left shoulder forward).\n"
        "Avoid generic 'facing forward' unless it is clearly frontal. Prefer 'slight right turn', 'mostly back', etc.\n"
        "Use only visible body cues; do not rely on background or lighting.\n"
        "If unclear, say 'unclear'. Return one short sentence."
    )


def prompt_task2_question(cam1, cam2, scene_type=None):
    """Task2: full question asking relative rotation between two camera views."""
    return (
        "Two images show the same person from Camera 1 and Camera 2.\n"
        "Camera 1 is the first image. Camera 2 is the second image.\n\n"
        "Your task is to estimate Camera 2's rotation relative to Camera 1, based only on how the person's visible orientation changes across the two views.\n"
        "1. Infer the body-facing direction in each image (describe qualitatively).\n"
        "2. Based on the difference, estimate how many degrees Camera 2 is rotated around the person relative to Camera 1 (roughly 0–180°).\n"
        "3. State whether Camera 2 is clockwise or counterclockwise relative to Camera 1.\n\n"
        "Output format:\n"
        "Camera 1 view: <your description>\n"
        "Camera 2 view: <your description>\n"
        "Relative rotation: <~N° clockwise/counterclockwise>\n"
        "Do not rely on background or lighting; use only the visible body orientation."
    )


# =============================================================================
# Task 3 prompts (per-view visibility)
# =============================================================================

def prompt_task3_question(person_desc, obj_label, scene_type=None):
    """Task3: question asking visibility per camera view."""
    who = person_ref(person_desc)
    return (
        f"The gaze target is '{lc(obj_label)}'. For each camera view, is it visible to {lc(who)}? "
        "Answer YES/NO per view."
    )


# =============================================================================
# Task 4 prompts (line-of-sight / verification)
# =============================================================================

def prompt_task4_accessibility_yesno(obj_phrase, person_desc, scene_type=None):
    """Task4: yes/no prompt about line-of-sight visibility."""
    who = person_ref(person_desc)
    return (
        f"Decide if the {lc(obj_phrase)} is within the line of sight of {lc(who)} in this view.\n"
        "Use head/torso direction and spatial position to judge if the person could see it.\n"
        "Answer YES or NO only."
    )


def prompt_task4_verify(answer_yesno, obj_phrase, person_desc, explanation, scene_type=None):
    """Task4: verifier prompt to check explanation vs ground truth."""
    return (
        "You are a dataset verifier.\n"
        f"Ground-truth answer: {lc(answer_yesno)}\n"
        f"Queried object: {lc(obj_phrase)}\n"
        f"Person reference: {lc(person_desc)}\n\n"
        "Explanation:\n"
        f"\"{explanation}\"\n\n"
        "Question: Does this explanation make sense given the images and the ground-truth answer?\n"
        "If the explanation clearly matches the ground-truth visibility (e.g., 'visible' for YES, "
        "'not visible/behind' for NO), accept even if line-of-sight is not explicit.\n"
        "Respond ONLY with:\n"
        "PASS\n"
        "or\n"
        "FAIL"
    )


def prompt_task4_reasoning(answer_yesno, obj_phrase, person_desc, gaze_target=None, scene_type=None):
    """Task4: generate a short explanation for a YES/NO line-of-sight answer."""
    if gaze_target:
        if answer_yesno == "YES":
            fb = (
                f"Given the person's head/torso direction, they are looking toward the {lc(gaze_target)}, "
                f"so the queried object '{lc(obj_phrase)}' lies in front and within their forward field of view."
            )
        else:
            fb = (
                f"Given the person's head/torso direction, they are looking toward the {lc(gaze_target)}, "
                f"so the queried object '{lc(obj_phrase)}' is behind or to the side and outside their forward field of view."
            )
    else:
        fb = (
            f"Given the person's head/torso direction, the object '{lc(obj_phrase)}' is "
            f"{'in front of' if answer_yesno=='YES' else 'not in front of'} the person and "
            f"{'within' if answer_yesno=='YES' else 'outside'} their forward field of view."
        )

    prompt = (
        f"The answer is {lc(answer_yesno)}.\n"
        "Explain why based only on the pixels across the provided views.\n"
        f"Queried object: {lc(obj_phrase)}\n"
        "Person: the person\n"
        "Write 1-2 short sentences. Include a head/torso cue and a concrete spatial cue "
        "(left/right/front/behind, occlusion, distance). Also state whether the object is within the "
        "person's forward field of view. "
        "Do not add extra person details or unrelated objects. "
        "Do not mention annotations or ground-truth sources."
    )
    return prompt, fb


def prompt_task4_question(query_cam, obj_phrase, person_desc, scene_type=None):
    """Task4: final QA question asking line-of-sight visibility."""
    who = person_ref(person_desc)
    return (
        f"In {query_cam}, can {who} see the '{obj_phrase}' from their line of sight? "
        "Answer YES or NO and explain briefly using front/behind/occlusion."
    )


# =============================================================================
# Shared utilities
# =============================================================================

def prompt_choose_by_letter(question, choice_map, scene_type=None):
    """Utility: force a multi-choice answer as a single letter."""
    defs = " ".join([f"{k}={v}." for k, v in choice_map.items()])
    return (
        f"{question}\n"
        f"Answer with ONE LETTER only. {defs}\n"
        "Output ONLY the letter."
    )
