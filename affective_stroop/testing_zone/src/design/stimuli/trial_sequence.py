from sweetpea import *
import json

import os


def trial_sequence(length: int) -> str:
    """
    Generates full JS experiment code based on trial length.

    Args:
        length (int): Minimum number of trials per phase.

    Returns:
        str: A full JS experiment as a string (ready to send to Firebase).
    """
    # Word stimuli (now neutral vs emotional words  -> ADD MORE VARIETY IN THE LIST
    neutral_words = [
        "chair", "book", "table", "pencil", "lamp",
        "window", "bottle", "clock", "paper", "shoe",
        "cup", "door", "tree", "shirt", "hat",
        "car", "road", "phone", "pen", "bag"
    ]

    emotional_words = [
        "death", "anger", "fear", "pain", "grief",
        "cry", "murder", "sad", "blood", "hurt",
        "loss", "terror", "abuse", "suffer", "stress",
        "lonely", "hate", "danger", "trauma", "rage"
    ]


    ### Factors

    color = Factor("color", ["red", "blue", "green", "yellow"])

    word = Factor("word", neutral_words + emotional_words)

    ### Valence Factory (derived factors)

    def is_emotional(word):
        return word in emotional_words

    def is_neutral(word):
        return word in neutral_words

    valence = Factor("valence", [
        DerivedLevel("emotional", WithinTrial(is_emotional, [word])),
        DerivedLevel("neutral",   WithinTrial(is_neutral,   [word]))
    ])



    #trial length passed into the function
    base_constraints = [MinimumTrials(length)]

    # BLOCK STRUCTURE
    # Neutral block
    block_neutral = CrossBlock(
        [color, word, valence],
        [color],
        base_constraints + [Exclude(valence["emotional"])]
    )

    # Emotional block
    block_emotional = CrossBlock(
        [color, word, valence],
        [color],
        base_constraints + [Exclude(valence["neutral"])]
    )

    # Neutral block again
    block_neutral2 = CrossBlock(
        [color, word, valence],
        [color],
        base_constraints + [Exclude(valence["emotional"])]
    )

    # Synthesize trials
    exp1 = synthesize_trials(block_neutral,   1, CMSGen)
    exp2 = synthesize_trials(block_emotional, 1, CMSGen)
    exp3 = synthesize_trials(block_neutral2,  1, CMSGen)

    # Combine sequentially
    experiments = [
        ("neutral1", exp1[0]),
        ("emotional", exp2[0]),
        ("neutral2", exp3[0])
    ]

    # correct Format for jsPsych
    trials_for_jspsych = []

    for phase_label, block_data in experiments:
        n_trials = len(block_data["color"])     # Assuming all fields are same length
        for i in range(n_trials):
            trial = {
                "color": block_data["color"][i],
                "word": block_data["word"][i],
                "valence": block_data["valence"][i],
                "phase": phase_label
            }
            trials_for_jspsych.append(trial)


    # Get the absolute path to the project's root (2 levels up from this script) & then public
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", ".."))
    public_folder = os.path.join(project_root, "public")
    output_path = os.path.join(public_folder, "stroop_trials.json")

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(trials_for_jspsych, f, indent=2)

    print(f"✅ trials.json successfully saved to: {output_path}")

    return trials_for_jspsych
