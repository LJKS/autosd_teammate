
import { initJsPsych } from "jspsych";
import "jspsych/css/jspsych.css";
import htmlKeyboardResponse from "@jspsych/plugin-html-keyboard-response";

const main = async (id, condition_json) => {
  const jsPsych = initJsPsych();

  // Load trials (SweetPea-generated)
 // const response = await fetch("/stroop_trials.json");
 // const trialList = await response.json();

  //the conditions were passed in the worklfow runner on state (not actually concitions but our experiment code for the main trials)
  const trialList = condition_json['experiment_code'];
  console.log("🔍 trialList / condition_json:", trialList);


  // Constants
  const FIXATION_DURATION = 800;
  const SOA_DURATION = 400;
  const STIMULUS_DURATION = 10000;
  const FEEDBACK_DURATION = 800;

  const keyToColor = {
    c: "red",
    d: "blue",
    n: "green",
    j: "yellow",
  };

  // Instruction Block
  const instructions = [
    {
      type: htmlKeyboardResponse,
      stimulus:
        "In this experiment you must respond to the COLOR of the word, not its meaning.<br><br>Press SPACE to continue.",
      choices: [" "],
    },
    ...Object.entries(keyToColor).map(([k, col]) => ({
      type: htmlKeyboardResponse,
      stimulus: `If the color is <span style="color:${col};">${col.toUpperCase()}</span>, press the key: ${k}<br><br>Press ${k} to continue.`,
      choices: [k],
    })),
    {
      type: htmlKeyboardResponse,
      stimulus:
        "A short Training phase will now begin.<br><br>Press SPACE to continue.",
      choices: [" "],
    },
    {
      type: htmlKeyboardResponse,
      stimulus: `
        Welcome to the practice phase!<br><br>
        Here is a little reminder:<br><br>
        'c' → <span style="color:red;">Red</span><br>
        'd' → <span style="color:blue;">Blue</span><br>
        'n' → <span style="color:green;">Green</span><br>
        'j' → <span style="color:yellow;">Yellow</span><br><br>
        Press SPACE to start.
      `,
      choices: [" "],
    },
  ];
  // Practice trials (fixed set)
  const practiceTrials = Object.entries(keyToColor).flatMap(([k, col]) => [
    // Actual trial
    {
      type: htmlKeyboardResponse,
      stimulus: `<div style="color:${col}; font-size:48px;">book</div>`,
      choices: [k],
      trial_duration: null,
      response_ends_trial: true,
      on_finish: (data) => {
        data.correct = data.response === k;
        data.color = col;
        data.word = "BOOK";
        data.valence = "neutral";
        data.phase = "practice";
      },
    },

    // Feedback trial
    /*{
      type: htmlKeyboardResponse,
      stimulus: () => {
        const last = jsPsych.data.getLastTrialData().values()[0];
        return last.correct ? "<span style='color:green;'>CORRECT</span>" 
                            : "<span style='color:red;'>WRONG</span>";
      },
      trial_duration: 800,
      choices: "NO_KEYS",
    }*/
  ]);


  // Pause screen
  const pause = {
    type: htmlKeyboardResponse,
    stimulus: "Take a little break.<br><br>Press SPACE to continue.",
    choices: [" "],
  };

  // Stroop block using timeline_variables
  const stroopBlock = {
    timeline: [
      {
        type: htmlKeyboardResponse,
        stimulus: "+",
        trial_duration: FIXATION_DURATION,
        choices: "NO_KEYS",
      },
      {
        type: htmlKeyboardResponse,
        stimulus: "",
        trial_duration: SOA_DURATION,
        choices: "NO_KEYS",
      },
      {
        type: htmlKeyboardResponse,
        stimulus: () =>
          `<div style="color:${jsPsych.timelineVariable("color")}; font-size:48px;">${jsPsych.timelineVariable("word")}</div>`,
        choices: Object.keys(keyToColor),
        trial_duration: null,
        response_ends_trial: true,
        data: {
          word: jsPsych.timelineVariable("word"),
          color: jsPsych.timelineVariable("color"),
          valence: jsPsych.timelineVariable("valence"),
          phase: jsPsych.timelineVariable("phase"),
        },
        on_finish: (data) => {
          console.log("Response:", data.response); 
          data.correct = keyToColor[data.response] === data.color;
        },
      },
      /* optional feedback
      {
        conditional_function: () => jsPsych.timelineVariable("phase") === "training",
        timeline: [{
          type: htmlKeyboardResponse,
          stimulus: () => {
            const correct = jsPsych.data.getLastTrialData().trials[0].correct;
            return correct ? "CORRECT" : "WRONG";
          },
          trial_duration: FEEDBACK_DURATION,
        }]
      }
      */
    ],
    timeline_variables: trialList, // << here you pass all trials
  };

  // Build full experiment timeline
  const timeline = [];
  timeline.push(...instructions);
  timeline.push(...practiceTrials);
  timeline.push(pause);
  timeline.push(stroopBlock);

  // Run
  await jsPsych.run(timeline);

  // const nonPractice = jsPsych.data.get().filterCustom((d) => d.phase !== "practice");
  // const acc = nonPractice.filter({ correct: true }).count() / nonPractice.count();
  // const mean_rt = nonPractice.select("rt").mean();

  return JSON.stringify({
    id,
    n_trials: trialList.length,
    //accuracy: acc,
    trials: jsPsych.data.get().values(),
  });
};

export default main;