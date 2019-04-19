from cssi.questionnaire import SSQ

PRE = {
    "blurredVision": 1,
    "burping": 2,
    "difficultyConcentrating": 3,
    "difficultyFocusing": 1,
    "dizzyEyesClosed": 1,
    "dizzyEyesOpen": 0,
    "eyestrain": 0,
    "fatigue": 1,
    "fullnessOfHead": 0,
    "generalDiscomfort": 0,
    "headache": 1,
    "increasedSalivation": 2,
    "nausea": 2,
    "stomachAwareness": 0,
    "sweating": 1,
    "vertigo": 0
}

POST = {
    "blurredVision": 1,
    "burping": 2,
    "difficultyConcentrating": 3,
    "difficultyFocusing": 2,
    "dizzyEyesClosed": 1,
    "dizzyEyesOpen": 0,
    "eyestrain": 0,
    "fatigue": 1,
    "fullnessOfHead": 3,
    "generalDiscomfort": 0,
    "headache": 1,
    "increasedSalivation": 2,
    "nausea": 2,
    "stomachAwareness": 3,
    "sweating": 1,
    "vertigo": 0
}

ssq = SSQ(pre=PRE, post=POST)
ssq.score()
