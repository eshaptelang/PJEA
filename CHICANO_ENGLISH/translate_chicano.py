import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ds = load_dataset("google/boolq", split="train")
df = ds.to_pandas()
df.insert(loc=1, column='question_translated', value=None)
df.insert(loc=4, column='passage_translated', value=None)

df = df [["question", "question_translated", "passage", "passage_translated", "answer"]]

few_shot_prompt = """
    You are a native speaker of Chicano English and you task is to rewrite a given standard american english passage into chicano.

    The rewrite shouldn't be a literal translation, but instead a cultural and linguisitic transformation that incorporates elements of Chicano vernacular, grammar and tone.
    Preserve meaning, but change sentence structure, word choice, and idiomatic expressions to match chicano english style. Prioritize fluency over strict feature matching

    Here are a few examples for reference. In the examples, a feature within chicano is given, and then an example of how it is used is shown.
    F is the feature in the language, SAE is Standard American English, CE is Chicano English.

    F: Multiple negation / negative concorda
    SAE: He won't cause any harm
    CE: He won't do no harm

    F: Me instead of I in coordinate subjects
    SAE: My brother and I
    CE: My brother and me

    F: Benefactive "personal dative" construction
    SAE: I got a new car; She got a new car; They got a new car
    CE: I got me a new car; She got her a new car; They got them a new car

    F: Forms or phrases for the second person plural pronoun other than you 
    SAE: 'you guys'
    CE: 'all of you'

    Proximal and distal demonstratives with 'here' and 'there'
    SAE: those books there
    CE: them there books

    Levelling of the difference between present perfect and simple past: simple past for StE present perfect
    SAE: Have you ever been in London
    CE: Were you ever in London

    F: Loosening of sequence of tenses rule
    SAE: I noticed the van I had come in
    CE: I noticed the van I came in

    F: Double modals
    SAE: I tell you what we should do
    CE: I tell you what we might should do

    F: New quasi-modals: aspectual meanings
    SAE: They're about to leave town
    CE: They're fixin' to leave town

    F: New quasi-modals: aspectual meanings
    SAE: I'm about to go
    CE: I'm finna go

    F: New quasi-modals: aspectual meanings
    SAE: It used to not matter whether you walked in late or not
    CE: It useta didn't matter whether you walked in late or not

    F: Levelling of past tense/past participle verb forms: regularization of irregular verb paradigms
    SAE: caught
    CE: catched

    F: Levelling of past tense/past participle verb forms: regularization of irregular verb paradigms
    SAE: spoken
    CE: spoke

    F: Levelling of past tense/past participle verb forms: past tense replacing the past participle
    SAE: he went
    CE: he had went

    F: Was for conditional were
    SAE: if I were you
    CE: if I was you

    F: Ain't as the negated form of be
    SAE: They're all in there, aren't they
    CE: They're all in there, ain't they

    F: Invariant don't for all persons in the present tense
    SAE: He doesn't like me
    CE: He don't like me

    F: Existential / presentational there's/there is/there was with plural subjects
    SAE: There are two men waiting in the hall
    CE: There's two men waiting in the hall

    F: Variant forms of dummy subject there in existential clauses
    SAE: There is something bad wrong with her
    CE: It's something bad wrong with her

    F: Variant forms of dummy subject there in existential clauses
    SAE: There's a new person here
    CE: It's a new person here

    F: 	Relativizer that or what in non-restrictive contexts
    SAE: My daughter, who lives in london
    CE: My daughter, that lives in london

    F: Degree modifier adverbs have the same form as adjectives
    SAE: That's really good
    CE: That's real good

    F: Inverted word order in indirect questions
    SAE: I'm wondering what you are going to do
    CE: I'm wondering what are you gonna do

    F: Adverbs and prepositions
    SAE: Come fast
    CE: Come quick

    F: Like as a focussing device
    SAE: How did you get away with that
    CE: How did you get away with that like

    F: Like as a quotative particle
    SAE: And she said 'What do you mean?'
    CE: And she was like 'What do you mean?'

    F: Emphatic reflexives with own
    SAE: Everybody took care of themselves
    CE: Everybody took care of their own self

    F: Subject pronoun drop: referential pronouns
    SAE: A: Do you have tickets? B: No, I sold them already
    CE: A: You got tickets? B: No, sold already

    F: Absence of plural marking only after quantifiers
    SAE: We did all our subjects in English
    CE: We did all our subject in English

    F: Invariant be as permanent marker
    SAE: Mandarin is the national language of China
    CE: Mandarin be the national language of China

    F: Zero past tense forms of regular verbs
    SAE: I walked
    CE: I walk

    F: Deletion of auxiliary be: before gonna
    SAE: I am going to go work
    CE: I gonna go work

    F: Which for 'who'
    SAE: My brother, who
    CE: My brother, which

    F: Gapping/zero-relativization in subject position
    SAE: The man who lives there is a nice chap
    CE: Them man who lives there is a nice chap
"""

def translate_text(text, source_dial="English", target_dial="Chicano English"):
    full_prompt = few_shot_prompt + f"\nSAE: {text}\nCE:"
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"{full_prompt}"},
        ],
        temperature = 0.3
    )
    return response.choices[0].message.content.strip()

for i, row in df.iterrows():
    # Translate question if not done
    if pd.isna(row["question_translated"]):
        question_translation = translate_text(row["question"])
        df.at[i, "question_translated"] = question_translation

    # Translate passage if not done
    if pd.isna(row["passage_translated"]):
        passage_translation = translate_text(row["passage"])
        df.at[i, "passage_translated"] = passage_translation

    # Save progress every 10 rows
    if (i + 1) % 10 == 0:
        df.to_excel("CHICANO_ENGLISH/Chicano_boolq.xlsx", index=False)

df.to_excel("CHICANO_ENGLISH/Chicano_boolq.xlsx", index=False)
