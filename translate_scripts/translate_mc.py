import pandas as pd
from datasets import load_dataset
from openai import OpenAI
import time
import os
from dotenv import load_dotenv
from dotenv import dotenv_values
import asyncio
import argparse
from tqdm import tqdm

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Will add batch processing later
    '''parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        required=False,
                        help='batch size'),'''
    parser.add_argument('--model_name',
                        type=str,
                        default='gpt-4o',
                        required=False,
                        help='Model name')
    parser.add_argument('--dataset',
                        type=str,
                        default='',
                        required=True,
                        help='Dataset')
    parser.add_argument('--output_dir',
                        type=str,
                        default='',
                        required=True,
                        help='Output Directory')
    parser.add_argument('--n_choices',
                        type=int,
                        default=0,
                        required=True,
                        help='Number of choices')
    parser.add_argument('--no_premise',
                        type=bool,
                        default=False,
                        required=False,
                        help='Is there a premise')
    parser.add_argument('--target_dial',
                        type=str,
                        default='',
                        required=True,
                        help='target dialect')
    parser.add_argument('--temp',
                        type=int,
                        default=0.3,
                        required=False,
                        help='model temperature')
    args = parser.parse_args()
    print(args)

    if (args.target_dial == 'AAVE' or 'African American Vernacular English'):
        few_shot_prompt = """
            Translate the following from Standard American English to African American Vernacular English (AAVE)

            SAE: My brother and I were late
            AAVE: My brother and me were late.

            SAE: He fixed the table himself
            AAVE: He fixed the table hisself.

            SAE: Do you all want to go?
            AAVE: Do y'all want to go?

            SAE: I am going to take the children to school.
            AAVE: I am going to take the childrens to school.

            SAE: The leaves were cut by knives.
            AAVE: The leafs were cut by knifes.

            SAE: I was with him and his group.
            AAVE: I was with him and dem.

            SAE: One of those things was pink.
            AAVE: One of them things was pink.

            SAE: That is so much easier to follow.
            AAVE: That is so much more easier to follow.

            SAE: He's the most regular kind of guy I know.
            AAVE: He is the regularest kind of guy I know.

            SAE: One of the prettiest sunsets.
            AAVE: One of the most pretty sunsets.

            SAE: He is always sick.
            AAVE: He be sick.

            SAE: I won't go anywhere.
            AAVE: Uh ain ga go nowhere.

            SAE: I'm about to go.
            AAVE: I'm finna go.

            SAE: She has done her homework.
            AAVE: She has did her homework.

            SAE: He won't do any harm.
            AAVE: He won't do no harm.

            SAE: They're all in there, aren't they?
            AAVE: They're all in there, ain't they?

            SAE: I ain't had a look at them yet.
            AAVE: I haven't had a look at them yet.

            SAE: He doesn't like me.
            AAVE: He don't like me.

            SAE: Then he comes in and acts like nothing happened.
            AAVE: Then he come in and act like nothing happened.

            SAE: There are two men waiting in the hall.
            AAVE: There is two men waiting in the hall.

            SAE: There is something seriously wrong with her.
            AAVE: They is something bad wrong with her.

            SAE: She is running late.
            AAVE: She running late.

            SAE: He is a good teacher.
            AAVE: He a good teacher.

            SAE: She is smart.
            AAVE: She smart.

            SAE: She is at home.
            AAVE: She at home.

            SAE: My cousin, who lives in Atlanta, is coming to visit.
            AAVE: My cousin, what live in Atlanta, coming to visit.

            SAE: When you on switch on the alarm system you press this button.
            AAVE: When you on the alarm system you press this button.

            SAE: My husband and I were late.
            AAVE: Me husband and myself were late.

            SAE: I got myself a new car.
            AAVE: I got me a new car.

            SAE: They built it themselves.
            AAVE: They built it theyselves.

            SAE: We made that decision ourselves, without any help.
            AAVE: We made that decision ourself, without no help.

            SAE: Us kids used to steal the candy like crazy.
            AAVE: Us kids used to pinch the sweets like hell.

            SAE: What did he take?
            AAVE: What-all did he take?

            SAE: That President has two Secretaries of States.
            AAVE: That President has two Secretary of States.

            SAE: He picked up some pieces of wood to start the fire.
            AAVE: He picked up some woods to start the fire.

            SAE: The door closed.
            AAVE: That door bin close.

            SAE: These shoes are tight.
            AAVE: Deez here shoes tight.

            SAE: It's the teacher's desk.
            AAVE: It the teacher desk.

            SAE: She is fighting harder than him.
            AAVE: She the fightingest one.

            SAE: Something has fallen down the sink.
            AAVE: There's something fallen down the sink.

            SAE: Did you eat what I sent you?
            AAVE: You don ate what I has sent you?

            SAE: She has already finished her homework.
            AAVE: She done finished her homework.

            SAE: If you love your enemies, they will eat you alive in this society.
            AAVE: If you love your enemies, they be done eat you alive in this society.

            SAE: I have been cutting the bread.
            AAVE: I been cut the bread.

            SAE: I noticed the van I had come in.
            AAVE: I noticed the van I came in.

            SAE: I will start school next month.
            AAVE: I would start school next month.

            SAE: I'll tell you what we might need to do.
            AAVE: I tell you what we might should do.

            SAE: This can't be true.
            AAVE: This mustn't be true.

            SAE: She has already gone to the store.
            AAVE: She done goed to the store already.

            SAE: She gave me the keys yesterday.
            AAVE: She give me the keys yesterday.

            SAE: I saw her yesterday.
            AAVE: I seen her yesterday.

            SAE: He talked to me yesterday.
            AAVE: He talk to me yesterday.

            SAE: We have a mess around here.
            AAVE: We has a muck round here.

            SAE: She is sick.
            AAVE: Shi stei sik.

            SAE: They are interested
            AAVE: They've got interested.

            SAE: She has gone already, hasn't she?
            AAVE: She gone already, ain't it?

            SAE: I have eaten my lunch.
            AAVE: I eaten my lunch.

            SAE: You were hungry but he was thirsty.
            AAVE: You were hungry but he were thirsty.

            SAE: The man who lives there is a nice chap.
            AAVE: The man lives there is a nice chap.

            SAE: This is the house which I painted yesterday.
            AAVE: This is the house which I painted it yesterday.

            SAE: You are not going to pass unless you get an 88, and some universities are not going to give those marks.
            AAVE: You are not going to pass unless you are going to get 88 which some universities are not going to give those marks.

            SAE: It's harder than you think.
            AAVE: It's harder than what you think it is.

            SAE: That's really good.
            AAVE: That's real good.

            SAE: He runs hard every day.
            AAVE: He run hard every day.

            SAE: I suffer too much.
            AAVE: I sofa tuu motch.

            SAE: Riding bikes is what I see them doing.
            AAVE: They ride bikes is what I see them do.

            SAE: There's an old house up here, but nobody lives in it.
            AAVE: There's an old house up here, but don't nobody live in it.

            SAE: I'm wondering what you are going to do.
            AAVE: I'm wondering what are you gonna do.

            SAE: What does he want?
            AAVE: What he wants?

            SAE: Do you get the point?
            AAVE: You get the point?

            SAE: The thing I like most is apples.
            AAVE: The most thing I like is apples.

            SAE: And she said, 'What do you mean?'
            AAVE: And she was like 'What do you mean?'

            SAE: The car doesn't want to start this morning.
            AAVE: She don't wanna start this morning.

            SAE: It's raining.
            AAVE: Thass rainen.

            SAE: I saw him at the store yesterday.
            AAVE: I seen 'em at the store yesterday.

            SAE: My mother, who is a primary school teacher, lived in California.
            AAVE: My mother, he's a primary school teacher, lived in California.

            SAE: Everybody took care of themselves.
            AAVE: Everybody took care of their own self.

            SAE: He was reading his book.
            AAVE: He was reading he book.

            SAE: It is their book.
            AAVE: It's they book.

            SAE: You can't come in here unless you pay your fare.
            AAVE: Yu kyaan kom in here unless yu pie yu fier.

            SAE: It was their book.
            AAVE: It was them book.

            SAE: Our George was a nice one.
            AAVE: Us George was a nice one.

            SAE: Those girls are always late.
            AAVE: Them girls always late.

            SAE: Those cookies? I ate them.
            AAVE: Them cookies? I ate it.

            SAE: Do you have tickets? No, I sold them already.
            AAVE: You got tickets? No, sold already.

            SAE: As I've made clear before, I'm going to talk about solutions, not problems.
            AAVE: As I made it clear before, I am going to talk about solutions, not problems.

            SAE: It is very hot outside today.
            AAVE: Is very hot outside today.

            SAE: My dad and his friends like to play tennis.
            AAVE: My Daddy gang like to play tennis.

            SAE: She has three kids.
            AAVE: She got three kid.

            SAE: My sister is a pretty girl.
            AAVE: My sister are pretty girl.

            SAE: The trees don't grow very tall up there.
            AAVE: The tree don't grow very tall up there.

            SAE: I saw him at the store yesterday.
            AAVE: I seen 'em at the store yesterday.

            SAE: My mother, who is a primary school teacher, lived in California.
            AAVE: My mother, he's a primary school teacher, lived in California.

            SAE: Everybody took care of themselves.
            AAVE: Everybody took care of their own self.

            SAE: He was reading his book.
            AAVE: He was reading he book.

            SAE: It is their book.
            AAVE: It's they book.

            SAE: You can't come in here unless you pay your fare.
            AAVE: Yu kyaan kom in here unless yu pie yu fier.

            SAE: It was their book.
            AAVE: It was them book.

            SAE: Our George was a nice one.
            AAVE: Us George was a nice one.

            SAE: Those girls are always late.
            AAVE: Them girls always late.

            SAE: Those cookies? I ate them.
            AAVE: Them cookies? I ate it.

            SAE: Do you have tickets? No, I sold them already.
            AAVE: You got tickets? No, sold already.

            SAE: As I've made clear before, I'm going to talk about solutions, not problems.
            AAVE: As I made it clear before, I am going to talk about solutions, not problems.

            SAE: It is very hot outside today.
            AAVE: Is very hot outside today.

            SAE: My dad and his friends like to play tennis.
            AAVE: My Daddy gang like to play tennis.

            SAE: She has three kids.
            AAVE: She got three kid.

            SAE: My sister is a pretty girl.
            AAVE: My sister are pretty girl.

            SAE: The trees don't grow very tall up there.
            AAVE: The tree don't grow very tall up there.

            SAE: They saw a green snake tangled around a tree.
            AAVE: They seen one green snake tangled round a tree.

            SAE: The girlfriend of the man I met is a real beauty.
            AAVE: The man I met's girlfriend is a real beauty.

            SAE: A long time ago he was my sister's husband.
            AAVE: Long time he was for my sister husband.

            SAE: The unemployment situation is much more severe than in Singapore.
            AAVE: The unemployment position is much severe than in Singapore.

            SAE: That's worse than what he did.
            AAVE: That's worse what he did.

            SAE: He loves his car more than his children.
            AAVE: He loves his car than his children.

            SAE: They would have more powder on their hands than in their faces.
            AAVE: They would have more powder on their hands and in their faces.

            SAE: He is one of the most radical students that you can ever find.
            AAVE: He is one of the radical students that you can ever find.

            SAE: What do you want?
            AAVE: What are you wanting?

            SAE: On my breaks, I usually go to the library, Chinatown, or the city.
            AAVE: On my holidays, I be usually going to the library, Chinatown, or the city.

            SAE: I drink three or four cups with a meal.
            AAVE: I drinks three and four cups to a meal.

            SAE: He is sick a lot.
            AAVE: He do be sick a lot.

            SAE: She is always acting like that.
            AAVE: She stap acting like that.

            SAE: When you are standing there you can see the flames.
            AAVE: When you're stood there you can see the flames.

            SAE: I have seen that movie.
            AAVE: I did see that movie.

            SAE: They've not left school yet.
            AAVE: They're not left school yet.

            SAE: That girl who smiled at me seems nice.
            AAVE: That girl what did smile at me seems nice.

            SAE: He had gone to the store before I called him.
            AAVE: He had go to the store before I called him.

            SAE: I am about to cook your meal.
            AAVE: I am coming to cook your meal.

            SAE: If he called me, I would be happy.
            AAVE: If he would call me, I'd be happy.

            SAE: I wish that people in the world would get educated.
            AAVE: I wish that people in the world will get educated.

            SAE: We almost drowned that day.
            AAVE: We liketa drowned that day.

            SAE: I have gotten a book.
            AAVE: I done got a book.

            SAE: He didn't come.
            AAVE: He never came.

            SAE: Can I go home?
            AAVE: I want to go home, can or not?

            SAE: Isn't he arriving tomorrow? No, he isn't.
            AAVE: Isn't he arriving tomorrow? Yes.

            SAE: I see the house.
            AAVE: I sees the house.

            SAE: I sing and dance.
            AAVE: I sing and dances.

            SAE: Here I am.
            AAVE: Here I be.

            SAE: My brother, who lives down the street, is coming over.
            AAVE: My brother, which lives down the street, is coming over.

            SAE: This is the man who painted my house.
            AAVE: This is the man at painted my house.

            SAE: My father was one of the founders of the Underground Railroad, which helped the slaves run away to the North.
            AAVE: My father was one of the founders o' de Underground Railroad where help de slaves to run way to de North.

            SAE: This is the man who painted my house.
            AAVE: This is the man what painted my house.

            SAE: But these little fellows who had stayed before were the only ones who knew what really happened that night.
            AAVE: But these, these little fellahs that which had stayed befoʼ are the only ones who knew what happened.
            """
    elif (args.target_dial == 'Chicano English'):
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

    # Change :1 into whatever number of data points you want to run, get rid of it for all
    # Leave 'super_glue' for now for testing
    ds = load_dataset('super_glue', args.dataset, split="train[:30]")
    df = ds.to_pandas()

    choice_columns=[]
    # inserting premise_translated column and as many choice{i}_translated columns as needed
    if not args.no_premise:
        df['premise_translated'] = None
    for i in range(args.n_choices):
        df[f"choice{i+1}_translated"] = None
        choice_columns.append(f"choice{i+1}")
    df['question_translated'] = None

    # removing idx column
    del df['idx']

    def translate_text(text, source_dial="English", target_dial=args.target_dial):
        full_prompt = few_shot_prompt + f"\nSAE: {text}\n{args.target_dial}:"
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "user", "content": f"{full_prompt}"},
            ],
            temperature = args.temp
        )
        return response.choices[0].message.content.strip()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Translating rows"):
        # Translate question if not done
        if pd.isna(row["question_translated"]):
            question_translation = translate_text(row["question"])
            df.at[i, "question_translated"] = question_translation

        # Translate premise/passage if exists
        if not args.no_premise:
            if pd.isna(row["premise_translated"]):
                question_translation = translate_text(row["premise"])
                df.at[i, "premise_translated"] = question_translation

        # Translate choices if not done
        for choice_col in choice_columns:
            if pd.isna(row[f"{choice_col}_translated"]):
                passage_translation = translate_text(row[choice_col])
                df.at[i, f"{choice_col}_translated"] = passage_translation
            

        # Save progress every 10 rows
        if (i + 1) % 10 == 0:
            df.to_excel(args.output_dir, index=False)

    df.to_excel(args.output_dir, index=False)

if __name__ == '__main__':
    asyncio.run(main())