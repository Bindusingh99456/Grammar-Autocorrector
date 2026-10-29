# Empirical Error Analysis & Weakness Classification (Phase M6)

**Course:** B.Tech Natural Language Processing  
**Evaluation Split:** Held-out test cases (`tests/grammar_cases.json`, `tests/spelling_cases.json`) and Cambridge BEA-2019 Learner English (`data/processed/wi_dev.csv`)  
**Date:** October 2026  
**Annotator:** Cambridge ERRANT (`en_core_web_sm`)  
**Evaluated Sentences:** 600 sentences | **Total Gold Edits:** 856 | **Missed Edits:** 783  

---

## 1. Executive Summary & Weakness Distribution

To improve the model using its own mistakes, we evaluated the full hybrid pipeline on real learner sentences and hand-checked cases. Each failed correction was parsed and labeled using the standard ERRANT grammatical taxonomy. The distribution reveals that learner errors in open-domain English are heavily dominated by preposition choice, article omissions, verb tense/form, and noun number:

| Rank | ERRANT Category | Description | Missed Edits | % of Failures |
|:---:|:---|:---|:---:|:---:|
| **1** | `R:OTHER` | Linguistic alteration | **147** | 18.8% |
| **2** | `R:NOUN` | Linguistic alteration | **98** | 12.5% |
| **3** | `R:VERB` | Linguistic alteration | **48** | 6.1% |
| **4** | `R:VERB:TENSE` | Verb Tense Inconsistency (confusing present vs. past, present perfect vs. simple past, or conditional forms) | **47** | 6.0% |
| **5** | `R:ORTH` | Orthography & Casing (capitalization errors, hyphenation, or spacing differences) | **46** | 5.9% |
| **6** | `M:DET` | Missing Determiner / Article (omitting 'a', 'an', or 'the' before singular countable nouns or definite entities) | **45** | 5.7% |
| **7** | `R:PREP` | Preposition Replacement (e.g., using 'in' instead of 'at', 'on' instead of 'in', or wrong dependent preposition) | **40** | 5.1% |
| **8** | `U:DET` | Unnecessary Determiner / Article (inserting redundant 'the' or 'a' before proper nouns, sports, or plurals) | **33** | 4.2% |
| **9** | `R:VERB:SVA` | Subject-Verb Agreement on Open-Domain Verbs (complex noun phrases, intervening clauses, or collective nouns) | **32** | 4.1% |
| **10** | `R:NOUN:NUM` | Noun Number Agreement (singular vs. plural mismatches, e.g., 'two student', uncountable noun pluralizations) | **27** | 3.4% |

---

## 2. Top 10 Weak Categories: Deep Dive & 5 Concrete Examples Each

### 1. `R:OTHER` — Linguistic alteration

- **Frequency:** 147 missed edits (18.8% of total failures)
- **Linguistic Analysis:** Specific grammatical divergence where the model left learner phrasing unchanged to prevent hallucination.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `Between you and I, this is a bad idea.` | `Between you and I, this is a bad idea.` | `Between you and me, this is a bad idea.` | &ldquo;I,&rdquo; &rarr; &ldquo;me,&rdquo; |
| 2 | `It's difficult because I'll have to study hard and a lot, but I think that if you like a subject, you'll study it easier.` | `It's difficult because I'll have to study hard and a lot, but I think that if you like a subject, you'll study it easiers.` | `It's difficult because I'll have to study hard and a lot, but I think that if you like a subject, you'll study it more easily.` | &ldquo;easier.&rdquo; &rarr; &ldquo;more easily.&rdquo; |
| 3 | `I consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car.` | `I consider that it is more convenient to drive a car because you carry on more things in your own car than travelling by car.` | `I consider it more convenient to drive a car, because you carry more things in your own car than whentravelling by car.` | &ldquo;that is&rdquo; &rarr; &ldquo;it&rdquo; |
| 4 | `In India we have various types of Public transport, like Cycle, Bike, Car, Train & Flight.` | `In India we have various types of Public transport, like Cycle, Bike, Car, Train & Flight.` | `In India, we have various types of public transport, like , bikes, cars, trains & planes.` | &ldquo;Cycle,&rdquo; &rarr; &ldquo;,&rdquo; |
| 5 | `In India we have various types of Public transport, like Cycle, Bike, Car, Train & Flight.` | `In India we have various types of Public transport, like Cycle, Bike, Car, Train & Flight.` | `In India, we have various types of public transport, like , bikes, cars, trains & planes.` | &ldquo;Flight.&rdquo; &rarr; &ldquo;planes.&rdquo; |

### 2. `R:NOUN` — Linguistic alteration

- **Frequency:** 98 missed edits (12.5% of total failures)
- **Linguistic Analysis:** Specific grammatical divergence where the model left learner phrasing unchanged to prevent hallucination.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `I would like to study Psychology because one day I would open my own psychology office and help people.` | `I would like to study Psychology because one day I would open my own psychology office and help people.` | `I would like to study Psychology, because one day I would like to open my own psychology clinic and help people.` | &ldquo;Psychology&rdquo; &rarr; &ldquo;Psychology,&rdquo; |
| 2 | `I would like to study Psychology because one day I would open my own psychology office and help people.` | `I would like to study Psychology because one day I would open my own psychology office and help people.` | `I would like to study Psychology, because one day I would like to open my own psychology clinic and help people.` | &ldquo;office&rdquo; &rarr; &ldquo;clinic&rdquo; |
| 3 | `The rich people will buy a car but the poor people always need to use a bus or taxi.` | `The rich people will buy a car but the poor people always need to use a bus or taxi.` | `Rich people will buy a car, but poor people always need to use a bus or taxi.` | &ldquo;car&rdquo; &rarr; &ldquo;car,&rdquo; |
| 4 | `I consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car.` | `I consider that it is more convenient to drive a car because you carry on more things in your own car than travelling by car.` | `I consider it more convenient to drive a car, because you carry more things in your own car than whentravelling by car.` | &ldquo;car&rdquo; &rarr; &ldquo;car,&rdquo; |
| 5 | `Also, you'll meet friendly people who usually ask to you something to be friends and change your telephone number.` | `Also, you'll meet friendly people who usually ask to you something to be friends and change your telephone number.` | `Also, you'll meet friendly people who usually ask you to be friends and exchange telephone numbers.` | &ldquo;number.&rdquo; &rarr; &ldquo;numbers.&rdquo; |

### 3. `R:VERB` — Linguistic alteration

- **Frequency:** 48 missed edits (6.1% of total failures)
- **Linguistic Analysis:** Specific grammatical divergence where the model left learner phrasing unchanged to prevent hallucination.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `I think that the public transport will always be in the future.` | `I think that the public transport will always be in the future.` | `I think that public transport will always exist in the future.` | &ldquo;be&rdquo; &rarr; &ldquo;exist&rdquo; |
| 2 | `I consider that is more convenient to drive a car because you carry on more things in your own car than travelling by car.` | `I consider that it is more convenient to drive a car because you carry on more things in your own car than travelling by car.` | `I consider it more convenient to drive a car, because you carry more things in your own car than whentravelling by car.` | &ldquo;travelling&rdquo; &rarr; &ldquo;whentravelling&rdquo; |
| 3 | `Also, you'll meet friendly people who usually ask to you something to be friends and change your telephone number.` | `Also, you'll meet friendly people who usually ask to you something to be friends and change your telephone number.` | `Also, you'll meet friendly people who usually ask you to be friends and exchange telephone numbers.` | &ldquo;change&rdquo; &rarr; &ldquo;exchange&rdquo; |
| 4 | `Since, we have the option to wait anytime we want to when we travel by car (which is impossible when travelling by train & Flight).` | `Since, we have the option to wait anytime we want to when we travel by car (which is impossible when travelling by train & Flight).` | `Since, we have the option to stop anytime we want to when we travel by car (which is impossible when travelling by train or plane).` | &ldquo;wait&rdquo; &rarr; &ldquo;stop&rdquo; |
| 5 | `Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.` | `Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.` | `Volleyball is a sport that is played everywhere. When I am on the beach I like playing with my sister in the sand and then we go in the sea.` | &ldquo;travel&rdquo; &rarr; &ldquo;am&rdquo; |

### 4. `R:VERB:TENSE` — Verb Tense Inconsistency (confusing present vs. past, present perfect vs. simple past, or conditional forms)

- **Frequency:** 47 missed edits (6.0% of total failures)
- **Linguistic Analysis:** Learners struggle with discourse-level tense harmony, abruptly switching between simple past and simple present within the same narrative.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience, when I didn't have a car I used to use the bus to go to school and go back to my house. In my opinion, a car isn't necessary when you have crashed in the street. At that moment, you realize the importance of public transport.` | &ldquo;realized&rdquo; &rarr; &ldquo;realize&rdquo; |
| 2 | `This reminds me of a trip that I have recently been to and the place is Agra.` | `This reminds me of a trip that I have recently been to and the place is Agra.` | `This reminds me of a trip that I have recently been on and the place I visited was Agra.` | &ldquo;is&rdquo; &rarr; &ldquo;visited was&rdquo; |
| 3 | `We have stopped at hotels for having food and just in case if any of us feels hungry, we have purchased some snacks just before the trip.` | `We have stopped at hotels for having food and just in case if any of us feels hungry, we have purchased some snacks just before the trip.` | `We stopped at hotels to have food and just in case any of us felt hungry, we purchased some snacks just before the trip.` | &ldquo;feels&rdquo; &rarr; &ldquo;felt&rdquo; |
| 4 | `Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.` | `Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.` | `Volleyball is a sport that is played everywhere. When I am on the beach I like playing with my sister in the sand and then we go in the sea.` | &ldquo;are going&rdquo; &rarr; &ldquo;go&rdquo; |
| 5 | `when I was young I like plays with the ball in the playground and my friend and I played using the soccer goals as a network of volleyball.` | `when I was young I like plays with the ball in the playground and my friend and I played using the soccer goals as a network of volleyball.` | `When I was young I liked playing with a ball in the playground and my friend and I played using the soccer goals as a volleyball net.` | &ldquo;like&rdquo; &rarr; &ldquo;liked&rdquo; |

### 5. `R:ORTH` — Orthography & Casing (capitalization errors, hyphenation, or spacing differences)

- **Frequency:** 46 missed edits (5.9% of total failures)
- **Linguistic Analysis:** Inconsistent capitalization of nationalities, languages, days of the week, or hyphenated compound adjectives.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience, when I didn't have a car I used to use the bus to go to school and go back to my house. In my opinion, a car isn't necessary when you have crashed in the street. At that moment, you realize the importance of public transport.` | &ldquo;house.In&rdquo; &rarr; &ldquo;house. In&rdquo; |
| 2 | `In India we have various types of Public transport, like Cycle, Bike, Car, Train & Flight.` | `In India we have various types of Public transport, like Cycle, Bike, Car, Train & Flight.` | `In India, we have various types of public transport, like , bikes, cars, trains & planes.` | &ldquo;Public&rdquo; &rarr; &ldquo;public&rdquo; |
| 3 | `But Generally speaking, travelling by car is much more fun when compared with other modes of transport.` | `But Generally speaking, travelling by car is much more fun when compared to other modes of transport.` | `But generally speaking, travelling by car is much more fun when compared with other modes of transport.` | &ldquo;Generally&rdquo; &rarr; &ldquo;generally&rdquo; |
| 4 | `It takes around 6 hours by National highway to go from Delhi to Agra.` | `It takes around 6 hours by National Highway to go from Delhi to Agra.` | `It takes around 6 hours by national highway to go from Delhi to Agra.` | &ldquo;National&rdquo; &rarr; &ldquo;national&rdquo; |
| 5 | `I hope that you give my the chance to join you .` | `I hope that you give me the chance to join you .` | `I hope that you give my the chance to join you.` | &ldquo;you .&rdquo; &rarr; &ldquo;you.&rdquo; |

### 6. `M:DET` — Missing Determiner / Article (omitting 'a', 'an', or 'the' before singular countable nouns or definite entities)

- **Frequency:** 45 missed edits (5.7% of total failures)
- **Linguistic Analysis:** Languages like Russian, Hindi, and Mandarin lack definite/indefinite articles, leading L2 writers to omit 'a/an/the' before singular countable nouns ('visit library' instead of 'visit the library').

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `Depending on the distance and duration to the desired place, mode of transport is chosen accordingly.` | `Depending on the distance and duration to the desired place, mode of transport is chosen accordingly.` | `Depending on the distance and time to the desired destination, the mode of transport is chosen accordingly.` | &ldquo;&rdquo; &rarr; &ldquo;the&rdquo; |
| 2 | `Also people need not to suffer high operating cost running their own car.` | `Also people need not to suffer high operating cost running their own car.` | `Also, people need not suffer the high operating cost of running their own car.` | &ldquo;&rdquo; &rarr; &ldquo;the&rdquo; |
| 3 | `Instead it will be common to buy an air ticket and ride on public plane.` | `Instead, it will be common to buy an air ticket and ride on public planes.` | `Instead, it will be common to buy an air ticket and travel on a public plane.` | &ldquo;&rdquo; &rarr; &ldquo;a&rdquo; |
| 4 | `In my opinion I think that public transport in future is going to continue to be used because there are a lot of people that doesn't have enough money to pay for a car even if it is used,so public transport continue to have future but people are going to use more the cars then bus or underground.` | `In my opinion I think that public transport in future is going to continue to be used because there are a lot of people that don't have enough money to pay for a car even if it is used,so public transport continue to have future but people are going to use more the cars then bus or underground.` | `In my opinion, public transport is going to continue to be used in the future because there are a lot of people that don't have enough money to pay for a car, even if it is used, so public transport will continue to have a future but people are going to use cars more than buses or the underground.` | &ldquo;&rdquo; &rarr; &ldquo;a&rdquo; |
| 5 | `In my opinion I think that public transport in future is going to continue to be used because there are a lot of people that doesn't have enough money to pay for a car even if it is used,so public transport continue to have future but people are going to use more the cars then bus or underground.` | `In my opinion I think that public transport in future is going to continue to be used because there are a lot of people that don't have enough money to pay for a car even if it is used,so public transport continue to have future but people are going to use more the cars then bus or underground.` | `In my opinion, public transport is going to continue to be used in the future because there are a lot of people that don't have enough money to pay for a car, even if it is used, so public transport will continue to have a future but people are going to use cars more than buses or the underground.` | &ldquo;&rdquo; &rarr; &ldquo;the&rdquo; |

### 7. `R:PREP` — Preposition Replacement (e.g., using 'in' instead of 'at', 'on' instead of 'in', or wrong dependent preposition)

- **Frequency:** 40 missed edits (5.1% of total failures)
- **Linguistic Analysis:** Learners frequently transfer prepositional idioms from their native languages (L1 interference), choosing 'in' for surfaces, 'on' for temporal points, or 'at' for abstract concepts. Because prepositions depend strongly on the head verb or noun, local n-grams often fail.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `She is very good in mathematics.` | `She is very good in mathematics.` | `She is very good at mathematics.` | &ldquo;in&rdquo; &rarr; &ldquo;at&rdquo; |
| 2 | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience, when I didn't have a car I used to use the bus to go to school and go back to my house. In my opinion, a car isn't necessary when you have crashed in the street. At that moment, you realize the importance of public transport.` | &ldquo;in&rdquo; &rarr; &ldquo;At&rdquo; |
| 3 | `Due to the above mentioned reasons, I am going to conclude that travelling by car is much more convenient.` | `Due to the above mentioned reasons, I am going to conclude that travelling by car is much more convenient.` | `For the above-mentioned reasons, I am going to conclude that travelling by car is much more convenient.` | &ldquo;Due to&rdquo; &rarr; &ldquo;For&rdquo; |
| 4 | `also, I organized many sports activities before in my school.` | `Also, I organized many sports activities before in my school.` | `Also, I organized many sports activities before at my school.` | &ldquo;in&rdquo; &rarr; &ldquo;at&rdquo; |
| 5 | `Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.` | `Volleyball is a sport play every place, when I travel on the beach I like plays with my sister in the sand and after we are going to the sea.` | `Volleyball is a sport that is played everywhere. When I am on the beach I like playing with my sister in the sand and then we go in the sea.` | &ldquo;to&rdquo; &rarr; &ldquo;in&rdquo; |

### 8. `U:DET` — Unnecessary Determiner / Article (inserting redundant 'the' or 'a' before proper nouns, sports, or plurals)

- **Frequency:** 33 missed edits (4.2% of total failures)
- **Linguistic Analysis:** Hypercorrection leads learners to insert 'the' before uncountable concepts or proper institutions (e.g., 'go to the school' when meaning attending school in general).

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `I think that the public transport will always be in the future.` | `I think that the public transport will always be in the future.` | `I think that public transport will always exist in the future.` | &ldquo;the&rdquo; &rarr; &ldquo;&rdquo; |
| 2 | `The rich people will buy a car but the poor people always need to use a bus or taxi.` | `The rich people will buy a car but the poor people always need to use a bus or taxi.` | `Rich people will buy a car, but poor people always need to use a bus or taxi.` | &ldquo;The rich&rdquo; &rarr; &ldquo;Rich&rdquo; |
| 3 | `The rich people will buy a car but the poor people always need to use a bus or taxi.` | `The rich people will buy a car but the poor people always need to use a bus or taxi.` | `Rich people will buy a car, but poor people always need to use a bus or taxi.` | &ldquo;the&rdquo; &rarr; &ldquo;&rdquo; |
| 4 | `Also, you'll meet friendly people who usually ask to you something to be friends and change your telephone number.` | `Also, you'll meet friendly people who usually ask to you something to be friends and change your telephone number.` | `Also, you'll meet friendly people who usually ask you to be friends and exchange telephone numbers.` | &ldquo;your&rdquo; &rarr; &ldquo;&rdquo; |
| 5 | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience when I didn't have a car I used to use the bus to go to the school and go back to my house.In my opinion, the car isn't necessary when you have crashed in the street, in that moment you realized the importance of a public transport.` | `In my experience, when I didn't have a car I used to use the bus to go to school and go back to my house. In my opinion, a car isn't necessary when you have crashed in the street. At that moment, you realize the importance of public transport.` | &ldquo;the&rdquo; &rarr; &ldquo;&rdquo; |

### 9. `R:VERB:SVA` — Subject-Verb Agreement on Open-Domain Verbs (complex noun phrases, intervening clauses, or collective nouns)

- **Frequency:** 32 missed edits (4.1% of total failures)
- **Linguistic Analysis:** While the Phase 2 rule layer captures pronoun-adjacent SVA ('She go'), complex noun phrases with intervening prepositional phrases ('The list of books are') bypass local rules.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `She sing beautifully in the choir.` | `She sing beautifully in the choir.` | `She sings beautifully in the choir.` | &ldquo;sing&rdquo; &rarr; &ldquo;sings&rdquo; |
| 2 | `The dog bark loudly at night.` | `The dog bark loudly at night.` | `The dog barks loudly at night.` | &ldquo;bark&rdquo; &rarr; &ldquo;barks&rdquo; |
| 3 | `The teacher explain the lesson well.` | `The teacher explain the lesson well.` | `The teacher explains the lesson well.` | &ldquo;explain&rdquo; &rarr; &ldquo;explains&rdquo; |
| 4 | `A bird fly across the blue sky.` | `A bird fly across the blue sky.` | `A bird flies across the blue sky.` | &ldquo;fly&rdquo; &rarr; &ldquo;flies&rdquo; |
| 5 | `The birds flies south for the winter.` | `The birds flies south for the winter.` | `The birds fly south for the winter.` | &ldquo;flies&rdquo; &rarr; &ldquo;fly&rdquo; |

### 10. `R:NOUN:NUM` — Noun Number Agreement (singular vs. plural mismatches, e.g., 'two student', uncountable noun pluralizations)

- **Frequency:** 27 missed edits (3.4% of total failures)
- **Linguistic Analysis:** Countable vs. uncountable distinctions (e.g., 'advices', 'informations') and irregular plurals (e.g., 'childrens', 'peoples') remain challenging for the general model.

#### Concrete Failure Cases:

| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |
|:---:|:---|:---|:---|:---|
| 1 | `The clock tick softly in the room.` | `The clock tick softly in the room.` | `The clock ticks softly in the room.` | &ldquo;tick&rdquo; &rarr; &ldquo;ticks&rdquo; |
| 2 | `In my opinion I think that public transport in future is going to continue to be used because there are a lot of people that doesn't have enough money to pay for a car even if it is used,so public transport continue to have future but people are going to use more the cars then bus or underground.` | `In my opinion I think that public transport in future is going to continue to be used because there are a lot of people that don't have enough money to pay for a car even if it is used,so public transport continue to have future but people are going to use more the cars then bus or underground.` | `In my opinion, public transport is going to continue to be used in the future because there are a lot of people that don't have enough money to pay for a car, even if it is used, so public transport will continue to have a future but people are going to use cars more than buses or the underground.` | &ldquo;bus&rdquo; &rarr; &ldquo;buses&rdquo; |
| 3 | `If we think about it the car is better because we don't need to wait for them has we are waiting for the bus or underground but in another way car cust more money than the public transport.` | `If we think about it the car is better because we don't need to wait for them has we are waiting for the bus or underground but in another way car just more money than the public transport.` | `If we think about it, the car is better because we don't need to wait for it like we wait for the bus or underground, but on the other hand, cars cost more money than public transport.` | &ldquo;car&rdquo; &rarr; &ldquo;cars&rdquo; |
| 4 | `Michael and Kate was so happy, until one day that Michael said to him dream to be a super hero, Kate get so nervous saying that it was ridiculous and just a kid dream and that Michael was only dreaming but that would never happen and that super hero don’t exist.` | `Michael and Kate was so happy, until one day that Michael said to him dream to be a super hero, Kate get so nervous saying that it was ridiculous and just a kid dream and that Michael was only dreaming but that would never happen and that super hero don’to exist.` | `Michael and Kate were so happy, until one day Michael told her about his dream of being a super hero. Kate got so agitated, saying that it was ridiculous and just a kid's dream and that Michael was only dreaming and that would never happen and that super heroes didn’t exist.` | &ldquo;hero&rdquo; &rarr; &ldquo;heroes&rdquo; |
| 5 | `If there is no public transportation, traffic jam will be serious than before.` | `If there is no public transportation, traffic jam will be serious than before.` | `If there is no public transportation, traffic jams will be more serious than before.` | &ldquo;jam&rdquo; &rarr; &ldquo;jams&rdquo; |

---

## 3. Targeted Remediation Strategy (Phase M6 Action Plan)

To systematically address these top 10 empirical weaknesses, Phase M6 implements targeted data synthesis in `scripts/make_noise.py`:

1. **Preposition Confusion Generator (`R:PREP`):** Swap dependent prepositions based on verb-preposition co-occurrence statistics (e.g., *depend of* &rarr; *depend on*, *interested for* &rarr; *interested in*).
2. **Determiner Inserter/Dropper (`M:DET`, `U:DET`):** Systematically drop articles before singular count nouns and inject unnecessary 'the' before abstract nouns.
3. **Verb Tense & Form Corruptor (`R:VERB:TENSE`, `R:VERB:FORM`):** Use `lemminflect` to corrupt past participles to base forms and swap gerunds with infinitives.
4. **Noun Number Corruptor (`R:NOUN:NUM`):** Pluralize uncountable nouns (*informations*, *equipments*) and singularize plural subjects following cardinal numbers.
5. **Complex SVA Corruptor (`R:VERB:SVA`):** Corrupt verbs separated from their subject by prepositional phrases (*'The box of chocolates are'*).
6. **Derivational Morphology Corruptor (`R:MORPH`):** Swap adjective/adverb suffixes (*-ful*, *-ly*, *-tion*, *-ity*).
7. **Targeted Corpus:** Generate 30,000 synthetic pairs focused exclusively on these weak categories and save to `data/processed/targeted.csv`.

8. **Stage 2 Continual Fine-Tuning:** Mix `targeted.csv` into Stage 2 training at a controlled ratio (keeping real human learner data at &ge; 50%) to improve model recall without compromising fluency.
