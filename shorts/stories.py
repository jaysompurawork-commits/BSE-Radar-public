"""
Story library for YouTube Shorts.

Each story is a complete narrative arc: hook → rising tension → revelation → payoff.
NOT 10-second fact lists. These are 60-90 second mini-documentaries with a beginning,
middle, and end that keep the viewer watching.

Every story has:
  - title: clickable, curiosity-gap title
  - hook: first 5 seconds — the question or tension that makes them stay
  - scenes: list of dicts with narration + visual description + color palette
  - cta: call to action / memorable final line

Stories rotate daily so the channel never repeats.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Story definitions
# ---------------------------------------------------------------------------

STORIES: List[Dict[str, Any]] = [
    {
        "id": "napoleon_height",
        "title": "Napoleon Was NOT Short — The Lie That Fooled the World #Shorts",
        "category": "history",
        "hook": "Everyone knows Napoleon was tiny. Generals towered over him. Artists mocked him. But what if that's the biggest lie in history?",
        "scenes": [
            {
                "narration": "Napoleon Bonaparte. History's most famous short man. At least, that's what you've been told. But there's a problem with that story.",
                "visual": "dramatic portrait reveal",
                "color": "#1a1a2e",
                "accent": "#e94560",
                "duration": 8,
            },
            {
                "narration": "Napoleon stood 5 foot 6 in modern measurements. That was actually above average height for a French man in 1800. So where did the myth come from?",
                "visual": "height comparison chart animation",
                "color": "#16213e",
                "accent": "#0f3460",
                "duration": 10,
            },
            {
                "narration": "Two sources. First: British propaganda. While Napoleon was conquering Europe, British cartoonists needed to make him look weak. They drew him tiny, called him 'Little Boney.' The insult stuck across a continent.",
                "visual": "newspaper propaganda illustration",
                "color": "#1a1a2e",
                "accent": "#f5a623",
                "duration": 12,
            },
            {
                "narration": "Second: a measurement mix-up. Napoleon was 5 foot 2 in French inches — but French inches were longer than English inches. British readers saw '5 foot 2' and laughed, not knowing the units were different.",
                "visual": "ruler comparison animation",
                "color": "#0d0d0d",
                "accent": "#4ecdc4",
                "duration": 11,
            },
            {
                "narration": "His nickname among his own soldiers was 'le petit caporal' — the little corporal. But that was a term of affection, like calling a boss 'chief.' Not a comment on his height.",
                "visual": "soldiers animated scene",
                "color": "#1a1a2e",
                "accent": "#ff6b6b",
                "duration": 10,
            },
            {
                "narration": "Napoleon was surrounded by his Imperial Guard — men who had to be at least 6 feet tall to qualify. Standing next to giants every day made him look small in paintings.",
                "visual": "imperial guard size comparison",
                "color": "#16213e",
                "accent": "#a29bfe",
                "duration": 10,
            },
            {
                "narration": "Two hundred years of a lie. Started by enemy propaganda, spread by a unit conversion error, locked in by artistic exaggeration. Napoleon was average height. But average doesn't make history.",
                "visual": "dramatic close up animated",
                "color": "#1a1a2e",
                "accent": "#fd79a8",
                "duration": 9,
            },
        ],
        "hashtags": ["#Shorts", "#History", "#NapoleonBonaparte", "#HistoryMyths", "#FunFacts"],
        "description": "Napoleon wasn't short — here's where the myth actually came from. #Shorts #History",
    },
    {
        "id": "great_wall_space",
        "title": "You Can't See the Great Wall From Space — Astronauts Confirmed It #Shorts",
        "category": "science",
        "hook": "Every school textbook says it. Parents repeat it. Teachers confirm it. The Great Wall of China is visible from space. Except — no astronaut has ever actually seen it.",
        "scenes": [
            {
                "narration": "The Great Wall of China. Two thousand miles long. Supposedly visible from space with the naked eye. It's in every geography textbook. But it's completely false.",
                "visual": "satellite view animation",
                "color": "#0a0a0a",
                "accent": "#00b894",
                "duration": 9,
            },
            {
                "narration": "Chinese astronaut Yang Liwei was asked specifically about it after his 2003 spaceflight. He admitted he looked. He couldn't see it. China was embarrassed. They quietly rewrote their school textbooks.",
                "visual": "astronaut in space animated",
                "color": "#0d1117",
                "accent": "#58a6ff",
                "duration": 12,
            },
            {
                "narration": "Here's the physics. The wall is at most 30 feet wide. From the International Space Station, 250 miles up, that's like trying to see a human hair from 2 miles away. Your eye physically cannot do it.",
                "visual": "physics diagram animation",
                "color": "#161b22",
                "accent": "#f0883e",
                "duration": 11,
            },
            {
                "narration": "NASA confirmed it in 2004. The only way astronauts have photographed the wall is with zoom lenses — not the naked eye. Even then, they have to know exactly where to look.",
                "visual": "NASA photo zoom animation",
                "color": "#0d1117",
                "accent": "#3fb950",
                "duration": 10,
            },
            {
                "narration": "So where did the myth start? A 1932 Ripley's Believe It or Not column claimed it, before any human had ever been to space. People just... believed it. And repeated it for 90 years.",
                "visual": "old newspaper reveal animation",
                "color": "#1a1a2e",
                "accent": "#f9826c",
                "duration": 11,
            },
            {
                "narration": "The Great Wall IS an incredible feat of engineering — 2,000 years of construction, millions of workers. It doesn't need a myth to be impressive. The real story is better.",
                "visual": "great wall beauty shot animation",
                "color": "#0a0a0a",
                "accent": "#79c0ff",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Science", "#GreatWall", "#MythBusted", "#SpaceFacts"],
        "description": "No astronaut has ever seen the Great Wall from space. The real story is more interesting. #Shorts",
    },
    {
        "id": "einstein_failed_school",
        "title": "Einstein Didn't Fail Math — The Internet Lied to You #Shorts",
        "category": "motivation",
        "hook": "You've seen the inspirational post. Einstein failed math. If he can fail and still become a genius, you can do anything. There's just one problem — it never happened.",
        "scenes": [
            {
                "narration": "Albert Einstein. Reportedly failed math as a child. This story has inspired millions of struggling students. Posters hang in classrooms worldwide. And it's completely made up.",
                "visual": "classroom poster reveal",
                "color": "#1a1a2e",
                "accent": "#fdcb6e",
                "duration": 9,
            },
            {
                "narration": "Einstein learned calculus at age 12. He taught himself advanced mathematics before he turned 15. When someone showed him this myth, he literally laughed. He said: 'I never failed in mathematics. Before I was fifteen I had mastered differential and integral calculus.'",
                "visual": "young einstein study animation",
                "color": "#16213e",
                "accent": "#6c5ce7",
                "duration": 14,
            },
            {
                "narration": "Where did it come from? In Switzerland, the grading scale was flipped — 6 was the best score, 1 was the worst. An American biographer saw Einstein's Swiss report cards, looked at the low numbers, and assumed he was failing. He wasn't. He was acing everything.",
                "visual": "report card animation Swiss vs American",
                "color": "#0d0d0d",
                "accent": "#00cec9",
                "duration": 13,
            },
            {
                "narration": "Einstein DID fail an entrance exam — once — to the Swiss Federal Polytechnic, at age 15, a year younger than most applicants. He still scored highest in math and physics. He just hadn't studied French and botany.",
                "visual": "exam paper animation",
                "color": "#1a1a2e",
                "accent": "#e17055",
                "duration": 11,
            },
            {
                "narration": "The real Einstein story is actually more inspiring. He wasn't a secret failure who defied the odds. He was a brilliant, obsessive kid who thought so differently from his teachers that they called him slow — not because he couldn't learn, but because he refused to memorize without understanding.",
                "visual": "thinking animation with equations",
                "color": "#16213e",
                "accent": "#74b9ff",
                "duration": 14,
            },
            {
                "narration": "The truth doesn't need a comforting lie. Einstein's real lesson? Question everything — including the inspirational stories people tell about geniuses.",
                "visual": "final portrait dramatic",
                "color": "#1a1a2e",
                "accent": "#a29bfe",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Einstein", "#Motivation", "#HistoryMyths", "#Education"],
        "description": "Einstein didn't fail math — here's what actually happened. #Shorts #History",
    },
    {
        "id": "cleopatra_pyramid",
        "title": "Cleopatra Was Closer to the iPhone Than the Pyramids #Shorts",
        "category": "history",
        "hook": "When you picture Cleopatra, you probably see the pyramids behind her. She was Egyptian after all. But Cleopatra lived closer in time to you than she did to the construction of the pyramids.",
        "scenes": [
            {
                "narration": "Cleopatra VII died in 30 BC. The Great Pyramid of Giza was finished around 2560 BC. That's a gap of 2,530 years between them.",
                "visual": "timeline animation dramatic",
                "color": "#1a1a2e",
                "accent": "#f9ca24",
                "duration": 9,
            },
            {
                "narration": "By contrast, Cleopatra lived around 2,050 years ago. The iPhone launched in 2007 — less than 20 years ago. You are closer to Cleopatra in time than Cleopatra was to the pyramids.",
                "visual": "timeline comparison animation",
                "color": "#16213e",
                "accent": "#6ab04c",
                "duration": 12,
            },
            {
                "narration": "When Cleopatra walked through Egypt, the pyramids were already ancient monuments. They were over 2,000 years old. About as ancient to her as she is to us today.",
                "visual": "ancient egypt aerial animation",
                "color": "#0d0d0d",
                "accent": "#eb4d4b",
                "duration": 10,
            },
            {
                "narration": "Cleopatra wasn't even Egyptian by ethnicity. She was of Greek-Macedonian descent, a member of the Ptolemaic dynasty founded by one of Alexander the Great's generals. She was the FIRST ruler of her dynasty to actually learn the Egyptian language.",
                "visual": "greek macedonian map animation",
                "color": "#1a1a2e",
                "accent": "#be2edd",
                "duration": 13,
            },
            {
                "narration": "She spoke nine languages. She was a mathematician, a philosopher, and a naval strategist. The image of her as just a beautiful seductress was invented by her Roman enemies, who wrote the history books.",
                "visual": "library scroll animation",
                "color": "#16213e",
                "accent": "#f0932b",
                "duration": 11,
            },
            {
                "narration": "Time has a way of flattening everything. 3,000 years of Egyptian history gets compressed into one image — pyramids, pharaohs, queens — all existing at the same moment. They didn't. And the real story is far stranger.",
                "visual": "pyramid sunset dramatic animation",
                "color": "#1a1a2e",
                "accent": "#e55039",
                "duration": 8,
            },
        ],
        "hashtags": ["#Shorts", "#Cleopatra", "#History", "#MindBlown", "#HistoryFacts"],
        "description": "Cleopatra lived closer to the iPhone than to the pyramids. The math is wild. #Shorts",
    },
    {
        "id": "tunguska_event",
        "title": "In 1908, Something Flattened 800 Square Miles of Forest — We Still Don't Know Why #Shorts",
        "category": "science",
        "hook": "June 30, 1908. Siberia. An explosion equivalent to 185 nuclear bombs flattened 800 square miles of forest. Eighty million trees, gone in seconds. No crater. No debris. No explanation for 20 years.",
        "scenes": [
            {
                "narration": "June 30, 1908, 7:17 AM. A farmer in Siberia watched a column of blue light, as bright as the sun, move across the sky. Then the heat hit. Then silence.",
                "visual": "siberian forest dramatic animation",
                "color": "#0a0a0a",
                "accent": "#e74c3c",
                "duration": 10,
            },
            {
                "narration": "Five hundred miles away in a town called Vanavara, windows shattered. People were knocked off their feet. Seismographs as far away as England detected the shock.",
                "visual": "shockwave animation spreading",
                "color": "#1a1a2e",
                "accent": "#f39c12",
                "duration": 10,
            },
            {
                "narration": "When scientists finally reached the site 19 years later — in 1927 — they found 800 square miles of flattened trees, all pointing away from one central point. Like a massive hand had pressed them flat.",
                "visual": "aerial forest toppled animation",
                "color": "#0d0d0d",
                "accent": "#27ae60",
                "duration": 11,
            },
            {
                "narration": "But there was no crater. No meteorite fragments. No evidence of what caused it. If something hit the ground, it left no trace.",
                "visual": "scientists searching animation",
                "color": "#1a1a2e",
                "accent": "#3498db",
                "duration": 9,
            },
            {
                "narration": "The current best explanation: an asteroid or comet roughly 50 to 80 metres across exploded in the atmosphere at an altitude of 5 to 10 kilometres. The airburst released energy equivalent to 15 megatons of TNT — more than 1,000 times the Hiroshima bomb.",
                "visual": "asteroid explosion atmosphere animation",
                "color": "#16213e",
                "accent": "#e74c3c",
                "duration": 14,
            },
            {
                "narration": "Because it exploded before hitting the ground, it left no crater and vaporized itself completely. If it had arrived 4 hours later, the Earth's rotation would have placed St. Petersburg directly underneath it.",
                "visual": "earth rotation map animation",
                "color": "#0a0a0a",
                "accent": "#9b59b6",
                "duration": 11,
            },
            {
                "narration": "The Tunguska event remains the largest impact event in recorded human history. A reminder that space is not empty, and our planet has been hit before. The question isn't if it happens again. It's when.",
                "visual": "space stars dramatic close",
                "color": "#1a1a2e",
                "accent": "#1abc9c",
                "duration": 9,
            },
        ],
        "hashtags": ["#Shorts", "#Science", "#Tunguska", "#SpaceScience", "#Mysteries"],
        "description": "In 1908, something exploded over Siberia with the force of 185 nuclear bombs. #Shorts",
    },
    {
        "id": "honey_never_spoils",
        "title": "3,000-Year-Old Honey Found in Egyptian Tombs — Still Perfectly Edible #Shorts",
        "category": "science",
        "hook": "Archaeologists opened a sealed Egyptian tomb and found jars of honey. The honey was 3,000 years old. They tasted it. It was still good. Here's the science of why honey never dies.",
        "scenes": [
            {
                "narration": "In 1922, Howard Carter's team unsealed Tutankhamun's tomb. Among the treasures, they found sealed jars of honey. Three thousand years old. Still edible.",
                "visual": "tomb discovery animated cinematic",
                "color": "#1a1a2e",
                "accent": "#f1c40f",
                "duration": 10,
            },
            {
                "narration": "This isn't magic. It's chemistry. Honey has almost no water content — typically less than 20 percent. Bacteria and mould need water to survive and reproduce. Without it, they can't grow in honey.",
                "visual": "molecule animation water and honey",
                "color": "#0d0d0d",
                "accent": "#e67e22",
                "duration": 11,
            },
            {
                "narration": "Bees also add an enzyme called glucose oxidase. When mixed with the tiny amount of water in honey, this enzyme produces hydrogen peroxide — a natural antibacterial agent.",
                "visual": "bee enzyme animation",
                "color": "#16213e",
                "accent": "#2ecc71",
                "duration": 10,
            },
            {
                "narration": "Honey is also highly acidic, with a pH between 3 and 4.5. This acidic environment kills most pathogens. Combined with the low water content and the hydrogen peroxide, nothing can live in honey long enough to spoil it.",
                "visual": "pH scale animated",
                "color": "#1a1a2e",
                "accent": "#e74c3c",
                "duration": 11,
            },
            {
                "narration": "The only reason honey appears to 'go bad' is crystallization — the glucose separated from the liquid. That's not spoilage. A jar of crystallized honey can be liquefied by warming it gently, and it's perfectly safe to eat.",
                "visual": "crystallization animation",
                "color": "#0d0d0d",
                "accent": "#3498db",
                "duration": 11,
            },
            {
                "narration": "Honey in sealed containers has no expiry date. In theory, honey sealed correctly could last forever. The bees evolved this property over 100 million years. Ancient Egyptians knew it. They buried their pharaohs with it.",
                "visual": "timeline 100 million years animation",
                "color": "#1a1a2e",
                "accent": "#9b59b6",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Science", "#Honey", "#Egypt", "#FoodScience"],
        "description": "3,000-year-old honey from Egyptian tombs is still edible. Here's the chemistry. #Shorts",
    },
    {
        "id": "voynich_manuscript",
        "title": "The World's Most Mysterious Book — No One Can Read a Single Word #Shorts",
        "category": "facts_mysteries",
        "hook": "A 600-year-old book sits in a Yale library. It's filled with drawings of naked women, strange plants, and astrological charts. And every single word is written in a language no one on Earth can read.",
        "scenes": [
            {
                "narration": "The Voynich Manuscript. A handwritten book from the 15th century. 240 pages. Over 37,000 words. Illustrations on nearly every page. And despite a century of attempts, no one has decoded a single sentence.",
                "visual": "book manuscript pages animation",
                "color": "#1a1a2e",
                "accent": "#8e44ad",
                "duration": 11,
            },
            {
                "narration": "The writing follows rules. It has grammar patterns. The words are distributed the way real languages distribute words — common short words, rare long ones. Linguists believe it IS a real writing system, not random marks.",
                "visual": "text analysis animation patterns",
                "color": "#0d0d0d",
                "accent": "#27ae60",
                "duration": 11,
            },
            {
                "narration": "The illustrations make it stranger. There are plants that don't match any known species. Naked women bathing in elaborate pipe systems. Astrological charts for stars that don't exist in our sky.",
                "visual": "manuscript illustrations animated reveal",
                "color": "#16213e",
                "accent": "#e74c3c",
                "duration": 10,
            },
            {
                "narration": "During World War Two, America's top codebreakers — the people who cracked Nazi ciphers — tried to decode it. They failed. Cold War CIA cryptographers tried. They failed. Modern AI systems have run statistical analyses and still cannot translate it.",
                "visual": "codebreakers animation wartime",
                "color": "#1a1a2e",
                "accent": "#f39c12",
                "duration": 12,
            },
            {
                "narration": "Carbon dating places the vellum — the writing material — between 1404 and 1438. The book was sold multiple times over centuries, its price suggesting people believed it held great secrets. It reached its current home at Yale in 1969.",
                "visual": "auction historical animation",
                "color": "#0d0d0d",
                "accent": "#3498db",
                "duration": 11,
            },
            {
                "narration": "Three main theories exist. One: it's an elaborate hoax from the 15th century. Two: it's a personal cipher by a single author. Three: it's a lost language. No theory has been proven. The Voynich Manuscript remains the most studied undeciphered text in the world.",
                "visual": "three theories animated split",
                "color": "#1a1a2e",
                "accent": "#1abc9c",
                "duration": 9,
            },
            {
                "narration": "In Yale's Beinecke Library, the book sits. Anyone can request to see it. Some pages are online. You can look at every word. And you will understand none of it. Just like everyone else.",
                "visual": "library dramatic close animation",
                "color": "#0a0a0a",
                "accent": "#e67e22",
                "duration": 6,
            },
        ],
        "hashtags": ["#Shorts", "#Mystery", "#VoynichManuscript", "#History", "#Unsolved"],
        "description": "The Voynich Manuscript has 37,000 words and no one can read a single one. #Shorts",
    },
    {
        "id": "tardigrades_indestructible",
        "title": "This Animal Can Survive in Outer Space Without a Spacesuit #Shorts",
        "category": "science",
        "hook": "It's smaller than a grain of sand. It lives in your backyard. And it is, without question, the toughest creature that has ever existed on Earth — able to survive conditions that would kill every other living thing.",
        "scenes": [
            {
                "narration": "Meet the tardigrade. Half a millimetre long. Eight legs. A face that looks like a tiny suction cup. And the most indestructible animal on the planet.",
                "visual": "microscope zoom animation tardigrade",
                "color": "#0d0d0d",
                "accent": "#1abc9c",
                "duration": 9,
            },
            {
                "narration": "Tardigrades can survive in outer space. Not with protection — exposed. The vacuum, the cosmic radiation, the extreme temperatures. Scientists sent them to orbit and brought them back alive.",
                "visual": "space exposure animation dramatic",
                "color": "#1a1a2e",
                "accent": "#3498db",
                "duration": 10,
            },
            {
                "narration": "Their secret: they can enter a state called cryptobiosis. They expel almost all the water from their body, pull their legs in, and essentially stop being alive. Metabolism drops to 0.01 percent of normal. They just... wait.",
                "visual": "desiccation animation biology",
                "color": "#16213e",
                "accent": "#9b59b6",
                "duration": 12,
            },
            {
                "narration": "In this state, they can survive temperatures from -272 degrees Celsius to +150 degrees. That's from near absolute zero to hotter than boiling water. They can also survive 1,000 times the radiation dose that would kill a human.",
                "visual": "temperature scale dramatic animation",
                "color": "#0d0d0d",
                "accent": "#e74c3c",
                "duration": 11,
            },
            {
                "narration": "Scientists have revived tardigrades from a state of suspended animation after 30 years. Some estimates suggest dried tardigrades in museum collections — collected over a century ago — might still be revivable.",
                "visual": "revival animation museum",
                "color": "#1a1a2e",
                "accent": "#f1c40f",
                "duration": 10,
            },
            {
                "narration": "They have survived all 5 mass extinctions in Earth's history. They've been around for 530 million years. When the asteroid hit and the dinosaurs died, tardigrades just went into hibernation and waited for things to calm down.",
                "visual": "extinction events timeline animation",
                "color": "#0a0a0a",
                "accent": "#e67e22",
                "duration": 11,
            },
            {
                "narration": "When our sun eventually expands into a red giant and scorches the Earth, the tardigrade might outlast us all. The toughest thing on Earth isn't a lion. It's a tiny, eight-legged creature you've probably never noticed.",
                "visual": "sun expansion space animation",
                "color": "#1a1a2e",
                "accent": "#e74c3c",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Science", "#Tardigrade", "#Biology", "#SpaceFacts"],
        "description": "Tardigrades can survive outer space exposure. This is how. #Shorts #Science",
    },
    {
        "id": "dancing_plague_1518",
        "title": "In 1518, People Danced to Their Deaths and No One Could Stop Them #Shorts",
        "category": "facts_mysteries",
        "hook": "July 1518. Strasbourg. A woman named Frau Troffea walked into the street and started dancing. She couldn't stop. Within a month, 400 people had joined her. Some danced until they died.",
        "scenes": [
            {
                "narration": "July 14, 1518. Strasbourg, France. A woman named Frau Troffea stepped into the street and began to dance. She danced for days. She couldn't stop. She didn't eat or sleep. She just danced.",
                "visual": "medieval street scene animation",
                "color": "#1a1a2e",
                "accent": "#e74c3c",
                "duration": 11,
            },
            {
                "narration": "Within a week, thirty-four people had joined her. By August, the number had grown to four hundred. They danced day and night, in a frenzied, uncontrollable way. Witnesses described them as unable to stop even when their feet bled.",
                "visual": "crowd growing animation dramatic",
                "color": "#0d0d0d",
                "accent": "#e67e22",
                "duration": 12,
            },
            {
                "narration": "The authorities of Strasbourg made a catastrophic decision. They concluded the cure was MORE dancing. They hired professional musicians, cleared a stage, and encouraged the dancers. People began dying from exhaustion, strokes, and heart attacks.",
                "visual": "historical decision council animation",
                "color": "#16213e",
                "accent": "#f39c12",
                "duration": 12,
            },
            {
                "narration": "What caused it? Historians and medical experts have debated this for 500 years. Mass psychogenic illness — essentially mass hysteria — is the leading theory. The region was suffering from famines and disease. The stress may have triggered a collective breakdown.",
                "visual": "medical theory animated diagram",
                "color": "#1a1a2e",
                "accent": "#3498db",
                "duration": 12,
            },
            {
                "narration": "Another theory: ergot poisoning. Ergot is a fungus that grows on wet grain, and it contains compounds similar to LSD. It could cause involuntary muscle spasms and hallucinations. 1518 had unusually heavy rains and a bad harvest.",
                "visual": "grain fungus scientific animation",
                "color": "#0d0d0d",
                "accent": "#27ae60",
                "duration": 11,
            },
            {
                "narration": "By September, the plague had stopped as suddenly as it began. The dancers were taken to a shrine at Saverne and slowly recovered. No definitive cause was ever proven. The Dancing Plague of 1518 remains one of the strangest documented mass events in history.",
                "visual": "shrine pilgrimage animation",
                "color": "#1a1a2e",
                "accent": "#9b59b6",
                "duration": 9,
            },
            {
                "narration": "It wasn't the only one. Similar episodes, called choreomania, were recorded in Germany, Italy, and the Netherlands throughout the Middle Ages. History contains events so strange they seem impossible. This one is documented in contemporary civic records.",
                "visual": "historical records parchment animation",
                "color": "#0a0a0a",
                "accent": "#1abc9c",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#History", "#DancingPlague", "#Mystery", "#MiddleAges"],
        "description": "In 1518, hundreds of people danced uncontrollably for weeks. Some died. #Shorts",
    },
    {
        "id": "antikythera_mechanism",
        "title": "Ancient Greeks Built a Computer 2,000 Years Before We Invented One #Shorts",
        "category": "science",
        "hook": "In 1901, divers found a corroded lump of bronze in a 2,000-year-old Greek shipwreck. Scientists spent a century analysing it. What they found rewrote everything we thought we knew about ancient technology.",
        "scenes": [
            {
                "narration": "1901. Divers salvaging a Roman-era shipwreck off the Greek island of Antikythera pulled up a corroded lump of bronze. It was about the size of a shoebox. It didn't look like much.",
                "visual": "underwater shipwreck animation cinematic",
                "color": "#0d0d0d",
                "accent": "#3498db",
                "duration": 10,
            },
            {
                "narration": "Over the next century, scientists carefully studied the object using X-ray tomography. What they found inside was extraordinary: a complex system of at least 37 interlocking bronze gears.",
                "visual": "x-ray reveal gear system animation",
                "color": "#1a1a2e",
                "accent": "#f39c12",
                "duration": 10,
            },
            {
                "narration": "This device, from approximately 100 BC, could predict the positions of the sun and moon, track the cycles of eclipses, calculate the timing of the Olympic Games, and model the movements of five planets visible to the naked eye.",
                "visual": "planetary model animation spinning",
                "color": "#16213e",
                "accent": "#9b59b6",
                "duration": 12,
            },
            {
                "narration": "The gear ratios used show a level of mathematical knowledge that historians didn't believe the ancient Greeks possessed. The mechanism to calculate the Moon's irregular orbit alone required a specific gear ratio of 254 teeth to 19.",
                "visual": "gear ratio animated diagram",
                "color": "#0d0d0d",
                "accent": "#2ecc71",
                "duration": 11,
            },
            {
                "narration": "For context: comparable mechanical complexity didn't reappear in human technology until the 14th century astronomical clocks in Europe — 1,400 years later. This object is, technically, the world's first known analog computer.",
                "visual": "timeline jump animation 1400 years",
                "color": "#1a1a2e",
                "accent": "#e74c3c",
                "duration": 11,
            },
            {
                "narration": "The Antikythera mechanism is now in the National Archaeological Museum in Athens. Researchers are still discovering new details about how it worked. We've been able to reconstruct it using LEGO, with modern 3D printing, and it still functions correctly.",
                "visual": "museum display modern reconstruction",
                "color": "#0a0a0a",
                "accent": "#1abc9c",
                "duration": 11,
            },
            {
                "narration": "Whoever built this was a genius working without any of the tools we consider essential. We don't know their name. We may never know. But what they created was, by any definition, a computer — built 2,000 years before we thought to call it that.",
                "visual": "dramatic portrait unknown craftsman animation",
                "color": "#1a1a2e",
                "accent": "#f1c40f",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Science", "#Antikythera", "#History", "#AncientTechnology"],
        "description": "The Antikythera mechanism: a 2,000-year-old computer found in a shipwreck. #Shorts",
    },
    # More stories for variety
    {
        "id": "oxford_uni_older_aztec",
        "title": "Oxford University Is Older Than the Aztec Empire #Shorts",
        "category": "history",
        "hook": "Oxford University started teaching students in 1096. The Aztec Empire wasn't founded until 1428. That means Oxford was already 332 years old when the Aztecs built Tenochtitlan.",
        "scenes": [
            {
                "narration": "Teaching at Oxford University began in 1096 AD. Students were studying logic, philosophy and mathematics there in a city in England, while the Americas were home to civilizations we don't even know the names of.",
                "visual": "oxford historical animation",
                "color": "#1a1a2e",
                "accent": "#8e44ad",
                "duration": 11,
            },
            {
                "narration": "The Aztec Empire, one of the most advanced pre-Columbian civilizations ever built, was founded in 1428. At that point, Oxford had already been producing scholars for 332 years.",
                "visual": "aztec tenochtitlan animation",
                "color": "#0d0d0d",
                "accent": "#e67e22",
                "duration": 10,
            },
            {
                "narration": "For comparison, America declared independence in 1776. Oxford was already 680 years old. The United States is younger than Oxford by nearly seven centuries.",
                "visual": "comparison timeline animation",
                "color": "#16213e",
                "accent": "#3498db",
                "duration": 10,
            },
            {
                "narration": "Genghis Khan began building the Mongol Empire in 1206 — Oxford was already 110 years old. The Ottoman Empire started in 1299 — Oxford was 203. The Magna Carta was signed in 1215, while students had been studying at Oxford for 119 years.",
                "visual": "world events timeline animation",
                "color": "#1a1a2e",
                "accent": "#e74c3c",
                "duration": 14,
            },
            {
                "narration": "Oxford's oldest surviving building, the University Church of St Mary the Virgin, has been standing since 1280. That's 746 years. The building has watched empires rise and fall while students argue about the same questions they argued about in 1096.",
                "visual": "church building aged animation",
                "color": "#0a0a0a",
                "accent": "#f1c40f",
                "duration": 10,
            },
            {
                "narration": "History has a size problem. We compress centuries into single images — medieval, ancient, modern — and lose the actual scale of time. Oxford sitting there, older than the Aztec Empire, is a small correction to that compressed view.",
                "visual": "scale of time animation",
                "color": "#1a1a2e",
                "accent": "#1abc9c",
                "duration": 5,
            },
        ],
        "hashtags": ["#Shorts", "#History", "#Oxford", "#Aztec", "#TimeFacts"],
        "description": "Oxford University is older than the Aztec Empire. The math is wild. #Shorts",
    },
    {
        "id": "vikings_canada",
        "title": "Vikings Were in Canada 500 Years Before Columbus Landed Anywhere #Shorts",
        "category": "history",
        "hook": "The history books say Columbus discovered America in 1492. But someone else beat him there by 500 years, settled a village, had children, and wrote it all down. We just chose not to believe it until we found the ruins.",
        "scenes": [
            {
                "narration": "Around 1000 AD, a Norse explorer named Leif Erikson sailed west from Greenland and landed on the coast of North America. He called it Vinland. He built a settlement. He lived there for a winter.",
                "visual": "norse ship ocean animation cinematic",
                "color": "#0a0a0a",
                "accent": "#4ea8de",
                "duration": 12,
            },
            {
                "narration": "This was documented in two Norse sagas written in Iceland — the Saga of the Greenlanders and the Saga of Erik the Red. For centuries, historians classified these as myth. Then in 1960, Norwegian archaeologists found the village.",
                "visual": "archaeological dig animation",
                "color": "#1a1a2e",
                "accent": "#f1c40f",
                "duration": 12,
            },
            {
                "narration": "L'Anse aux Meadows, Newfoundland, Canada. An undisturbed Norse settlement with longhouses, a forge, iron-working tools, and artefacts that could only have come from Scandinavia. Dating confirmed: approximately 1000 AD.",
                "visual": "archaeological site animated reconstruction",
                "color": "#16213e",
                "accent": "#27ae60",
                "duration": 12,
            },
            {
                "narration": "Columbus landed in the Bahamas in 1492. That's 492 years after Vikings were already in Canada. The Vikings also didn't 'discover' America because millions of Indigenous people had been living there for at least 15,000 years before either of them arrived.",
                "visual": "timeline 15000 years animation",
                "color": "#0d0d0d",
                "accent": "#e74c3c",
                "duration": 12,
            },
            {
                "narration": "The Norse settlement was abandoned after conflict with the local people they called Skrælings — likely the ancestors of the Beothuk people. The Vikings left. Their sagas remained. And for 500 years, we called it a legend.",
                "visual": "conflict departure animation",
                "color": "#1a1a2e",
                "accent": "#9b59b6",
                "duration": 11,
            },
            {
                "narration": "L'Anse aux Meadows is now a UNESCO World Heritage Site. You can visit it. The Norse longhouses have been reconstructed. Standing there, you're standing in the ruins of European contact with the Americas — 500 years before it officially happened.",
                "visual": "modern tourist site animation",
                "color": "#0a0a0a",
                "accent": "#1abc9c",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Vikings", "#History", "#Canada", "#Columbus"],
        "description": "Vikings were in Canada 500 years before Columbus. We found the village. #Shorts",
    },
    {
        "id": "eels_mystery",
        "title": "We Have No Idea How Eels Reproduce and It's Driving Scientists Crazy #Shorts",
        "category": "nature",
        "hook": "Aristotle wrote about it in 350 BC. Sigmund Freud spent years searching for the answer. Biologists have studied it for 2,000 years. We still have absolutely no idea where eels come from.",
        "scenes": [
            {
                "narration": "Eels are everywhere. Rivers, lakes, oceans, dinner plates around the world. But here's what no scientist has ever seen: where they come from. No one has ever found an eel with reproductive organs mature enough to observe spawning.",
                "visual": "eel river animation cinematic",
                "color": "#0d0d0d",
                "accent": "#27ae60",
                "duration": 11,
            },
            {
                "narration": "Aristotle believed eels simply appeared from mud. In 1876, the young Sigmund Freud — before he became a psychoanalyst — spent an entire research fellowship dissecting hundreds of eels looking for their testes. He never found them.",
                "visual": "aristotle freud timeline animation",
                "color": "#1a1a2e",
                "accent": "#9b59b6",
                "duration": 12,
            },
            {
                "narration": "What we know: European eels are born somewhere in the Sargasso Sea near Bermuda. They drift on ocean currents to Europe as tiny larvae — a journey of 6,000 kilometres that takes 3 years. Then they live in rivers for up to 85 years as adults.",
                "visual": "ocean current map animation",
                "color": "#16213e",
                "accent": "#3498db",
                "duration": 12,
            },
            {
                "narration": "Then, at some point, they change. They stop eating. Their eyes triple in size. They turn silver. And they swim 6,000 kilometres back to the Sargasso Sea to breed and die.",
                "visual": "eel transformation animation biological",
                "color": "#0d0d0d",
                "accent": "#e74c3c",
                "duration": 10,
            },
            {
                "narration": "Not one single European eel has ever been observed spawning. Not one. We have tracked them approaching the Sargasso Sea. We have never found them there. The eels disappear into deep water and we lose the signal.",
                "visual": "tracking map animation signal lost",
                "color": "#1a1a2e",
                "accent": "#f39c12",
                "duration": 11,
            },
            {
                "narration": "The European eel is now critically endangered. Its population has collapsed by 95 percent in recent decades. We are watching a species disappear. And we don't even know enough about its reproduction to protect it.",
                "visual": "population decline graph animation",
                "color": "#0a0a0a",
                "accent": "#e67e22",
                "duration": 9,
            },
            {
                "narration": "2,000 years of human curiosity. Aristotle, Freud, modern marine biologists with satellite tags. And we still don't know. Some mysteries are not hiding in distant space or deep history. They're swimming in rivers you might have walked past today.",
                "visual": "river reflection dramatic close",
                "color": "#1a1a2e",
                "accent": "#1abc9c",
                "duration": 7,
            },
        ],
        "hashtags": ["#Shorts", "#Nature", "#Eels", "#Biology", "#Mystery"],
        "description": "We've never seen an eel reproduce. Scientists have looked for 2,000 years. #Shorts",
    },
]


def get_story_for_today() -> Dict[str, Any]:
    """Return a different story each day, cycling through all stories."""
    day_index = int(time.time() // 86400)
    story = STORIES[day_index % len(STORIES)]
    return story


def get_story_by_id(story_id: str) -> Dict[str, Any]:
    """Return a specific story by ID, or today's story if not found."""
    for s in STORIES:
        if s["id"] == story_id:
            return s
    return get_story_for_today()
